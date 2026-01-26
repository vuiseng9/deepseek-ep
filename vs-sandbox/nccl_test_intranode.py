import argparse
import time
import torch
import torch.distributed as dist

# noinspection PyUnresolvedReferences
import deep_ep
from utils import init_dist, bench, calc_diff, inplace_unique, per_token_cast_to_fp8, per_token_cast_back

# Test compatibility with low latency functions
import test_low_latency


# noinspection PyShadowingNames
def test_main(args: argparse.Namespace, num_sms: int, local_rank: int, num_ranks: int, rank: int, buffer: deep_ep.Buffer,
              group: dist.ProcessGroup):
    # Settings
    num_tokens, hidden = args.num_tokens, args.hidden
    num_topk, num_experts = args.num_topk, args.num_experts

    assert num_experts % num_ranks == 0
    if local_rank == 0:
        print(f'[config] num_tokens={num_tokens}, hidden={hidden}, num_topk={num_topk}', flush=True)

    # Random data
    x = torch.ones((num_tokens, hidden), dtype=torch.bfloat16, device='cuda') * rank
    x_pure_rand = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
    x_e4m3 = per_token_cast_to_fp8(x) if deep_ep.Buffer.is_sm90_compiled() else None
    x_e4m3 = (x_e4m3[0], x_e4m3[1].T.contiguous().T) if x_e4m3 is not None else None
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device='cuda').abs() + 1
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)[1]
    topk_idx = topk_idx.to(deep_ep.topk_idx_t)
    topk_weights = torch.ones((num_tokens, num_topk), dtype=torch.float32, device='cuda') * rank
    topk_weights_pure_rand = torch.randn((num_tokens, num_topk), dtype=torch.float32, device='cuda')
    rank_idx = topk_idx // (num_experts // num_ranks)
    rank_idx = rank_idx.to(torch.int64)
    rank_idx.masked_fill_(topk_idx == -1, -1)
    inplace_unique(rank_idx, num_ranks)

    # Expert meta
    num_tokens_per_expert = torch.zeros((num_experts, ), dtype=torch.int, device='cuda')
    for i in range(num_experts):
        num_tokens_per_expert[i] = (topk_idx == i).sum()
    gbl_num_tokens_per_expert = num_tokens_per_expert.clone()
    dist.all_reduce(gbl_num_tokens_per_expert, group=group)

    # Rank layout meta
    num_tokens_per_rank = torch.empty((num_ranks, ), dtype=torch.int, device='cuda')
    token_idx_in_rank = torch.full((num_ranks, num_tokens), -1, dtype=torch.long, device='cuda')
    for i in range(num_ranks):
        num_tokens_per_rank[i] = (rank_idx == i).sum()
        token_sel = (rank_idx == i).max(dim=-1)[0]
        count = token_sel.sum().item()
        tokens = torch.sort(token_sel.to(torch.int), descending=True)[1]
        tokens[:count] = torch.sort(tokens[:count])[0]
        token_idx_in_rank[i][tokens[:count]] = torch.arange(count, dtype=torch.long, device='cuda')
    token_idx_in_rank = token_idx_in_rank.T.contiguous().to(torch.int)
    is_token_in_rank = token_idx_in_rank >= 0
    gbl_num_tokens_per_rank = num_tokens_per_rank.clone()
    dist.all_reduce(gbl_num_tokens_per_rank, group=group)

    ref_num_tokens_per_rank, _, ref_num_tokens_per_expert, ref_is_token_in_rank, _ = \
        buffer.get_dispatch_layout(topk_idx, num_experts)
    assert torch.allclose(ref_num_tokens_per_rank, num_tokens_per_rank)
    assert torch.allclose(ref_num_tokens_per_expert, num_tokens_per_expert)
    assert torch.allclose(ref_is_token_in_rank, is_token_in_rank)
    t = bench(lambda: buffer.get_dispatch_layout(topk_idx, num_experts))[0]
    if local_rank == 0:
        print(f'[layout] Kernel performance: {t * 1000:.3f} ms', flush=True)
        print('', flush=True)
    group.barrier()
    time.sleep(1)

    # Config
    nvl_buffer_size = 256
    config = deep_ep.Config(num_sms, 8, nvl_buffer_size)

    # Test dispatch
    # noinspection PyShadowingNames
    def check_data(check_x, rank_prefix_matrix):
        assert torch.allclose(check_x.amin(dim=1), check_x.amax(dim=1))
        check_start = 0
        for i in range(num_ranks):
            check_end = rank_prefix_matrix[i][rank].item()
            assert (check_x[check_start:check_end, :].int() - i).sum().item() == 0
            check_start = check_end

    # NCCL dispatch helper function
    def nccl_dispatch(current_x, with_topk=False, topk_weights_data=None):
        # Prepare send/recv counts
        # send_counts[i] = number of tokens this rank sends to rank i
        send_counts = num_tokens_per_rank.cpu().tolist()
        
        # recv_counts[i] = number of tokens this rank receives from rank i
        # Need to gather num_tokens_per_rank from all ranks
        all_num_tokens_per_rank = [torch.zeros_like(num_tokens_per_rank) for _ in range(num_ranks)]
        dist.all_gather(all_num_tokens_per_rank, num_tokens_per_rank, group=group)
        recv_counts = [all_num_tokens_per_rank[i][rank].item() for i in range(num_ranks)]
        
        # Handle FP8 input
        is_fp8 = isinstance(current_x, tuple)
        if is_fp8:
            x_data, x_scale = current_x
            x_data = x_data.reshape(num_tokens, hidden)
        else:
            x_data = current_x
        
        # Prepare send buffer by packing tokens for each rank
        total_send = sum(send_counts)
        nccl_send_buf = torch.empty((total_send, hidden), dtype=x_data.dtype, device='cuda')
        offset = 0
        for i in range(num_ranks):
            mask = is_token_in_rank[:, i]
            count = send_counts[i]
            nccl_send_buf[offset:offset+count] = x_data[mask]
            offset += count
        
        # Prepare receive buffer
        total_recv = recv_counts[rank]
        nccl_recv_buf = torch.empty((total_recv, hidden), dtype=x_data.dtype, device='cuda')
        
        # Perform NCCL all-to-all-v
        dist.all_to_all_single(
            nccl_recv_buf, nccl_send_buf,
            output_split_sizes=recv_counts,
            input_split_sizes=send_counts,
            group=group
        )
        
        # Handle FP8 scale if needed
        recv_x = nccl_recv_buf
        if is_fp8:
            # For FP8, also need to handle scales
            nccl_send_scale = torch.empty((total_send,), dtype=x_scale.dtype, device='cuda')
            offset = 0
            for i in range(num_ranks):
                mask = is_token_in_rank[:, i]
                count = send_counts[i]
                nccl_send_scale[offset:offset+count] = x_scale[mask]
                offset += count
            
            nccl_recv_scale = torch.empty((total_recv,), dtype=x_scale.dtype, device='cuda')
            dist.all_to_all_single(
                nccl_recv_scale, nccl_send_scale,
                output_split_sizes=recv_counts,
                input_split_sizes=send_counts,
                group=group
            )
            recv_x = (nccl_recv_buf, nccl_recv_scale)
        
        # Handle topk if needed
        recv_topk_idx = None
        recv_topk_weights = None
        if with_topk:
            # Pack topk_idx
            nccl_send_topk_idx = torch.empty((total_send, num_topk), dtype=topk_idx.dtype, device='cuda')
            offset = 0
            for i in range(num_ranks):
                mask = is_token_in_rank[:, i]
                count = send_counts[i]
                # Adjust topk_idx to local expert range for target rank
                local_topk = topk_idx[mask].clone()
                local_topk = torch.where(local_topk >= 0, local_topk % (num_experts // num_ranks), local_topk)
                nccl_send_topk_idx[offset:offset+count] = local_topk
                offset += count
            
            recv_topk_idx = torch.empty((total_recv, num_topk), dtype=topk_idx.dtype, device='cuda')
            dist.all_to_all_single(
                recv_topk_idx.view(-1), nccl_send_topk_idx.view(-1),
                output_split_sizes=[c * num_topk for c in recv_counts],
                input_split_sizes=[c * num_topk for c in send_counts],
                group=group
            )
            
            # Pack topk_weights
            nccl_send_topk_weights = torch.empty((total_send, num_topk), dtype=topk_weights_data.dtype, device='cuda')
            offset = 0
            for i in range(num_ranks):
                mask = is_token_in_rank[:, i]
                count = send_counts[i]
                nccl_send_topk_weights[offset:offset+count] = topk_weights_data[mask]
                offset += count
            
            recv_topk_weights = torch.empty((total_recv, num_topk), dtype=topk_weights_data.dtype, device='cuda')
            dist.all_to_all_single(
                recv_topk_weights.view(-1), nccl_send_topk_weights.view(-1),
                output_split_sizes=[c * num_topk for c in recv_counts],
                input_split_sizes=[c * num_topk for c in send_counts],
                group=group
            )
        
        # Build rank_prefix_matrix for compatibility
        rank_prefix_matrix = torch.zeros((num_ranks, num_ranks), dtype=torch.int, device='cuda')
        for i in range(num_ranks):
            rank_prefix_matrix[:, i] = torch.tensor([sum(gbl_num_tokens_per_rank[:j+1].cpu().tolist()) for j in range(num_ranks)], device='cuda')
        
        # Build recv_num_tokens_per_expert_list
        recv_num_tokens_per_expert_list = gbl_num_tokens_per_expert.view(num_ranks, -1)[rank].cpu().tolist()
        
        handle = (rank_prefix_matrix, None)
        
        return recv_x, recv_topk_idx, recv_topk_weights, recv_num_tokens_per_expert_list, handle

    # NCCL combine helper function
    def nccl_combine(recv_x, handle, with_topk=False, recv_topk_weights=None):
        rank_prefix_matrix = handle[0]
        
        # Prepare send/recv counts (reverse of dispatch)
        # send_counts[i] = number of tokens this rank sends to rank i (reverse of dispatch recv)
        # recv_counts[i] = number of tokens this rank receives from rank i (reverse of dispatch send)
        all_num_tokens_per_rank = [torch.zeros_like(num_tokens_per_rank) for _ in range(num_ranks)]
        dist.all_gather(all_num_tokens_per_rank, num_tokens_per_rank, group=group)
        send_counts = [all_num_tokens_per_rank[i][rank].item() for i in range(num_ranks)]
        recv_counts = num_tokens_per_rank.cpu().tolist()
        
        # Send buffer is the received data
        total_send = sum(send_counts)
        nccl_send_buf = recv_x.reshape(total_send, hidden)
        
        # Receive buffer for combined result
        total_recv = sum(recv_counts)
        nccl_recv_buf = torch.empty((total_recv, hidden), dtype=nccl_send_buf.dtype, device='cuda')
        
        # Perform NCCL all-to-all-v (reverse direction)
        dist.all_to_all_single(
            nccl_recv_buf, nccl_send_buf,
            output_split_sizes=recv_counts,
            input_split_sizes=send_counts,
            group=group
        )
        
        # Unpack received data back to original token positions
        combined_x = torch.zeros((num_tokens, hidden), dtype=nccl_recv_buf.dtype, device='cuda')
        offset = 0
        for i in range(num_ranks):
            mask = is_token_in_rank[:, i]
            count = recv_counts[i]
            combined_x[mask] += nccl_recv_buf[offset:offset+count]
            offset += count
        
        combined_topk_weights = None
        if with_topk:
            # Combine topk_weights
            nccl_send_topk_weights = recv_topk_weights.reshape(total_send, num_topk)
            nccl_recv_topk_weights = torch.empty((total_recv, num_topk), dtype=nccl_send_topk_weights.dtype, device='cuda')
            
            dist.all_to_all_single(
                nccl_recv_topk_weights.view(-1), nccl_send_topk_weights.view(-1),
                output_split_sizes=[c * num_topk for c in recv_counts],
                input_split_sizes=[c * num_topk for c in send_counts],
                group=group
            )
            
            combined_topk_weights = torch.zeros((num_tokens, num_topk), dtype=nccl_recv_topk_weights.dtype, device='cuda')
            offset = 0
            for i in range(num_ranks):
                mask = is_token_in_rank[:, i]
                count = recv_counts[i]
                combined_topk_weights[mask] += nccl_recv_topk_weights[offset:offset+count]
                offset += count
        
        return combined_x, combined_topk_weights

    for previous_mode in (False, ):  # Skip previous_mode=True for NCCL
        for async_mode in (False, ):  # Skip async_mode=True for NCCL
            for current_x in filter(lambda elem: elem is not None, (x_pure_rand, x, x_e4m3)):
                for with_topk in (False, True):
                    if local_rank == 0:
                        print(
                            f'[testing] Running with {"FP8" if isinstance(current_x, tuple) else "BF16"}, {"with" if with_topk else "without"} top-k (async={async_mode}, previous={previous_mode}) ...',
                            flush=True,
                            end='')
                    
                    topk_weights_data = topk_weights_pure_rand if current_x is x_pure_rand else topk_weights
                    recv_x, recv_topk_idx, recv_topk_weights, recv_num_tokens_per_expert_list, handle = nccl_dispatch(
                        current_x, with_topk=with_topk, topk_weights_data=topk_weights_data
                    )
                    recv_x = per_token_cast_back(*recv_x) if isinstance(recv_x, tuple) else recv_x

                    # Checks
                    rank_prefix_matrix = handle[0]
                    assert gbl_num_tokens_per_rank[rank].item() == recv_x.size(
                        0), f'{gbl_num_tokens_per_rank[rank].item()} != {recv_x.size(0)}'
                    assert gbl_num_tokens_per_expert.view(num_ranks, -1)[rank].tolist() == recv_num_tokens_per_expert_list
                    if current_x is not x_pure_rand:
                        check_data(recv_x, rank_prefix_matrix)
                    recv_topk_weights_clone = None
                    if with_topk:
                        # Check `topk_idx`
                        assert (recv_topk_idx.eq(-1) |
                                ((recv_topk_idx >= 0) &
                                 (recv_topk_idx < (num_experts // num_ranks)))).sum().item() == recv_topk_idx.numel()
                        for i, count in enumerate(recv_num_tokens_per_expert_list):
                            assert recv_topk_idx.eq(i).sum().item() == count

                        # Check `topk_weights`
                        recv_topk_weights_clone = recv_topk_weights.clone()
                        if current_x is not x_pure_rand:
                            recv_topk_weights[recv_topk_idx.eq(-1)] = recv_topk_weights.amax(
                                dim=1, keepdim=True).expand_as(recv_topk_weights)[recv_topk_idx.eq(-1)]
                            check_data(recv_topk_weights, rank_prefix_matrix)

                    # Skip num_worst_tokens test for NCCL

                    # Skip cached dispatch test for NCCL

                    # Test combine
                    combined_x, combined_topk_weights = nccl_combine(
                        recv_x, handle, with_topk=with_topk, recv_topk_weights=recv_topk_weights
                    )
                    check_x = combined_x.float() / is_token_in_rank.sum(dim=1).unsqueeze(1)
                    ref_x = x_pure_rand if current_x is x_pure_rand else x
                    assert calc_diff(check_x, ref_x) < 5e-6
                    if with_topk:
                        check_topk_weights = combined_topk_weights if (current_x
                                                                       is x_pure_rand) else (combined_topk_weights /
                                                                                             is_token_in_rank.sum(dim=1).unsqueeze(1))
                        ref_topk_weights = topk_weights_pure_rand if current_x is x_pure_rand else topk_weights
                        assert calc_diff(check_topk_weights, ref_topk_weights) < 1e-9

                    # For later tuning
                    dispatch_bf16_nvl_recv_bytes = recv_x.numel() * 2
                    combine_bf16_nvl_send_bytes = dispatch_bf16_nvl_recv_bytes

                    if local_rank == 0:
                        print(' passed', flush=True)
    if local_rank == 0:
        print('', flush=True)

    # Tune dispatch performance with NCCL
    best_dispatch_results = None
    fp8_factor = (1 + 4 / 128) / 2
    for current_x in filter(lambda elem: elem is not None, (x_e4m3, x)):
        best_time, best_results = 1e10, None
        nvl_recv_bytes = (dispatch_bf16_nvl_recv_bytes * fp8_factor) if isinstance(current_x, tuple) else dispatch_bf16_nvl_recv_bytes
        
        # NCCL doesn't have tunable parameters like custom kernel
        recv_x, _, _, _, handle = nccl_dispatch(current_x, with_topk=False, topk_weights_data=None)
        t = bench(lambda: nccl_dispatch(current_x, with_topk=False, topk_weights_data=None))[0]
        if local_rank == 0:
            print(
                f'[tuning] NCCL dispatch ({"FP8" if isinstance(current_x, tuple) else "BF16"}): '
                f'{nvl_recv_bytes / 1e9 / t:.2f} GB/s (NVL), {t * 1e6:.2f} us',
                flush=True)
            print('', flush=True)

    # Tune combine performance with NCCL
    recv_x, _, _, _, handle = nccl_dispatch(x, with_topk=False, topk_weights_data=None)
    t = bench(lambda: nccl_combine(recv_x, handle, with_topk=False, recv_topk_weights=None))[0]
    if local_rank == 0:
        print(
            f'[tuning] NCCL combine: {combine_bf16_nvl_send_bytes / 1e9 / t:.2f} GB/s (NVL), {t * 1e6:.2f} us',
            flush=True)
        print('', flush=True)


# noinspection PyUnboundLocalVariable,PyShadowingNames
def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    test_ll_compatibility, num_rdma_bytes = False, 0
    if test_ll_compatibility:
        ll_num_tokens, ll_hidden, ll_num_experts, ll_num_topk = 16, 5120, 256, 9
        num_rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(ll_num_tokens, ll_hidden, num_ranks, ll_num_experts)

    buffer = deep_ep.Buffer(group,
                            int(2e9),
                            num_rdma_bytes,
                            low_latency_mode=test_ll_compatibility,
                            num_qps_per_rank=(ll_num_experts // num_ranks if test_ll_compatibility else 1),
                            explicitly_destroy=True,
                            allow_mnnvl=args.allow_mnnvl,
                            use_fabric=args.use_fabric)
    torch.manual_seed(rank)

    for i in (24, ):
        test_main(args, i, local_rank, num_ranks, rank, buffer, group)
        if local_rank == 0:
            print('', flush=True)

    # Test compatibility with low latency functions
    if test_ll_compatibility:
        buffer.clean_low_latency_buffer(ll_num_tokens, ll_hidden, ll_num_experts)
        test_low_latency.test_main(ll_num_tokens, ll_hidden, ll_num_experts, ll_num_topk, rank, num_ranks, group, buffer, seed=1)

    # Destroy the buffer runtime and communication group
    buffer.destroy()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test intranode EP kernels with NCCL')
    parser.add_argument('--num-processes', type=int, default=8, help='Number of processes to spawn (default: 8)')
    parser.add_argument('--num-tokens', type=int, default=4096, help='Number of tokens (default: 4096)')
    parser.add_argument('--hidden', type=int, default=7168, help='Hidden dimension size (default: 7168)')
    parser.add_argument('--num-topk', type=int, default=8, help='Number of top-k experts (default: 8)')
    parser.add_argument('--num-experts', type=int, default=256, help='Number of experts (default: 256)')
    parser.add_argument('--allow-mnnvl', action="store_true", help='Enable MNNVL support')
    parser.add_argument('--use-fabric', action="store_true", help='Enable fabric mode')
    args = parser.parse_args()

    num_processes = args.num_processes
    torch.multiprocessing.spawn(test_loop, args=(num_processes, args), nprocs=num_processes)
