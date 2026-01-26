10.13.114.101
10.13.114.102

2: eth0: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 9000 qdisc mq state UP group default qlen 1000
    link/ether 52:54:00:3b:d6:b4 brd ff:ff:ff:ff:ff:ff
    altname enp1s0
    inet 10.13.114.102/24 brd 10.13.114.255 scope global eth0
       valid_lft forever preferred_lft forever
    inet6 fe80::5054:ff:fe3b:d6b4/64 scope link 
       valid_lft forever preferred_lft forever

ibv_devices
ls /sys/class/infiniband

# change node id accordingly
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET

export nodeid=1
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7

export IFNAME=eth0
export MASTER_ADDR=10.13.114.101
export MASTER_PORT=29500
export GLOO_SOCKET_IFNAME=$IFNAME
export NCCL_SOCKET_IFNAME=$IFNAME

torchrun --nnodes=2 --nproc-per-node=8 --node-rank=$nodeid \
  --master-addr=$MASTER_ADDR --master-port=$MASTER_PORT smoke_nccl.py


# deep ep

export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7

export IFNAME=eth0
export MASTER_ADDR=10.13.114.101
export MASTER_PORT=29500
export GLOO_SOCKET_IFNAME=$IFNAME
export NCCL_SOCKET_IFNAME=$IFNAME

export RANK=1
export WORLD_SIZE=2

python tests/test_internode.py



# opus node 1

export NVSHMEM_DEBUG=TRACE
kill -9 $(pgrep -f "multiprocessing") 2>/dev/null
kill -9 $(pgrep -f "test_internode") 2>/dev/null

export NCCL_NET_GDR_LEVEL=0
export NVSHMEM_DISABLE_CUDA_VMM=1

export RANK=1
unset NCCL_IB_PKEY
export NCCL_DEBUG=INFO
export PYTHONUNBUFFERED=1

export MASTER_PORT=29500
export MASTER_ADDR=10.13.114.101
export WORLD_SIZE=2
export IFNAME=eth0
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7
export NVSHMEM_HCA_LIST=$NCCL_IB_HCA
export NVSHMEM_IB_GID_INDEX=0
export GLOO_SOCKET_IFNAME=$IFNAME
export NCCL_SOCKET_IFNAME=$IFNAME

python tests/test_internode.py --pressure-test-mode 1  2>&1 | tee node${RANK}.log



# NEW

export RANK=1
# unset NCCL_IB_PKEY
# export NCCL_DEBUG=INFO
# export PYTHONUNBUFFERED=1

export MASTER_PORT=29500
export MASTER_ADDR=10.13.113.101
export WORLD_SIZE=2
export IFNAME=eth0
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7
export NVSHMEM_HCA_LIST=$NCCL_IB_HCA
export NVSHMEM_IB_GID_INDEX=0
export GLOO_SOCKET_IFNAME=$IFNAME
export NCCL_SOCKET_IFNAME=$IFNAME

python tests/test_internode.py 

--pressure-test-mode 1  2>&1 | tee node${RANK}.log