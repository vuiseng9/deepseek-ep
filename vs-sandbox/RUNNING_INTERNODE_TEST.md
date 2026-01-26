# Running DeepEP Internode Test on 2-Node H100 Cluster

This guide explains how to run the internode performance test (`test_internode.py`) on a 2-node cluster with 8xH100 GPUs per node (16 GPUs total).

## Prerequisites

1. **Hardware**: 2 nodes with 8xH100 GPUs each connected via:
   - NVLink for intranode communication
   - InfiniBand/RDMA network for internode communication

2. **Software**: 
   - DeepEP installed with NVSHMEM support (see README.md)
   - PyTorch with NCCL backend
   - Proper RDMA drivers and InfiniBand configuration

## Step 1: Configure Cluster Settings in tests/utils.py

The `init_dist()` function in [tests/utils.py](tests/utils.py) needs to be configured based on your cluster environment. The current implementation uses environment variables for distributed setup.

### Environment Variables Required

You need to set these environment variables for distributed PyTorch:

- `MASTER_ADDR`: IP address of the master node (node 0)
- `MASTER_PORT`: Port for distributed communication (default: 8361)
- `WORLD_SIZE`: Number of nodes (2 in your case)
- `RANK`: Node rank (0 for master node, 1 for second node)

### Current Configuration

The current `init_dist()` function in `tests/utils.py` already supports this setup:

```python
def init_dist(local_rank: int, num_local_ranks: int):
    ip = os.getenv('MASTER_ADDR', '127.0.0.1')
    port = int(os.getenv('MASTER_PORT', '8361'))
    num_nodes = int(os.getenv('WORLD_SIZE', 1))
    node_rank = int(os.getenv('RANK', 0))
    
    # ... rest of the initialization
```

**No code changes are needed in tests/utils.py** if you set the environment variables correctly.

### Alternative: Hardcode Your Cluster Settings (Optional)

If you prefer to hardcode your cluster configuration instead of using environment variables, you can modify `tests/utils.py`:

```python
def init_dist(local_rank: int, num_local_ranks: int):
    # Hardcoded settings for your 2-node cluster
    ip = '192.168.1.100'  # Replace with your master node IP
    port = 8361
    num_nodes = 2  # Fixed for your 2-node setup
    
    # node_rank should still come from environment or launcher
    node_rank = int(os.getenv('RANK', 0))
    
    # ... rest of the code stays the same
```

## Step 2: Understand test_internode.py Arguments

The script accepts the following command-line arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--num-processes` | 8 | Number of GPU processes per node (should be 8 for your setup) |
| `--num-tokens` | 4096 | Number of tokens per batch (matches DeepSeek-V3 training config) |
| `--hidden` | 7168 | Hidden dimension size (matches DeepSeek-V3 config) |
| `--num-topk-groups` | min(num_nodes, 4) | Number of top-k groups (will be 2 for your 2-node setup) |
| `--num-topk` | 8 | Number of top-k experts per token |
| `--num-experts` | 256 | Total number of experts |
| `--pressure-test-mode` | 0 | 0=single run, 1=pressure test w/o benchmarks, 2=pressure test with benchmarks |
| `--test-ll-compatibility` | False | Test compatibility with low-latency kernels |

### For Matching README.md Table (16 EP)

To match the "Internode, Dispatch #EP = 16" row from the README.md table, use these settings:

```bash
--num-processes 8      # 8 GPUs per node
--num-tokens 4096      # Default batch size
--hidden 7168          # DeepSeek-V3 hidden size
--num-topk-groups 2    # 2 nodes
--num-topk 8           # Top-8 routing
--num-experts 16       # 16 total experts for 16 EP
```

## Step 3: Launch the Test

You have two main options for launching the multi-node test:

### Option A: Using torchrun (Recommended)

On **both nodes**, run the following command (adjust the master node IP):

```bash
# On Node 0 (Master):
NCCL_DEBUG=INFO torchrun \
    --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr=MASTER_NODE_IP \
    --master_port=29500 \
    tests/test_internode.py \
    --num-processes 1 \
    --num-tokens 4096 \
    --hidden 7168 \
    --num-topk-groups 2 \
    --num-topk 8 \
    --num-experts 16

# On Node 1:
NCCL_DEBUG=INFO torchrun \
    --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=1 \
    --master_addr=MASTER_NODE_IP \
    --master_port=29500 \
    tests/test_internode.py \
    --num-processes 1 \
    --num-tokens 4096 \
    --hidden 7168 \
    --num-topk-groups 2 \
    --num-topk 8 \
    --num-experts 16
```

**Note**: When using `torchrun`, set `--num-processes 1` because `torchrun` handles the multi-process spawning.

### Option B: Using Environment Variables

Set environment variables and run on each node:

```bash
# IMPORTANT: First, identify active InfiniBand devices on BOTH nodes
# Run this to see active devices:
ibstatus | grep -E "Infiniband device|State" | grep -B1 "Active"

# Set NVSHMEM_HCA_LIST to use your active IB devices (PORT_ACTIVE with MTU 4096)
# CRITICAL: Both nodes MUST use the exact same HCA list
# Use only InfiniBand devices (ibp*), NOT RoCE devices (rocep*)
export NVSHMEM_HCA_LIST=ibp26s0,ibp60s0,ibp77s0,ibp94s0,ibp156s0,ibp188s0,ibp204s0,ibp220s0

# On Node 0 (Master):
export MASTER_ADDR=10.15.25.41
export MASTER_PORT=8361
export WORLD_SIZE=2
export RANK=0

python tests/test_internode.py \
    --num-processes 8 \
    --num-tokens 4096 \
    --hidden 7168 \
    --num-topk-groups 2 \
    --num-topk 8 \
    --num-experts 16

# On Node 1:
# Use the SAME NVSHMEM_HCA_LIST as Node 0
export NVSHMEM_HCA_LIST=ibp26s0,ibp60s0,ibp77s0,ibp94s0,ibp156s0,ibp188s0,ibp204s0,ibp220s0
export MASTER_ADDR=10.15.25.41
export MASTER_PORT=8361
export WORLD_SIZE=2
export RANK=1

python tests/test_internode.py \
    --num-processes 8 \
    --num-tokens 4096 \
    --hidden 7168 \
    --num-topk-groups 2 \
    --num-topk 8 \
    --num-experts 16
```

### Option C: Using a Job Scheduler (SLURM example)

If you're using SLURM, create a job script:

```bash
#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64

# Load modules if needed
# module load cuda/12.3
# module load nccl

export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=8361
export WORLD_SIZE=$SLURM_NNODES

srun bash -c '
export RANK=$SLURM_PROCID
python tests/test_internode.py \
    --num-processes 8 \
    --num-tokens 4096 \
    --hidden 7168 \
    --num-topk-groups 2 \
    --num-topk 8 \
    --num-experts 16
'
```

Submit with: `sbatch run_internode_test.sh`

## Step 4: Interpreting Results

The test will output:

1. **Layout kernel performance**: Time to compute dispatch layout
2. **Testing phase**: Validates correctness with different configurations (FP8/BF16, with/without top-k)
3. **Tuning phase**: Auto-tunes dispatch and combine kernels with different chunk sizes
4. **Best performance results**: Shows optimal configuration with bandwidth numbers

### Expected Output for 16 EP

Look for lines like:

```
[tuning] Best dispatch (FP8): SMs 24, NVL chunk X, RDMA chunk Y: 
    ZZZ us, XX.XX GB/s (RDMA), YY.YY GB/s (NVL)

[tuning] Best combine: SMs 24, NVL chunk X, RDMA chunk Y:
    ZZZ us, XX.XX GB/s (RDMA), YY.YY GB/s (NVL)
```

The RDMA bandwidth numbers should be compared with the README.md table (target: ~43 GB/s for 16 EP).

## Step 5: Testing Other Configurations

### For 32 EP (README.md table):

```bash
--num-experts 32
```

### For 64 EP (README.md table):

```bash
--num-experts 64
```

### For Pressure Testing:

```bash
--pressure-test-mode 1  # Multiple iterations without benchmarks
# or
--pressure-test-mode 2  # Multiple iterations with benchmarks
```

## Troubleshooting

### IBGDA (GPU-Direct RDMA) Initialization Errors

If you see errors like:
```
init failed for transport: IBGDA
cudaHostRegister with IoMemory failed with error=800
ibgda_nic_mem_gpu_map failed
GPU cannot map UAR of device mlx5_X
```

This means NVSHMEM IBGDA (InfiniBand GPU Direct Async) is not properly configured. **IBGDA must be enabled for DeepEP internode kernels to work.**

#### Solution: Configure IBGDA

Follow the instructions in [third-party/README.md](third-party/README.md#2-enable-nvshmem-ibgda-support). You have **two options**:

**Option 1: Configure NVIDIA Driver (Traditional IBGDA - Best Performance)**

1. Edit `/etc/modprobe.d/nvidia.conf`:
   ```bash
   sudo nano /etc/modprobe.d/nvidia.conf
   ```

2. Add this line:
   ```bash
   options nvidia NVreg_EnableStreamMemOPs=1 NVreg_RegistryDwords="PeerMappingOverride=1;"
   ```

3. Update and reboot:
   ```bash
   sudo update-initramfs -u
   sudo reboot
   ```

4. After reboot, verify the settings:
   ```bash
   # Check if settings are applied
   cat /proc/driver/nvidia/params | grep RmEnableStreamMemOPs
   cat /proc/driver/nvidia/params | grep PeerMappingOverride
   ```

**Option 2: Install GDRCopy (CPU-Assisted IBGDA - Easier Setup)**

This has slightly lower performance but doesn't require driver reconfiguration:

1. Download GDRCopy from [here](https://developer.download.nvidia.com/compute/redist/gdrcopy/)

2. Install (for deb-based systems):
   ```bash
   # For CUDA 12.6, use CUDA 12.8 directory
   # Download required packages (use latest version 2.5.1):
   
   cd /tmp
   wget https://developer.download.nvidia.com/compute/redist/gdrcopy/CUDA%2012.8/ubuntu22_04/x64/gdrdrv-dkms_2.5.1-1_amd64.Ubuntu22_04.deb
   wget https://developer.download.nvidia.com/compute/redist/gdrcopy/CUDA%2012.8/ubuntu22_04/x64/libgdrapi_2.5.1-1_amd64.Ubuntu22_04.deb
   
   # Install the essential packages
   sudo dpkg -i gdrdrv-dkms_2.5.1-1_amd64.Ubuntu22_04.deb
   sudo dpkg -i libgdrapi_2.5.1-1_amd64.Ubuntu22_04.deb
   
   # Note: The gdrcopy meta-package is optional and requires gdrcopy-tests
   # The two packages above are sufficient for IBGDA support
   ```

3. Load the kernel module:
   ```bash
   sudo modprobe gdrdrv
   
   # Make it persistent across reboots
   echo "gdrdrv" | sudo tee -a /etc/modules
   ```

4. Verify it's loaded:
   ```bash
   lsmod | grep gdrdrv
   ```

#### Quick Diagnostics

Before configuring IBGDA, check your current setup:

```bash
# Check NVIDIA driver settings
cat /proc/driver/nvidia/params | grep -E "RmEnableStreamMemOPs|PeerMappingOverride"

# Check if GDRCopy is loaded
lsmod | grep gdrdrv

# Check GPU topology
nvidia-smi topo -m

# Verify InfiniBand devices
ibv_devices
ibv_devinfo

# Test RDMA connectivity between nodes
ib_write_bw  # On node 0
ib_write_bw <node0_ip>  # On node 1
```

#### After Configuration

Once you've configured IBGDA using either option above, re-run the test. The warnings should disappear and you should see successful initialization.

### NCCL/RDMA Issues

1. **Check RDMA connectivity**:
   ```bash
   ibstatus
   ibv_devinfo
   ```

2. **Enable NCCL debugging**:
   ```bash
   export NCCL_DEBUG=INFO
   export NCCL_DEBUG_SUBSYS=ALL
   ```

3. **Check NVSHMEM environment**:
   ```bash
   export NVSHMEM_DEBUG=INFO
   ```

### Common Errors

- **"Connection refused"**: Check `MASTER_ADDR` and `MASTER_PORT` are correct
- **"Timeout"**: Firewall may be blocking ports, or nodes can't communicate
- **RDMA errors**: Check InfiniBand/RDMA drivers and network configuration
- **NVSHMEM not found**: Ensure NVSHMEM is installed and `NVSHMEM_DIR` was set during compilation
- **IBGDA warnings**: See "IBGDA (GPU-Direct RDMA) Initialization Errors" section above

### Verify Network Configuration

Before running tests, verify your RDMA network:

```bash
# Check IB devices
ibv_devices

# Test RDMA bandwidth between nodes (run on both nodes)
ib_write_bw -d mlx5_0  # On node 0
ib_write_bw -d mlx5_0 NODE0_IP  # On node 1
```

## Network Optimization Tips (from README.md)

1. **Virtual Lanes**: Set `NVSHMEM_IB_SL` to configure traffic isolation
2. **Adaptive Routing**: Enable for heavy loads, disable for light loads
3. **Congestion Control**: Default is disabled (works well in most cases)

## Auto-tuning for Your Cluster

The test automatically tunes chunk sizes for optimal performance. The best configuration will be printed in the output. You can save these configurations and use them in your production code via:

```python
config = deep_ep.Config(num_sms, nvl_chunk_size, nvl_buffer_size, rdma_chunk_size, rdma_buffer_size)
```

## Summary

1. **No code changes needed** in `tests/utils.py` if using environment variables
2. **Use default arguments** for DeepSeek-V3 config (4096 tokens, 7168 hidden)
3. **Adjust `--num-experts`** to test different EP counts (16, 32, 64, etc.)
4. **Launch on both nodes** using torchrun, env vars, or a job scheduler
5. **Check RDMA bandwidth** in the output and compare with README.md table

The test will validate correctness and auto-tune for best performance on your specific cluster configuration.











# 1. Edit the config file
sudo nano /etc/modprobe.d/nvidia.conf

# 2. Add this line (create the file if it doesn't exist):
options nvidia NVreg_EnableStreamMemOPs=1 NVreg_RegistryDwords="PeerMappingOverride=1;"

# 3. Save and exit (Ctrl+X, then Y, then Enter)

# 4. Update initramfs
sudo update-initramfs -u

# 5. Reboot both nodes
sudo reboot

After reboot, verify:
cat /proc/driver/nvidia/params | grep EnableStreamMemOPs
cat /proc/driver/nvidia/params | grep PeerMappingOverride

If you see the values set to 1, it worked. Then run your test again.

To revert if needed:
sudo nano /etc/modprobe.d/nvidia.conf
# Remove or comment out the line
sudo update-initramfs -u
sudo reboot