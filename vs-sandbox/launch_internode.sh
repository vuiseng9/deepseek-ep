#!/bin/bash
# Launch script for test_internode.py on 2 nodes
# Usage:
#   Node 0 (master): RANK=0 ./launch_internode.sh
#   Node 1:          RANK=1 ./launch_internode.sh

set -e

# =====================================================
# Configuration - MODIFY THESE FOR YOUR CLUSTER
# =====================================================

# Master node IP (must be reachable from all nodes)
export MASTER_ADDR="${MASTER_ADDR:-10.13.114.101}"
export MASTER_PORT="${MASTER_PORT:-29500}"

# Number of nodes
export WORLD_SIZE="${WORLD_SIZE:-2}"

# Node rank (0 for master, 1 for second node) - set via environment
export RANK="${RANK:-0}"

# Network interface for socket-based communication
export IFNAME="${IFNAME:-eth0}"
export GLOO_SOCKET_IFNAME=$IFNAME
export NCCL_SOCKET_IFNAME=$IFNAME

# =====================================================
# InfiniBand HCA Configuration
# Run 'ibstat' to see available HCAs on your system
# =====================================================
export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7}"

# NVSHMEM InfiniBand configuration (must match NCCL_IB_HCA)
export NVSHMEM_HCA_LIST="${NVSHMEM_HCA_LIST:-mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7}"

# GID index for RoCE/InfiniBand - CRITICAL for IB connectivity
# Common values:
#   - 0: Native InfiniBand (use this for IB networks)
#   - 3: RoCEv2 over IPv4 (use this for RoCE/Ethernet networks)
# Run 'show_gids' to find valid indices (non-zero GID addresses)
export NVSHMEM_IB_GID_INDEX="${NVSHMEM_IB_GID_INDEX:-0}"

# =====================================================
# Debug options (enable for troubleshooting)
# =====================================================
# Uncomment these to see detailed debug output:
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=INIT,NET
# export NVSHMEM_DEBUG=TRACE
# export NVSHMEM_DEBUG_SUBSYS=ALL
# export TORCH_DISTRIBUTED_DEBUG=DETAIL

# Less verbose but still useful:
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

# =====================================================
# Additional NVSHMEM settings (usually don't need to change)
# =====================================================
# These are set automatically by deep_ep, but can be overridden:
# export NVSHMEM_DISABLE_P2P=0
# export NVSHMEM_IB_ENABLE_IBGDA=1
# export NVSHMEM_DISABLE_NVLS=1
# export NVSHMEM_DISABLE_MNNVL=1

# =====================================================
# Print configuration
# =====================================================
echo "=============================================="
echo "DeepEP Internode Test Launch Configuration"
echo "=============================================="
echo "RANK:               $RANK"
echo "WORLD_SIZE:         $WORLD_SIZE"
echo "MASTER_ADDR:        $MASTER_ADDR"
echo "MASTER_PORT:        $MASTER_PORT"
echo "IFNAME:             $IFNAME"
echo "NCCL_IB_HCA:        $NCCL_IB_HCA"
echo "NVSHMEM_HCA_LIST:   $NVSHMEM_HCA_LIST"
echo "NVSHMEM_IB_GID_INDEX: $NVSHMEM_IB_GID_INDEX"
echo "=============================================="

# =====================================================
# Run the test
# =====================================================
cd "$(dirname "$0")"

python tests/test_internode.py "$@"
