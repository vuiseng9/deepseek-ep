FROM nvcr.io/nvidia/pytorch:25.06-py3

SHELL ["/bin/bash", "-lc"]
USER root
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
        bash-completion \
        git-lfs \
        wget \
        build-essential \
        gdb && \
    rm -rf /var/lib/apt/lists/*

# Configure bash
RUN cat << 'EOF' >> ~/.bashrc
if [ -f /etc/bash_completion ]; then
  . /etc/bash_completion
fi
PS1='${debian_chroot:+($debian_chroot)}\[\033[01;32m\]\u@\h\[\033[00m\] : \[\033[01;34m\]\w\[\033[01;31m\]$(__git_ps1)\[\033[00m\]\n$ '
EOF

# Fix mlx5 library symlink
RUN ln -s /usr/lib/x86_64-linux-gnu/libmlx5.so.1 /usr/lib/x86_64-linux-gnu/libmlx5.so

# Install NVSHMEM
RUN cd /usr/local && \
    wget https://developer.download.nvidia.com/compute/nvshmem/redist/libnvshmem/linux-x86_64/libnvshmem-linux-x86_64-3.5.19_cuda12-archive.tar.xz && \
    tar -xf libnvshmem-linux-x86_64-3.5.19_cuda12-archive.tar.xz && \
    rm libnvshmem-linux-x86_64-3.5.19_cuda12-archive.tar.xz && \
    mv libnvshmem-linux-x86_64-3.5.19_cuda12-archive nvshmem-3.5.19-cu12 && \
    ln -s nvshmem-3.5.19-cu12 nvshmem

# Set NVSHMEM environment variables
ENV NVSHMEM_DIR=/usr/local/nvshmem \
    LD_LIBRARY_PATH="${NVSHMEM_DIR}/lib:${LD_LIBRARY_PATH}" \
    PATH="${NVSHMEM_DIR}/bin:${PATH}"

RUN echo 'export NVSHMEM_DIR=/usr/local/nvshmem' >> ~/.bashrc && \
    echo 'export LD_LIBRARY_PATH="${NVSHMEM_DIR}/lib:$LD_LIBRARY_PATH"' >> ~/.bashrc && \
    echo 'export PATH="${NVSHMEM_DIR}/bin:$PATH"' >> ~/.bashrc

# Install deepseek-ep
RUN git clone https://github.com/vuiseng9/deepseek-ep -b 260120-mlmoe /workspace/deepseek-ep && \
    cd /workspace/deepseek-ep && \
    TORCH_CUDA_ARCH_LIST=9.0 python setup.py install

# Install megatron-lm and prepare data
RUN git clone https://github.com/vuiseng9/megatron-lm -b 260120-2f3fa0-deepep /workspace/megatron-lm && \
    cd /workspace/megatron-lm/examples/gpt3 && \
    make install-dependencies && \
    make prepare-ds-openwebtext-10k

WORKDIR /workspace