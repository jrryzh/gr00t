#!/bin/bash
#SBATCH -p efm_p                 # 分区
#SBATCH -N 2                     # 节点数（多机）
#SBATCH --ntasks-per-node=1      # 每节点只跑 1 个 launcher
#SBATCH --gres=gpu:8             # 每节点 8 张 GPU
#SBATCH --cpus-per-task=128      # 每个 launcher 使用的 CPU 核
#SBATCH --job-name=gr00t_finetune
#SBATCH --output=logs/%x-%j.out  # %x=job 名，%j=job ID

#################### 1. 软件环境 ####################
source ~/.bashrc
conda activate gr00t            # 或 module load xxx
cd /mnt/petrelfs/zhangjinyu/code_repo/gr00t-main

#################### 2. 分布式参数 ###################
GPUS_PER_NODE=8                 # 与 --gres=gpu:8 一致
MASTER_PORT=29500               # 任选空闲端口
# Slurm 会把所有节点 hostnames 放进 $SLURM_NODELIST
MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n 1)

export NCCL_IB_DISABLE=0        # 视集群网络情况调整
export NCCL_SOCKET_IFNAME=eth0  # 或 ib0、enp175s0 等
export OMP_NUM_THREADS=8        # 合理分配 CPU 线程

echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "SLURM_NNODES=$SLURM_NNODES   SLURM_NODEID=$SLURM_NODEID"

#################### 3. 正式启动 #####################
srun --label torchrun \
        --nnodes=$SLURM_NNODES \
        --node_rank=$SLURM_NODEID \
        --nproc_per_node=$GPUS_PER_NODE \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        scripts/gr00t_finetune.py \
          --num-gpus $GPUS_PER_NODE \
          --dataset-path \
            data/demonstrations_lerobot/bench_v6_ball_h264/bench_v6_ball_0_0_h264 \
            data/demonstrations_lerobot/bench_v6_ball_h264/bench_v6_ball_0_1_h264 \
            data/demonstrations_lerobot/bench_v6_ball_h264/bench_v6_ball_0_2_h264 \
            data/demonstrations_lerobot/bench_v6_ball_h264/bench_v6_ball_1_0_h264 \
            data/demonstrations_lerobot/bench_v6_ball_h264/bench_v6_ball_1_1_h264 \
            data/demonstrations_lerobot/bench_v6_ball_h264/bench_v6_ball_1_2_h264 \
            data/demonstrations_lerobot/bench_v6_ball_h264/bench_v6_ball_2_0_h264 \
            data/demonstrations_lerobot/bench_v6_ball_h264/bench_v6_ball_2_1_h264 \
            data/demonstrations_lerobot/bench_v6_ball_h264/bench_v6_ball_2_2_h264 \
          --output-dir ./logs/bench_v6_ball_lerobot_h264 \
          --save-steps 5000 \
          --max-steps 200000
