#!/bin/bash
#SBATCH -J gr00t_ft_6tasks
#SBATCH -p efm_p
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:8
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --exclude=SH-IDCA1404-10-140-54-70,SH-IDCA1404-10-140-54-65

# ---------- 通信环境 ----------
export MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n1)
export MASTER_PORT=29500                  # 固定或随机均可
export NCCL_SOCKET_IFNAME=bond0           # 视机房网卡名而定
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# ---- wandb ----
export WANDB_PROJECT="gr00t_finetune_6tasks"

# ---------- 进入项目并激活环境 ----------
cd /mnt/petrelfs/zhangjinyu/code_repo/gr00t-main
source ~/.bashrc
conda activate gr00t

# JINYU: debug
echo "DEBUG  SLURM_GPUS_PER_NODE = ${SLURM_GPUS_PER_NODE}"
echo "DEBUG  SLURM_NTASKS_PER_NODE = ${SLURM_NTASKS_PER_NODE}"


# ---------- 启动 ----------
srun torchrun \
  --nnodes $SLURM_NNODES \
  --nproc_per_node 8 \
  --rdzv_backend c10d \
  --rdzv_id $SLURM_JOB_ID \
  --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
  scripts/gr00t_finetune_multinode.py \
    --num_gpus 8 \
    --dataset-path data/demonstrations_lerobot/bench_v6_6tasks_h264/bench_v6_ball_0_0_h264 \
    data/demonstrations_lerobot/bench_v6_6tasks_h264/bench_v6_ball_0_1_h264 \
    data/demonstrations_lerobot/bench_v6_6tasks_h264/bench_v6_ball_0_2_h264 \
    data/demonstrations_lerobot/bench_v6_6tasks_h264/bench_v6_ball_1_0_h264 \
    data/demonstrations_lerobot/bench_v6_6tasks_h264/bench_v6_ball_1_1_h264 \
    data/demonstrations_lerobot/bench_v6_6tasks_h264/bench_v6_ball_1_2_h264 \
    data/demonstrations_lerobot/bench_v6_6tasks_h264/bench_v6_ball_2_0_h264 \
    data/demonstrations_lerobot/bench_v6_6tasks_h264/bench_v6_ball_2_1_h264 \
    data/demonstrations_lerobot/bench_v6_6tasks_h264/bench_v6_ball_2_2_h264 \
    data/demonstrations_lerobot/bench_v6_6tasks_h264/bench_v6_lighting_devices_11_0_0_h264 \
    data/demonstrations_lerobot/bench_v6_6tasks_h264/bench_v6_lighting_devices_11_0_1_h264 \
    data/demonstrations_lerobot/bench_v6_6tasks_h264/bench_v6_lighting_devices_11_0_2_h264 \
    data/demonstrations_lerobot/bench_v6_6tasks_h264/bench_v6_lighting_devices_11_1_0_h264 \
    data/demonstrations_lerobot/bench_v6_6tasks_h264/bench_v6_lighting_devices_11_1_1_h264 \
    data/demonstrations_lerobot/bench_v6_6tasks_h264/bench_v6_lighting_devices_11_1_2_h264 \
    data/demonstrations_lerobot/bench_v6_6tasks_h264/bench_v6_lighting_devices_11_2_0_h264 \
    data/demonstrations_lerobot/bench_v6_6tasks_h264/bench_v6_lighting_devices_11_2_1_h264 \
    data/demonstrations_lerobot/bench_v6_6tasks_h264/bench_v6_lighting_devices_11_2_2_h264 \
    data/demonstrations_lerobot/bench_v6_6tasks_h264/bench_v6_lighting_devices_6_0_0_h264 \
    data/demonstrations_lerobot/bench_v6_6tasks_h264/bench_v6_lighting_devices_6_0_1_h264 \
    data/demonstrations_lerobot/bench_v6_6tasks_h264/bench_v6_lighting_devices_6_0_2_h264 \
    data/demonstrations_lerobot/bench_v6_6tasks_h264/bench_v6_lighting_devices_6_1_0_h264 \
    data/demonstrations_lerobot/bench_v6_6tasks_h264/bench_v6_lighting_devices_6_1_1_h264 \
    data/demonstrations_lerobot/bench_v6_6tasks_h264/bench_v6_lighting_devices_6_1_2_h264 \
    data/demonstrations_lerobot/bench_v6_6tasks_h264/bench_v6_lighting_devices_6_2_0_h264 \
    data/demonstrations_lerobot/bench_v6_6tasks_h264/bench_v6_lighting_devices_6_2_1_h264 \
    data/demonstrations_lerobot/bench_v6_6tasks_h264/bench_v6_lighting_devices_6_2_2_h264 \
    data/demonstrations_lerobot/bench_v6_6tasks_h264/bench_v6_storage_items_0_0_0_h264 \
    data/demonstrations_lerobot/bench_v6_6tasks_h264/bench_v6_storage_items_0_0_1_h264 \
    data/demonstrations_lerobot/bench_v6_6tasks_h264/bench_v6_storage_items_0_0_2_h264 \
    data/demonstrations_lerobot/bench_v6_6tasks_h264/bench_v6_storage_items_0_1_0_h264 \
    data/demonstrations_lerobot/bench_v6_6tasks_h264/bench_v6_storage_items_0_1_1_h264 \
    data/demonstrations_lerobot/bench_v6_6tasks_h264/bench_v6_storage_items_0_1_2_h264 \
    data/demonstrations_lerobot/bench_v6_6tasks_h264/bench_v6_storage_items_0_2_0_h264 \
    data/demonstrations_lerobot/bench_v6_6tasks_h264/bench_v6_storage_items_0_2_1_h264 \
    data/demonstrations_lerobot/bench_v6_6tasks_h264/bench_v6_storage_items_3_0_0_h264 \
    data/demonstrations_lerobot/bench_v6_6tasks_h264/bench_v6_storage_items_3_0_1_h264 \
    data/demonstrations_lerobot/bench_v6_6tasks_h264/bench_v6_storage_items_3_0_2_h264 \
    data/demonstrations_lerobot/bench_v6_6tasks_h264/bench_v6_storage_items_3_1_0_h264 \
    data/demonstrations_lerobot/bench_v6_6tasks_h264/bench_v6_storage_items_3_1_1_h264 \
    data/demonstrations_lerobot/bench_v6_6tasks_h264/bench_v6_storage_items_3_1_2_h264 \
    data/demonstrations_lerobot/bench_v6_6tasks_h264/bench_v6_storage_items_3_2_0_h264 \
    data/demonstrations_lerobot/bench_v6_6tasks_h264/bench_v6_storage_items_3_2_1_h264 \
    data/demonstrations_lerobot/bench_v6_6tasks_h264/bench_v6_storage_items_3_2_2_h264 \
    data/demonstrations_lerobot/bench_v6_6tasks_h264/bench_v6_storage_items_9_0_0_h264 \
    data/demonstrations_lerobot/bench_v6_6tasks_h264/bench_v6_storage_items_9_0_1_h264 \
    data/demonstrations_lerobot/bench_v6_6tasks_h264/bench_v6_storage_items_9_0_2_h264 \
    data/demonstrations_lerobot/bench_v6_6tasks_h264/bench_v6_storage_items_9_1_0_h264 \
    data/demonstrations_lerobot/bench_v6_6tasks_h264/bench_v6_storage_items_9_1_1_h264 \
    data/demonstrations_lerobot/bench_v6_6tasks_h264/bench_v6_storage_items_9_1_2_h264 \
    data/demonstrations_lerobot/bench_v6_6tasks_h264/bench_v6_storage_items_9_2_0_h264 \
    data/demonstrations_lerobot/bench_v6_6tasks_h264/bench_v6_storage_items_9_2_1_h264 \
    data/demonstrations_lerobot/bench_v6_6tasks_h264/bench_v6_storage_items_9_2_2_h264 \
    --output-dir ./logs/bench_v6_6tasks_lerobot_h264_multinode \
    --save-steps 50000 \
    --max-steps 1000000

