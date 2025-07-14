#!/bin/bash
#SBATCH -J multinode_finetune_IROS_gnscenes_2tasks
#SBATCH -p efm_p
#SBATCH -N 8
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
export WANDB_PROJECT="multinode_finetune_IROS_gnscenes_2tasks"

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
    --dataset-path data/demonstrations_lerobot/colorful_cups/colorful_cups_split_1_h264 \
    data/demonstrations_lerobot/colorful_cups/colorful_cups_split_2_h264 \
    data/demonstrations_lerobot/colorful_cups/colorful_cups_split_3_h264 \
    data/demonstrations_lerobot/colorful_cups/colorful_cups_split_4_h264 \
    data/demonstrations_lerobot/colorful_cups/colorful_cups_split_5_h264 \
    data/demonstrations_lerobot/colorful_cups/colorful_cups_split_6_h264 \
    data/demonstrations_lerobot/colorful_cups/colorful_cups_split_7_h264 \
    data/demonstrations_lerobot/colorful_cups/colorful_cups_split_8_h264 \
    data/demonstrations_lerobot/colorful_cups/colorful_cups_split_9_h264 \
    data/demonstrations_lerobot/colorful_cups/colorful_cups_split_10_h264 \
    data/demonstrations_lerobot/select_drink_backup/select_drink_backup_split_1_h264 \
    data/demonstrations_lerobot/select_drink_backup/select_drink_backup_split_2_h264 \
    data/demonstrations_lerobot/select_drink_backup/select_drink_backup_split_3_h264 \
    data/demonstrations_lerobot/select_drink_backup/select_drink_backup_split_4_h264 \
    data/demonstrations_lerobot/select_drink_backup/select_drink_backup_split_5_h264 \
    data/demonstrations_lerobot/select_drink_backup/select_drink_backup_split_6_h264 \
    data/demonstrations_lerobot/select_drink_backup/select_drink_backup_split_7_h264 \
    data/demonstrations_lerobot/select_drink_backup/select_drink_backup_split_8_h264 \
    data/demonstrations_lerobot/select_drink_backup/select_drink_backup_split_9_h264 \
    data/demonstrations_lerobot/select_drink_backup/select_drink_backup_split_10_h264 \
    --output-dir ./logs/multinode_finetune_IROS_gnscenes_2tasks \
    --batch_size 16 \
    --save-steps 50000 \
    --max-steps 1000000

