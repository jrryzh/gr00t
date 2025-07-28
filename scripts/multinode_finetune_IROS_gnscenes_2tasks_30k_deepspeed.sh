#!/bin/bash
#SBATCH -J multinode_finetune_IROS_gnscenes_2tasks_deepspeed
#SBATCH -p efm_p
#SBATCH -N 4
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
export WANDB_RUN_ID="multinode_finetune_IROS_gnscenes_2tasks_30k_deepspeed"

# ---------- 进入项目并激活环境 ----------
cd /mnt/petrelfs/zhangjinyu/code_repo/Isaac-GR00T
source ~/.bashrc
conda activate gr00t

# ---------- Deepspeed 配置 ----------
export DS_CONFIG=./ds_zero2_8n64g_bf16.json   # ← 放在仓库里

# ---------- 启动 ----------
srun torchrun \
  --nnodes $SLURM_NNODES \
  --nproc_per_node 8 \
  --rdzv_backend c10d \
  --rdzv_id $SLURM_JOB_ID \
  --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
  scripts/gr00t_finetune_multinode.py \
    --deepspeed-config $DS_CONFIG \
    --num_gpus 8 \
    --dataset-path data/demonstrations_lerobot/colorful_cups/colorful_cups_split_1_h264 \
    data/demonstrations_lerobot/colorful_cups/colorful_cups_split_2_h264 \
    data/demonstrations_lerobot/colorful_cups/colorful_cups_split_3_h264 \
    data/demonstrations_lerobot/colorful_cups/colorful_cups_split_4_h264 \
    data/demonstrations_lerobot/colorful_cups/colorful_cups_split_5_h264 \
    data/demonstrations_lerobot/select_drink_backup/select_drink_backup_split_1_h264 \
    data/demonstrations_lerobot/select_drink_backup/select_drink_backup_split_2_h264 \
    data/demonstrations_lerobot/select_drink_backup/select_drink_backup_split_3_h264 \
    data/demonstrations_lerobot/select_drink_backup/select_drink_backup_split_4_h264 \
    data/demonstrations_lerobot/select_drink_backup/select_drink_backup_split_5_h264 \
    --output-dir ./logs/multinode_finetune_IROS_gnscenes_2tasks_30k_deepspeed \
    --batch_size 32 \
    --save-steps 5000 \
    --max-steps 100000

