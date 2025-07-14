#!/bin/bash
#SBATCH -p efm_p                  # 替换为你所在的 partition 名
#SBATCH -N 1                      # 1 个节点
#SBATCH --gres=gpu:8              # 使用 8 张 GPU
#SBATCH --cpus-per-task=128        # 每个任务使用的 CPU 核数
#SBATCH --job-name=gr00t_finetune
#SBATCH --output=logs/result.out

# 激活 conda 或加载环境模块
# source ~/.bashrc
# conda activate gr00t

# 执行训练脚本
cd /mnt/petrelfs/zhangjinyu/code_repo/gr00t-main
python scripts/gr00t_finetune.py \
    --dataset-path data/demonstrations_lerobot/bench_v6_ball_h264_absqpos/bench_v6_ball_0_0 \
    data/demonstrations_lerobot/bench_v6_ball_h264_absqpos/bench_v6_ball_0_1 \
    data/demonstrations_lerobot/bench_v6_ball_h264_absqpos/bench_v6_ball_0_2 \
    data/demonstrations_lerobot/bench_v6_ball_h264_absqpos/bench_v6_ball_1_0 \
    data/demonstrations_lerobot/bench_v6_ball_h264_absqpos/bench_v6_ball_1_1 \
    data/demonstrations_lerobot/bench_v6_ball_h264_absqpos/bench_v6_ball_1_2 \
    data/demonstrations_lerobot/bench_v6_ball_h264_absqpos/bench_v6_ball_2_0 \
    data/demonstrations_lerobot/bench_v6_ball_h264_absqpos/bench_v6_ball_2_1 \
    data/demonstrations_lerobot/bench_v6_ball_h264_absqpos/bench_v6_ball_2_2 \
    --num-gpus 8 \
    --output-dir ./logs/bench_v6_ball_lerobot_h264_absqpos \
    --save-steps 10000 \
    --max-steps 200000 

