#!/bin/bash
#SBATCH -p efm_p                  # 替换为你所在的 partition 名
#SBATCH -N 1                      # 1 个节点
#SBATCH --gres=gpu:8              # 使用 8 张 GPU
#SBATCH --cpus-per-task=128        # 每个任务使用的 CPU 核数
#SBATCH --job-name=gr00t_finetune_6tasks
#SBATCH --output=logs/gr00t_finetune_6tasks.out

# 激活 conda 或加载环境模块
# source ~/.bashrc
# conda activate gr00t

# 执行训练脚本
cd /mnt/petrelfs/zhangjinyu/code_repo/gr00t-main
python scripts/gr00t_finetune.py \
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
    --num-gpus 8 \
    --output-dir ./logs/bench_v6_6tasks_lerobot_h264 \
    --save-steps 50000 \
    --max-steps 1000000 

