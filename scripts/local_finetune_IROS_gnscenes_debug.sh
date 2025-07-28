#! /bin/bash

source ~/.bashrc
conda activate gr00t            # 或 module load xxx
cd /mnt/petrelfs/zhangjinyu/code_repo/Isaac-GR00T

export WANDB_PROJECT=gr00t_finetune_IROS_gnscenes_9tasks      # 项目名
export WANDB_NAME=debug    # run 名（也就是你想设置的 task name）

python scripts/gr00t_finetune.py \
    --num_gpus 1 \
    --dataset-path data/demonstrations_lerobot/IROS_C_RoboTiq_lerobot_v2/brush_paint_split_1_h264_v2/ \
    --output-dir ./logs/debug \
    --batch_size 16 \
    --save-steps 50000 \
    --max-steps 1000000
