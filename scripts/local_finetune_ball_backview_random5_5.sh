export WANDB_PROJECT="gr00t_finetune_ball_backview_random5_5"

python scripts/gr00t_finetune.py \
    --dataset-path data/demonstrations_lerobot/bench_v7/bench_v7_ball_0_0_h264 \
    data/demonstrations_lerobot/bench_v7/bench_v7_ball_0_1_h264 \
    data/demonstrations_lerobot/bench_v7/bench_v7_ball_0_2_h264 \
    data/demonstrations_lerobot/bench_v7/bench_v7_ball_1_0_h264 \
    data/demonstrations_lerobot/bench_v7/bench_v7_ball_1_1_h264 \
    data/demonstrations_lerobot/bench_v7/bench_v7_ball_1_2_h264 \
    data/demonstrations_lerobot/bench_v7/bench_v7_ball_2_0_h264 \
    data/demonstrations_lerobot/bench_v7/bench_v7_ball_2_1_h264 \
    data/demonstrations_lerobot/bench_v7/bench_v7_ball_2_2_h264 \
    --num-gpus 8 \
    --output-dir ./logs/bench_v7_ball_backview_random5_5 \
    --save-steps 10000 \
    --max-steps 200000 