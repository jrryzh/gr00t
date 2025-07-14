#!/bin/bash

# CKPT_DIR=/cpfs/user/zhangjinyu/code_repo/gr00t/checkpoint
# LOG_DIR=/cpfs/user/zhangjinyu/code_repo/gr00t/logs${CKPT_NUM}
"""
在这里指定ckpt 在submit_dlc_eval.sh中指定config_name
"""

## base scene
# for CKPT_NUM in 150000
# do
#     CKPT_DIR=/cpfs/user/zhangjinyu/code_repo/gr00t/checkpoint/ball_h264_deltapos
#     LOG_DIR=/cpfs/user/zhangjinyu/code_repo/gr00t/logs${CKPT_NUM}
#     CONFIG_PATH=/shared/smartbot/zhangjinyu/genmanip/configs/bench/bench_v6_simple_ball.yaml
#     DLC_NAME=Eval_Gr00t_ball_Scene_${CKPT_NUM}
#     for i in {1..20}
#     do
#         echo "Running iteration $i of 20"
#         scripts/aliyun/submit_dlc_eval.sh ${CKPT_DIR} ${CKPT_NUM} ${LOG_DIR} ${CONFIG_PATH} ${DLC_NAME}
#         echo "Completed iteration $i"
#         echo "----------------------------------------"
#     done
# done

# # longrange
# for CKPT_NUM in 200000
# do
#     CKPT_DIR=/cpfs/user/zhangjinyu/code_repo/gr00t/checkpoint/ball_h264_longrange
#     LOG_DIR=/cpfs/user/zhangjinyu/code_repo/gr00t/ball_h264_longrange_logs${CKPT_NUM}
#     CONFIG_PATH=/cpfs/user/zhangjinyu/code_repo/GenManip-Sim/configs/tasks/bench/bench_v6_simple_longrange/athletic_balls_final_simple_long_range.yml
#     DLC_NAME=Eval_Gr00t_longrange_Scene_${CKPT_NUM}
#     NUM_STEPS=1500
#     for i in {1..20}
#     do
#         echo "Running iteration $i of 20"
#         scripts/aliyun/submit_dlc_eval.sh ${CKPT_DIR} ${CKPT_NUM} ${LOG_DIR} ${CONFIG_PATH} ${DLC_NAME} ${NUM_STEPS}
#         echo "Completed iteration $i"
#         echo "----------------------------------------"
#     done
# done

## 使用sapien
# USE_SAPIEN=true

## six tasks
# for CKPT_NUM in 750000
# do
#     CKPT_DIR=/cpfs/user/zhangjinyu/code_repo/gr00t/checkpoint/6tasks_h264_deltapos
#     LOG_DIR=/cpfs/user/zhangjinyu/code_repo/gr00t/6tasks_h264_deltapos_logs${CKPT_NUM}
#     # CONFIG_PATH=/shared/smartbot/zhangjinyu/genmanip/configs/bench/bench_v6_simple_0630_for_fangjing_qwenact.yaml
#     CONFIG_PATH=/cpfs/user/zhangjinyu/code_repo/gr00t/configs/eval/Bench/bench_v6_simple-sapien.yaml
#     DLC_NAME=Eval_Gr00t_6tasks_Scene_${CKPT_NUM}_sapien
#     NUM_STEPS=500
#     for i in {1..30}
#     do
#         echo "Running iteration $i of 30"
#         if [ "$USE_SAPIEN" = true ]; then
#             scripts/aliyun/submit_dlc_eval_sapien.sh ${CKPT_DIR} ${CKPT_NUM} ${LOG_DIR} ${CONFIG_PATH} ${DLC_NAME} ${NUM_STEPS}
#         else
#             scripts/aliyun/submit_dlc_eval.sh ${CKPT_DIR} ${CKPT_NUM} ${LOG_DIR} ${CONFIG_PATH} ${DLC_NAME} ${NUM_STEPS}
#         fi
#         echo "Completed iteration $i"
#         echo "----------------------------------------"
#     done
# done


##  ----------------------------BENCH_V7_TASKS----------------------------
USE_SAPIEN=true

for CKPT_NUM in 70000
do
    CKPT_DIR=/cpfs/user/zhangjinyu/code_repo/gr00t/checkpoint/bench_v7_ball_backview_random5_5
    LOG_DIR=/cpfs/user/zhangjinyu/code_repo/gr00t/bench_v7_ball_backview_random5_5${CKPT_NUM}
    # CONFIG_PATH=/shared/smartbot/zhangjinyu/genmanip/configs/bench/bench_v6_simple_0630_for_fangjing_qwenact.yaml
    # CONFIG_PATH=/shared/smartbot/zhangjinyu/genmanip/configs/bench/bench_v6_simple_novelscene.yaml
    CONFIG_PATH=/shared/smartbot/zhangjinyu/genmanip/configs/bench/bench_v7_ball-sapien.yaml
    # DLC_NAME=Eval_Gr00t_novel_Scene_${CKPT_NUM}
    DLC_NAME=Eval_bench_v7_ball_${CKPT_NUM}_sapien
    OBS_TYPE=obs_camera # fixed backview
    NUM_STEPS=500
    for i in {1..30}
    do
        echo "Running iteration $i of 30"
        if [ "$USE_SAPIEN" = true ]; then
            scripts/aliyun/submit_dlc_eval_sapien.sh ${CKPT_DIR} ${CKPT_NUM} ${LOG_DIR} ${CONFIG_PATH} ${DLC_NAME} ${NUM_STEPS} ${OBS_TYPE}
        else
            scripts/aliyun/submit_dlc_eval.sh ${CKPT_DIR} ${CKPT_NUM} ${LOG_DIR} ${CONFIG_PATH} ${DLC_NAME} ${NUM_STEPS} ${OBS_TYPE}
        fi
        echo "Completed iteration $i"
        echo "----------------------------------------"
    done
done