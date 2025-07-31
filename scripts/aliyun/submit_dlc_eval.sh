#!/bin/bash

# Check if CKPT_NUM is provided as argument
if [ $# -eq 0 ]; then
    echo "Usage: $0 <CKPT_NUM>"
    echo "Example: $0 50000"
    exit 1
fi


CKPT_DIR=$1
CKPT_NUM=$2
LOG_DIR=$3
CONFIG_PATH=$4
DLC_NAME=$5
NUM_STEPS=$6
OBS_TYPE=$7
echo "Using LOG_DIR: ${LOG_DIR}"
echo "Using CKPT_NUM: ${CKPT_NUM}"

PORT_RECEIVE=$(shuf -i 20000-59999 -n 1)
PORT_SEND=$(shuf -i 20000-59999 -n 1)
MASTER_PORT=$(shuf -i 10000-19999 -n 1)
echo "PORT_RECEIVE: ${PORT_RECEIVE}"
echo "PORT_SEND: ${PORT_SEND}"
echo "MASTER_PORT: ${MASTER_PORT}"


# PORT_RECEIVE=43551
# PORT_SEND=51953
# MASTER_PORT=17138

COMMAND="
. /cpfs/user/zhangjinyu/misc/bashrc && \
. /cpfs/user/zhangjinyu/misc/proxy_on && \
export HOME=/cpfs/user/zhangjinyu && \
conda activate gr00t && \
echo $(which python) && \
cd /cpfs/user/zhangjinyu/code_repo/gr00t && \
python controller/controller.py \
        --receive_port ${PORT_SEND} \
        --send_port ${PORT_RECEIVE} \
        --model_path ${CKPT_DIR}/checkpoint-${CKPT_NUM} \
        --obs_type ${OBS_TYPE} &


sleep 60 &&

/isaac-sim/python.sh /cpfs/user/zhangjinyu/code_repo/GenManip-Sim/eval_V3.py \
    --receive_port ${PORT_RECEIVE} \
    --send_port ${PORT_SEND} \
    --config ${CONFIG_PATH} \
    --num_steps ${NUM_STEPS}
"

/cpfs/user/zhangjinyu/app/dlc submit pytorchjob \
  --name=${DLC_NAME} \
  --workspace_id=270969 \
  --data_sources=d-byhut1kzhlo3slxz3m,d-d49o5g0h2818sw8j1g \
  --resource_id=quotalplclkpgjgv \
  --workers=1 \
  --worker_gpu=1 \
  --worker_cpu=16 \
  --worker_memory=128Gi \
  --worker_image=pj4090acr-registry-vpc.cn-beijing.cr.aliyuncs.com/pj4090/gaoning:genmanip-vnc \
  --command="${COMMAND}" \
  --job_max_running_time_minutes=240 \
  --envs=NCCL_DEBUG=INFO \
  --priority=5