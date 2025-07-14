#!/bin/bash
#SBATCH --job-name=manip_sys2_genmanipdata_coco_llavasubset_0707     # name
#SBATCH -p efm_p
#SBATCH -N 6                    # nodes
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=128          # number of cores per tasks
#SBATCH --gres=gpu:8                 # number of gpus
#SBATCH --output=/mnt/petrelfs/zhuyangkun/tmp/slurm_logs/%x-%j.out           # output file name
#SBATCH --error=/mnt/petrelfs/zhuyangkun/tmp/slurm_logs/%x-%j.err
#SBATCH --exclude=SH-IDCA1404-10-140-54-70,SH-IDCA1404-10-140-54-65

# ---------------------------------------------------------------------------
# NCCL 配置
export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_HCA=mlx5_2,mlx5_3
export GPUS_PER_NODE=8
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=$((RANDOM % 101 + 20000))


export TOTAL_GPUS=$((GPUS_PER_NODE * SLURM_NNODES))
echo "Total GPUs: $TOTAL_GPUS"

# ---------------------------------------------------------------------------
# Output configuration
task_name=${SLURM_JOB_NAME}
output_dir=/mnt/petrelfs/zhuyangkun/exp_output/${task_name}
# mkdir -p /mnt/petrelfs/zhuyangkun/exp_output/logs
mkdir -p ${output_dir}
echo backup-cp "$0" "${output_dir}/"
cp "$0" "${output_dir}/" # 备份bash脚本

# log 日志
exec > >(tee -a "$output_dir/job.out") 2> >(tee -a "$output_dir/job.err" >&2)
echo "Logs will be written to $output_dir"


log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}
# --------------------------------------------------------------------------- 
# 打印 集群参数
log "Job name: ${SLURM_JOB_NAME}"
log "Job ID: ${SLURM_JOB_ID}"
log "Partition: ${SLURM_JOB_PARTITION}"
log "Node list: ${SLURM_NODELIST}"
log "Number of CPUs allocated: ${SLURM_CPUS_ON_NODE}"
log "SLURM_PROCID: ${SLURM_PROCID}"
log "SLURM_NNODES: ${SLURM_NNODES}"

# ---------------------------------------------------------------------------
log "Starting ENV initialization..."

# # 激活 bash/python 环境
# source /mnt/petrelfs/zhuyangkun/envs/bashrc

# # VPN
# proxy_on

# WANDB环境
export WANDB_ENTITY="zhuyangkun-southeastern-university"
export WANDB_PROJECT="embodied_sys2"
# export WANDB_MODE="disabled"

# ---------------------------------------------------------------------------
# dataset setting
export rh20t_datasets="rh20t_vla_current_box_train,rh20t_vla_contact_box_train,rh20t_vla_final_box_train,rh20t_vla_traj_qa_train,rh20t_vla_gripper_det_qa_train"
export droid_datasets="droid_vla_contact_box_train,droid_vla_current_box_train,droid_vla_final_box_train,droid_vla_traj_qa_train,droid_vla_gripper_det_qa_train"
export lang_datasets="robovqa_train%11"
export coco_datasets_11="asv2_conversation_en%11,asv2_detailed_description_en%11,asv2_region_captioning_en%11,coco_internvl_longcap_en%11,coco_karpathy_train_567_en%11,coco_neg_gpt4o_en%11,coco_poetry_zh%11,coco_rem_en_zh%11,cocorem_exist_yorn_en%11,cocotextv2_en%11,cocotextv2_gpt4o_en%11,okvqa_en%11,refcoco_grounding_aug_en%11,tallyqa_coco_en%11,toloka_grounding_aug_en%11,vqav2_en%11,vsr_en%11"
export coco_datasets="asv2_conversation_en,asv2_detailed_description_en,asv2_region_captioning_en,coco_internvl_longcap_en,coco_karpathy_train_567_en,coco_neg_gpt4o_en,coco_poetry_zh,coco_rem_en_zh,cocorem_exist_yorn_en,cocotextv2_en,cocotextv2_gpt4o_en,okvqa_en,refcoco_grounding_aug_en,tallyqa_coco_en,toloka_grounding_aug_en,vqav2_en,vsr_en"
# datasets="${rh20t_datasets},${coco_datasets},${lang_datasets}"
export llava_one_vis_datasets="aokvqa_cauldron_llava_format,sharegpt4v_coco,sharegpt4v_knowledge,sharegpt4v_llava,sharegpt4v_sam"

# gsys2_14kv2_datasets: 2*8卡，15000次迭代
export gsys2_14kv2_datasets="gsys2_14kv2_action_plan,gsys2_14kv2_gd_coco_rule,gsys2_14kv2_img_cap_rule,gsys2_14kv2_obj_cap_rule,gsys2_14kv2_obj_attr,gsys2_14kv2_obj_nearby,gsys2_14kv2_obj_senmatic"

# waic_datasets="waic25_overcook_llm"
# waic_datasets="waic25_overcook_vlm"

# export datasets="${coco_datasets}"
export datasets="${gsys2_14kv2_datasets},${coco_datasets},${llava_one_vis_datasets}"


# ---------------------------------------------------------------------------
# train setting
# Pretrain Model configuration
export pretrain_model=/mnt/petrelfs/share/efm_p/zhuyangkun/share_model/Qwen2.5-VL-3B-Instruct  # Using HuggingFace model ID

# DeepSpeed configuration
export deepspeed=./scripts/zero3.json

# Training hyperparameters ############ 经常修改的参数 !!!!!!!!!!!!!!!!!!!!!
export lr=3e-6
export grad_accum_steps=1
export per_device_train_batch_size=4
export per_device_eval_batch_size=16
# ---------------------------------------------------------------------------
# Training entry point

cd /mnt/petrelfs/zhuyangkun/workspace/System2VLA/project/qwen_trainer


srun torchrun \
  --nproc_per_node 8 \
  --nnodes "$SLURM_JOB_NUM_NODES" \
  --rdzv_id "$SLURM_JOB_ID" --rdzv_backend c10d \
  --rdzv_endpoint "$MASTER_ADDR:$MASTER_PORT" \
    qwenvl/train/train_qwen.py \
    --deepspeed ${deepspeed} \
    --model_name_or_path ${pretrain_model} \
    --dataset_use ${datasets} \
    --data_flatten True \
    --tune_mm_vision True \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --bf16 \
    --output_dir ${output_dir} \
    --num_train_epochs 1 \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size ${per_device_eval_batch_size} \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels 12845056 \
    --min_pixels 3136 \
    --eval_strategy no \
    --save_strategy steps \
    --save_steps 3000 \
    --save_total_limit 10 \
    --learning_rate ${lr} \
    --weight_decay 0 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --run_name ${task_name} \
    --report_to wandb