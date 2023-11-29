#!/bin/bash

export CUDA_VISIBLE_DEVICES="0"
export WANDB_PROJECT="wandb_project_name"

master_port=`shuf -i 12000-30000 -n 1`

lora_r=8
lora_alpha=$(( lora_r * 2 ))
learning_rate="5e-5"
num_epoch=10
batch_size=16
world_size=2

total_batch_size=128
gradient_accumulation_steps=$(( total_batch_size / world_size / batch_size))
total_batch_size=$(( gradient_accumulation_steps * world_size * batch_size ))

run_name="e${num_epoch}_llama2_7b_qvko_r${lora_r}_a${lora_alpha}_lr${learning_rate}_bs${total_batch_size}"

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo "Dir: ${DIR}"

torchrun --nproc_per_node=${world_size} --master_port=${master_port} train.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --data_path ./alpaca_data.json \
    --output_dir ${DIR}/${run_name}/ \
    --run_name  ${run_name}\
    --bf16 True \
    --num_train_epochs ${num_epoch} \
    --per_device_train_batch_size ${batch_size} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --warmup_steps 300 \
    --save_strategy "epoch" \
    --lr_scheduler_type "constant_with_warmup" \
    --save_total_limit 10 \
    --learning_rate ${learning_rate} \
    --model_max_length 512 \
    --logging_steps 8 \
    --tf32 True \
    --ddp_find_unused_parameters False \
    --use_lora True \
    --load_in_4bit True \
    --lora_r ${lora_r} \
    --lora_alpha ${lora_alpha} \
    --lora_target_modules q_proj v_proj k_proj o_proj
