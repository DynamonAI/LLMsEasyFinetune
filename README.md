# Alpaca Finetuning

This repo is an unofficial implementation for Stanford's Alpaca models that support full-parameter finetune, LoRA and QLoRA, based on [standford_alpaca](https://github.com/tatsu-lab/stanford_alpaca), [alpaca-lora](https://github.com/tloen/alpaca-lora).

## Overview
For more information about Alpaca amd LLaMA, please read the original documents.

[0]: Alpaca: A Strong, Replicable Instruction-Following Model. Rohan Taori*, Ishaan Gulrajani*, Tianyi Zhang*, Yann Dubois*, Xuechen Li*, Carlos Guestrin, Percy Liang, Tatsunori B. Hashimoto. https://crfm.stanford.edu/2023/03/13/alpaca.html

[1]: LLaMA: Open and Efficient Foundation Language Models. Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, Guillaume Lample. https://arxiv.org/abs/2302.13971v1

## Quick Start

Install the requirements.
```
pip install -r requirements.txt
```
We recommend using `wandb` to monitor the training status.

The `run.sh` is an simple example for finetuning a alpaca with QLoRA.
```
torchrun --nproc_per_node=1 --master_port=27312 train.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --data_path ./alpaca_data.json \
    --output_dir ./example_output_dir \
    --run_name  example_output \
    --bf16 True \
    --num_train_epochs 5 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 8 \
    --warmup_steps 300 \
    --save_strategy "epoch" \
    --lr_scheduler_type "constant_with_warmup" \
    --save_total_limit 10 \
    --learning_rate 5e-5 \
    --model_max_length 512 \
    --logging_steps 8 \
    --tf32 True \
    --ddp_find_unused_parameters False \
    --use_lora True \
    --load_in_4bit True \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_target_modules q_proj v_proj k_proj o_proj
```
We also support `--load_in_8bit True`

You can also use LoRA without quantize by ignoring the flag `load_in_4bit` and `load_in_8bit`.

If you want to do full-parameter finetuning, you can disable `use_lora` or ignore it.

The flag `lora_target_modules` support `q_proj`, `v_proj`, `k_proj`, `o_proj`.


You can use `--resume_from_checkpoint ${path_to_lora_checkpoint}` to load your trained lora.


## Server
We provide `server_lora.py` to run your model on your server, and `client.py` to communicate with the server in interactive mode.

## Author
- [Huawei Lin](https://huaweilin.net/)



