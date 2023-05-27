# VisualGPT




````
torchrun --nproc_per_node=4 --nnodes=1 autodl-tmp/train.py \
    --model_name_or_path "togethercomputer/RedPajama-INCITE-Base-7B-v0.1" \
    --sharegpt_data_path autodl-tmp/data/wizard_vicuna_dataset_unfiltered.json \
    --databricks_dolly_15k_path autodl-tmp/data/databricks-dolly-15k.jsonl \
    --evol_instruct_data_path autodl-tmp/data/evol_instruct.json \
    --bf16 True \
    --output_dir "autodl-tmp/output" \
    --cache_dir "autodl-tmp/cache" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 250 \
    --save_total_limit 5 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --logging_steps 1 \
    --deepspeed "autodl-tmp/configs/default_offload_opt_param.json"
````
