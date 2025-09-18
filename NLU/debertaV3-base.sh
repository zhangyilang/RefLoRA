export CUBLAS_WORKSPACE_CONFIG=:4096:8
export SEED=0
export PYTHONHASHSEED=$SEED
export TASK_NAME=cola
export TQDM_DISABLE=1

torchrun \
    --nnodes=1 --nproc-per-node=1 --standalone \
    run_glue.py \
    --model_name_or_path microsoft/deberta-v3-base \
    --task_name $TASK_NAME \
    --do_train --do_eval \
    --max_seq_length 64 \
    --per_device_train_batch_size 32 \
    --learning_rate 1e-3 \
    --num_train_epochs 5 --warmup_steps 100 \
    --cls_dropout 0.15 --weight_decay 0 \
    --output_dir ./output/$TASK_NAME/ --overwrite_output_dir \
    --logging_steps 20 \
    --eval_strategy epoch \
    --save_strategy epoch \
    --lora_r 8 --lora_alpha 8 \
    --target_modules query_proj,key_proj,value_proj,output.dense,intermediate.dense \
    --seed $SEED \
    --refactor True --use_scalar False --reflora_warmup 100
