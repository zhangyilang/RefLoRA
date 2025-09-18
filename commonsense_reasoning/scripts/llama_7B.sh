export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=42
export LORA_R=32
export LORA_ALPHA=64
export OUTPUT_DIR=./output/llama-7b/reflora_r${LORA_R}
export LOG_DIR=./logs/llama-7b/reflora_r${LORA_R}
export TQDM_DISABLE=1

mkdir -p ${OUTPUT_DIR} ${LOG_DIR}

# Fine-tuning (--refactor --use_scalar for RefLoRA-S)
touch ${LOG_DIR}/finetune.log
torchrun \
    --nnodes=1 --nproc-per-node=2 --standalone \
    finetune.py \
    --base_model 'yahma/llama-7b-hf' \
    --data_path 'ft-training_set/commonsense_170k.json' \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 16  --micro_batch_size 8 --num_epochs 2 \
    --learning_rate 2e-4 --cutoff_len 256 --val_set_size 120 \
    --eval_step 1000 --save_step 1000  --adapter_name lora \
    --target_modules '["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]' \
    --lora_r ${LORA_R} --lora_alpha ${LORA_ALPHA} \
    --refactor --reflora_warmup 200 \
    | tee -a ${LOG_DIR}/finetune.log

# Evaluation function
evaluate() {
    local dataset=$1
    touch ${LOG_DIR}/${dataset}.log
    torchrun \
        --nnodes=1 --nproc-per-node=1 --standalone \
        evaluate.py \
        --model LLaMA-7B \
        --adapter LoRA \
        --dataset ${dataset} \
        --base_model 'yahma/llama-7b-hf' \
        --batch_size 1 \
        --lora_weights ${OUTPUT_DIR} \
        | tee -a ${LOG_DIR}/${dataset}.log
}

# Evaluation
for dataset in boolq piqa social_i_qa hellaswag winogrande ARC-Challenge ARC-Easy openbookqa; do
    evaluate ${dataset}
done