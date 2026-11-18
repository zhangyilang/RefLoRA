"""End-to-end fine-tuning throughput and peak GPU memory of LoRA vs. RefLoRA on one GPU.

Each call measures one (model, method) pair in a fresh process, and prints one JSON line:
    python benchmarks/throughput.py --model llama3-8b --method reflora

Weights are randomly initialized from the model config, since neither throughput nor memory
depends on their values; inputs are random tokens.
"""
import argparse
import json
import statistics
import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import (AutoConfig, AutoModelForCausalLM, AutoModelForSequenceClassification,
                          TrainerCallback, TrainingArguments)

from reflora import Refactorer, RefTrainer

# setups follow NLU/debertaV3-base.sh (MRPC) and commonsense_reasoning/scripts/llama3_8B.sh
SETUPS = {
    "deberta-v3-base": dict(
        configs=["microsoft/deberta-v3-base"], auto=AutoModelForSequenceClassification, task_type="SEQ_CLS",
        r=8, alpha=8, targets=["query_proj", "key_proj", "value_proj", "output.dense", "intermediate.dense"],
        batch_size=32, seq_len=128, dtype=torch.float32, bf16=False, lr=4e-4),
    "llama3-8b": dict(
        configs=["meta-llama/Meta-Llama-3-8B", "meta-llama/Llama-3.1-8B"], auto=AutoModelForCausalLM,
        task_type="CAUSAL_LM", r=32, alpha=64, targets=["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"],
        batch_size=8, seq_len=256, dtype=torch.bfloat16, bf16=True, lr=1e-4),
}
METHODS = {
    "lora": None,
    "reflora-per-layer": dict(batched=False),
    "reflora": dict(batched=True),
    "reflora-s-per-layer": dict(use_scalar=True, batched=False),
    "reflora-s": dict(use_scalar=True),
}


class StepTimer(TrainerCallback):
    def __init__(self):
        self.times = []

    def on_step_begin(self, args, state, control, **kwargs):
        torch.cuda.synchronize()
        self.start = time.perf_counter()

    def on_step_end(self, args, state, control, **kwargs):
        torch.cuda.synchronize()
        self.times.append(time.perf_counter() - self.start)


def load_config(setup):
    for name in setup["configs"]:
        try:
            return AutoConfig.from_pretrained(name), name
        except OSError:
            continue
    raise OSError(f"cannot load any of {setup['configs']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=SETUPS, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--skip", type=int, default=20, help="initial steps excluded from timing")
    parser.add_argument("--reflora_warmup", type=int, default=10)
    args = parser.parse_args()
    setup = SETUPS[args.model]

    torch.manual_seed(0)
    config, config_name = load_config(setup)
    with torch.device("cuda"):
        model = setup["auto"].from_config(config, torch_dtype=setup["dtype"])
    model = get_peft_model(model, LoraConfig(task_type=setup["task_type"], r=setup["r"], lora_alpha=setup["alpha"],
                                             target_modules=setup["targets"], lora_dropout=0.0))

    refactorer = None
    if METHODS[args.method] is not None:
        refactorer = Refactorer(model, warmup_steps=args.reflora_warmup, **METHODS[args.method])

    n = setup["batch_size"] * args.steps
    input_ids = torch.randint(0, config.vocab_size, (n, setup["seq_len"]))
    labels = torch.randint(0, 2, (n,)) if setup["task_type"] == "SEQ_CLS" else input_ids
    dataset = [{"input_ids": input_ids[i], "attention_mask": torch.ones_like(input_ids[i]), "labels": labels[i]}
               for i in range(n)]

    training_args = TrainingArguments(
        output_dir="/tmp/reflora-throughput", max_steps=args.steps, per_device_train_batch_size=setup["batch_size"],
        learning_rate=setup["lr"], lr_scheduler_type="constant", bf16=setup["bf16"], optim="adamw_torch",
        logging_steps=10 ** 6, save_strategy="no", report_to="none", disable_tqdm=True, seed=0)
    timer = StepTimer()
    trainer = RefTrainer(model=model, args=training_args, train_dataset=dataset, refactorer=refactorer,
                         callbacks=[timer])
    torch.cuda.reset_peak_memory_stats()
    trainer.train()

    step_time = statistics.median(timer.times[args.skip:])
    print(json.dumps(dict(model=args.model, method=args.method, config=config_name, gpu=torch.cuda.get_device_name(),
                          steps_per_s=1 / step_time, step_ms=step_time * 1e3,
                          peak_mem_mb=torch.cuda.max_memory_allocated() / 2 ** 20)), flush=True)


if __name__ == "__main__":
    main()
