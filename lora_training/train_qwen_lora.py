from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def format_chat(tokenizer, row: dict) -> str:
    messages = row["messages"]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return "\n".join(f"{message['role']}: {message['content']}" for message in messages)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a Qwen LoRA adapter for novel writing.")
    parser.add_argument("--model", default="Qwen/Qwen3-4B", help="Base model id or local path.")
    parser.add_argument("--train-file", type=Path, default=Path("lora_training/data/train.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("lora_training/lora_output/qwen_novel_lora"))
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=0, help="Use only the first N samples. 0 means all samples.")
    parser.add_argument("--max-steps", type=int, default=-1, help="Stop after this many optimizer steps. -1 means full epochs.")
    parser.add_argument("--target-loss", type=float, default=0.0, help="Stop training when logged loss is at or below this value. 0 disables it.")
    parser.add_argument("--target-loss-patience", type=int, default=3, help="Required consecutive logged losses at or below target-loss.")
    parser.add_argument("--target-loss-min-steps", type=int, default=0, help="Ignore target-loss before this optimizer step.")
    parser.add_argument("--save-steps", type=int, default=250, help="Save checkpoints every N steps. 0 saves by epoch only.")
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit loading. Not recommended for 16GB VRAM.")
    args = parser.parse_args()

    rows = load_rows(args.train_file)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]
    if not rows:
        raise SystemExit(
            "No training samples found. Put novel .txt/.md files in "
            "lora_training/source_texts and run prepare_lora_data.bat first."
        )

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForLanguageModeling,
        TrainerCallback,
        Trainer,
        TrainingArguments,
    )

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU was not detected. LoRA training on CPU is not practical.")

    if len(rows) < 20:
        print("[WARN] Dataset has fewer than 20 samples. The adapter will not learn much.")

    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"Samples: {len(rows)}")
    print(f"Base model: {args.model}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if not args.no_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=quantization_config,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    if quantization_config is not None:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    texts = [format_chat(tokenizer, row) for row in rows]
    dataset = Dataset.from_dict({"text": texts})

    def tokenize(batch):
        tokenized = tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_length,
            padding=False,
        )
        tokenized["labels"] = [ids[:] for ids in tokenized["input_ids"]]
        return tokenized

    tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    save_strategy = "steps" if args.save_steps > 0 else "epoch"
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        fp16=True,
        logging_steps=5,
        save_strategy=save_strategy,
        save_steps=args.save_steps if args.save_steps > 0 else 500,
        save_total_limit=2,
        report_to=[],
        optim="paged_adamw_8bit" if not args.no_4bit else "adamw_torch",
        gradient_checkpointing=True,
        remove_unused_columns=False,
    )

    callbacks = []
    if args.target_loss > 0:
        class TargetLossCallback(TrainerCallback):
            def __init__(self, target_loss: float, patience: int, min_steps: int) -> None:
                self.target_loss = target_loss
                self.patience = max(1, patience)
                self.min_steps = max(0, min_steps)
                self.hits = 0

            def on_log(self, training_args, state, control, logs=None, **kwargs):
                logs = logs or {}
                loss = logs.get("loss")
                if loss is None or state.global_step < self.min_steps:
                    return control
                if float(loss) <= self.target_loss:
                    self.hits += 1
                    print(
                        f"[target-loss] hit {self.hits}/{self.patience}: "
                        f"loss={float(loss):.4f} <= {self.target_loss}"
                    )
                    if self.hits >= self.patience:
                        print("[target-loss] target reached. Stopping after this step and saving adapter.")
                        control.should_training_stop = True
                else:
                    self.hits = 0
                return control

        callbacks.append(
            TargetLossCallback(
                target_loss=args.target_loss,
                patience=args.target_loss_patience,
                min_steps=args.target_loss_min_steps,
            )
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=collator,
        callbacks=callbacks,
    )
    trainer.train()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved LoRA adapter to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
