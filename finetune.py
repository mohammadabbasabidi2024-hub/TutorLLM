"""LoRA (QLoRA) instruction-tuning on the C-generated dataset.
Usage: python finetune.py
"""
import json

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

import config
from prompts import SYSTEM_PROMPT, build_user_message


def load_examples(path: str, tok) -> Dataset:
    rows = []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        q = r["instruction"] + (f"\n{r['input']}" if r.get("input") else "")
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_message(q)},   # same template as inference
            {"role": "assistant", "content": r["output"]},
        ]
        rows.append({"text": tok.apply_chat_template(messages, tokenize=False)})
    return Dataset.from_list(rows)


def main() -> None:
    tok = AutoTokenizer.from_pretrained(config.BASE_MODEL)
    train_ds = load_examples(config.TRAIN_FILE, tok)

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16)
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                      target_modules="all-linear", task_type="CAUSAL_LM")
    args = SFTConfig(
        output_dir=config.ADAPTER_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        dataset_text_field="text",
        model_init_kwargs={"quantization_config": bnb, "torch_dtype": torch.bfloat16},
        report_to="none",          # set to "wandb" for Weights & Biases tracking
    )
    trainer = SFTTrainer(model=config.BASE_MODEL, args=args,
                         train_dataset=train_ds, peft_config=lora)
    trainer.train()
    trainer.save_model(config.ADAPTER_DIR)
    print("Saved adapter to", config.ADAPTER_DIR)


if __name__ == "__main__":
    main()
