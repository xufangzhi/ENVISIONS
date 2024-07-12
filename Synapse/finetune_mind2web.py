from datetime import datetime
import os
import logging
import sys
import argparse
import torch
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
)
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_from_disk

logger = logging.getLogger("synapse")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--top_k_elements", type=int, default=50)
    parser.add_argument("--base_model", type=str)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--lora_dir", type=str, default=None)
    parser.add_argument("--no_trajectory", action="store_true", default=False)

    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()
    wandb_project = "naive" if args.no_trajectory else "trajectory"
    os.environ["WANDB_PROJECT"] = wandb_project

    if args.no_trajectory:
        per_device_train_batch_size = 1
    else:
        per_device_train_batch_size = 1

    batch_size = 128
    gradient_accumulation_steps = batch_size // per_device_train_batch_size

    # load model
    device_map = "auto"
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        cache_dir=args.cache_dir,
        use_flash_attention_2=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, cache_dir=args.cache_dir)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    def tokenize_prompt(data_point):
        tokenized_full_prompt = tokenizer(
            data_point["input"] + data_point["output"],
            padding=False,
            return_tensors=None,
        )
        tokenized_full_prompt["labels"] = tokenized_full_prompt["input_ids"].copy()
        tokenized_user_prompt = tokenizer(
            data_point["input"],
            padding=False,
            return_tensors=None,
        )
        user_prompt_len = len(tokenized_user_prompt["input_ids"])
        tokenized_full_prompt["labels"] = [
            -100
        ] * user_prompt_len + tokenized_full_prompt["labels"][user_prompt_len:]

        return tokenized_full_prompt

    peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)

    model.print_trainable_parameters()

    if torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    # load dataset
    train_data = load_from_disk(
        os.path.join(
            args.data_dir,
            f"train/{'naive' if args.no_trajectory else 'trajectory'}_top{args.top_k_elements}",
        )
    )
    train_data = train_data.map(
        tokenize_prompt, remove_columns=train_data.column_names, num_proc=8
    ).shuffle()
    val_data = None

    # Training
    cur_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
    output_dir = (
        args.lora_dir + f"-{'naive' if args.no_trajectory else 'trajectory'}-{cur_time}"
    )
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=4,
            learning_rate=3e-4,
            bf16=True,
            logging_steps=5,
            optim="adamw_torch",
            evaluation_strategy="steps" if val_data is not None else "no",
            save_strategy="steps",
            eval_steps=50 if val_data is not None else None,
            save_steps=50,
            output_dir=output_dir,
            load_best_model_at_end=False,
            ddp_find_unused_parameters=None,
            group_by_length=True,
            report_to="wandb",
            run_name=f"{args.base_model}-{cur_time}",
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )

    model.config.use_cache = False

    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
    ).__get__(model, type(model))

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train()

    model.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
