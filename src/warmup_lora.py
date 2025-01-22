import os
import gc
import json
import numpy as np
import random
import argparse
import torch
from tqdm import tqdm
from peft import TaskType, get_peft_model, LoraConfig, PeftModel
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
from transformers import DefaultDataCollator
from typing import Dict, List

import prompt_template
from root_dir_path import ROOT_DIR
from utils import get_model

seed = 42 
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


class TrainingData(Dataset):
    ignored_id = -100

    def __init__(self, origin_dataset, tokenizer, args):
        max_length = args.block_size
        self.dataset = []
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        for data in origin_dataset:
            if not args.with_cot:
                prompt_ids = prompt_template.get_prompt(
                    tokenizer=tokenizer, 
                    question=data["question"], 
                    passages=None, 
                    answer=None,
                    with_cot=args.with_cot
                )
            else:
                prompt_ids = data["prompt_ids"]

            answer = data["answer"]
            if not answer.endswith("."):
                answer += "."
            answer_ids = tokenizer.encode(answer, add_special_tokens=False)
            answer_ids.append(tokenizer.eos_token_id)

            input_ids = prompt_ids + answer_ids
            if len(input_ids) > max_length:
                input_ids = input_ids[:max_length]
            pad_length = max_length - len(input_ids)
            attention_mask = [1] * len(input_ids) + [0] * pad_length
            labels = input_ids + [self.ignored_id] * pad_length
            input_ids += [pad_token_id] * pad_length

            self.dataset.append({
                "input_ids": input_ids, 
                "labels": labels, 
                "attention_mask": attention_mask,
            })
        self.total_len = len(self.dataset)
    
    def __len__(self):
        return self.total_len
    
    def __getitem__(self, idx) -> Dict[str, list]:
        return self.dataset[idx]


class TrainingDataCollator(DefaultDataCollator):
    def __init__(self, tokenizer, device):
        super().__init__()
        self.tokenizer = tokenizer
        self.device = device
    
    def __call__(self, examples: List[Dict[str, list]]) -> Dict[str, torch.Tensor]:
        input_ids, labels, attention_mask = tuple(
            map(lambda x: [example[x] for example in examples], ["input_ids", "labels", "attention_mask"])
        )
        return {
            "input_ids": torch.tensor(input_ids).to(self.device),
            "labels": torch.tensor(labels).to(self.device),
            "attention_mask": torch.tensor(attention_mask).to(self.device),
        }
    

def main(args):
    model, tokenizer, _generation_config = get_model(args.model_name)
    # load train data
    if not args.with_cot:
        data_dir = os.path.join(ROOT_DIR, "warmup", "data", "direct")
        dataset = []
        for name in os.listdir(data_dir):
            tmp = json.load(open(os.path.join(data_dir, name), "r"))
            random.shuffle(tmp)
            dataset += tmp[:1000]
    else:
        with open(os.path.join(ROOT_DIR, "warmup", "data", "cot", "train_data.json"), "r") as fin:
            dataset = json.load(fin)
        last_dataset = None
        for data in dataset:
            if last_dataset != data["from"]:
                prompt_template.get_fewshot(data["from"])
                last_dataset = data["from"]
            data["prompt_ids"] = prompt_template.get_prompt(
                tokenizer=tokenizer, 
                question=data["question"], 
                passages=None, 
                answer=None,
                with_cot=True
            )
            data["answer"] = data["cot"]
    random.shuffle(dataset)

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=['down_proj', 'gate_proj', 'up_proj'],
        inference_mode=False,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0, # !!!
    )
    model = get_peft_model(model, peft_config)
    model.is_parallelizable = True
    model.model_parallel = True

    train_data = TrainingData(dataset, tokenizer, args)
    train_dataloader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.per_device_train_batch_size,
        collate_fn=TrainingDataCollator(tokenizer, model.device),
        shuffle=False,
    )

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(model_parameters, lr=args.learning_rate)
    logging_step = 10
    losses = []
    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.num_train_epochs}")):
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            if step % logging_step == 0:
                print(f"Epoch {epoch+1}, Step {step}, Loss: {loss.item():.4f}")
                losses.append(loss.item())
    save_path = os.path.join(
        ROOT_DIR, 
        "warmup", 
        "lora_base_weight",
        args.model_name,
        "cot" if args.with_cot else "direct", 
    )
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    with open(os.path.join(save_path, "training_config.json"), "w") as fout:
        json.dump(vars(args), fout, indent=4)
    plt.figure(dpi=300)
    plt.plot(losses)
    plt.title("Training Loss Curve")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(save_path, "loss.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--with_cot", action="store_true")
    # Train
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--block_size", type=int, default=3000)
    # LoRA
    parser.add_argument("--lora_rank", type=int, default=2)
    parser.add_argument("--lora_alpha", type=int, default=32)

    args = parser.parse_args()
    main(args)