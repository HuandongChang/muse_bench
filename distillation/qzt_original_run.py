# run_distillation.py

import torch
# import wandb
from transformers import Trainer, TrainingArguments
from qzt_original import DistillTrainer  # Import your custom DistillTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
# import bitsandbytes as bnb
from peft import get_peft_model, LoraConfig
import os

os.environ["WANDB_MODE"] = "disabled"


teacher_model_path = "swj0419/7b_iclm"
student_model_path = "muse-bench/MUSE-books_target"
tokenizer_name = "meta-llama/Llama-2-7b-hf"
train_file_path = "/n/home04/huandongchang/muse_bench/data/books/raw/forget.txt"
eval_file_path = "/n/home04/huandongchang/muse_bench/data/books/raw/forget.txt"
output_dir = "./distillation_3epoch_soft_label_only"

# class TextDataset(Dataset):
#     def __init__(self, file_path, tokenizer, max_length=2048):
#         self.tokenizer = tokenizer
#         self.max_length = max_length
#         with open(file_path, 'r', encoding='utf-8') as f:
#             self.lines = f.readlines()

#     def __len__(self):
#         return len(self.lines)

#     def __getitem__(self, idx):
#         line = self.lines[idx].strip()
#         encoding = self.tokenizer(
#             line,
#             return_tensors="pt",
#             padding="max_length",
#             truncation=True,
#             max_length=self.max_length
#         )
#         return {
#             "input_ids": encoding["input_ids"].squeeze(),
#             "attention_mask": encoding["attention_mask"].squeeze()
#         }

class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Read all lines from the file
        with open(file_path, 'r', encoding='utf-8') as f:
            self.lines = [line.strip() for line in f if line.strip()]  # Ignore empty lines

        # Preprocess: Combine lines to maximize usage of max_length
        self.examples = self._chunk_lines()

    def _chunk_lines(self):
        examples = []
        current_chunk = ""
        for line in self.lines:
            # Add the current line to the chunk
            if current_chunk:
                current_chunk += " " + line
            else:
                current_chunk = line

            # Tokenize and check if the chunk exceeds max_length
            tokenized_chunk = self.tokenizer(
                current_chunk,
                truncation=False,  # Do not truncate here
                return_tensors="pt"
            )
            if tokenized_chunk["input_ids"].shape[1] > self.max_length:
                # Save the chunk before it exceeds max_length
                examples.append(current_chunk)
                # Start a new chunk with the current line
                current_chunk = line
        
        # Add the last chunk if it's not empty
        if current_chunk:
            examples.append(current_chunk)
        
        return examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        chunk = self.examples[idx]
        encoding = self.tokenizer(
            chunk,
            return_tensors="pt",
            padding="max_length",
            truncation=True,  # Ensure the final chunk is exactly max_length
            max_length=self.max_length
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze()
        }


def main():


    # Load the student model and tokenizer
    student_model = AutoModelForCausalLM.from_pretrained(student_model_path,  torch_dtype=torch.float16)
                                                         #load_in_8bit=True)  # Optional: automatically places model on available GPUs
    lora_config = LoraConfig(
    r=8,               # Rank of the low-rank matrices
    lora_alpha=32,     # Scaling parameter
    lora_dropout=0.1
)
    
    student_model = get_peft_model(student_model, lora_config)
                                                        
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Load datasets
    train_dataset = TextDataset(train_file_path, tokenizer)
    eval_dataset = TextDataset(eval_file_path, tokenizer)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="steps",
        eval_steps=1000,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        logging_dir="./logs",
        logging_steps=100,
        save_steps=100,
        save_total_limit=20,
        load_best_model_at_end=False,
        max_grad_norm=1.0,
        num_train_epochs=3
    )
    

    # Initialize WandB for logging if required
    # wandb.init(project="distillation_project")

    # Initialize the DistillTrainer
    trainer = DistillTrainer(
        teacher_model_path=teacher_model_path,
        model=student_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )
    print(trainer.optimizer)

    # Start training
    trainer.train()

if __name__ == "__main__":
    main()
