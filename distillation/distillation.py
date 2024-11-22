import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from torch.utils.data import Dataset
from torch.nn.functional import kl_div, softmax, log_softmax
import os

# Check if GPU is available and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model names and file path
teacher_model_name = "swj0419/7b_iclm"
student_model_name = "muse-bench/MUSE-books_target"
tokenizer_name = "meta-llama/Llama-2-7b-hf"
file_path = "/n/home04/huandongchang/muse_bench/data/books/raw/forget_short.txt"
output_dir = "./distilled_student_old"

# Load teacher and student models in full precision (FP32) and move to GPU
teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model_name, torch_dtype=torch.float16).to(device)
student_model = AutoModelForCausalLM.from_pretrained(student_model_name, torch_dtype=torch.float16).to(device)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
tokenizer.pad_token = tokenizer.eos_token

# Prepare dataset
class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        with open(file_path, 'r', encoding='utf-8') as f:
            self.lines = f.readlines()

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx].strip()
        encoding = self.tokenizer(
            line,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze()
        }

# Load data
train_dataset = TextDataset(file_path, tokenizer)

# Define distillation loss function
def distillation_loss(student_logits, teacher_logits, temperature=2.0):
    teacher_probs = softmax(teacher_logits / temperature, dim=-1)
    student_log_probs = log_softmax(student_logits / temperature, dim=-1)
    return kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature ** 2)

# Custom Trainer for distillation
class DistillationTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        
        
    
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        
        # Forward pass for teacher in eval mode to save memory
        with torch.no_grad():
            teacher_outputs = teacher_model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Forward pass for student model
        student_outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Compute distillation loss
        loss = distillation_loss(student_outputs.logits, teacher_outputs.logits)
        return (loss, student_outputs) if return_outputs else loss

# Define training arguments without mixed precision
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    learning_rate=5e-5,
    num_train_epochs=1,
    logging_dir='./logs',
    save_total_limit=1,
    ddp_find_unused_parameters=False, # Set to False to avoid DDP errors for unused parameters
    fp16=True   
)

# Initialize and run the trainer
trainer = DistillationTrainer(
    model=student_model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer
)

# Launch training
trainer.train()
