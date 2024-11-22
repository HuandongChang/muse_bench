from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from torch.nn import functional as F
import torch
from torch.utils.data import random_split
from torch.utils.data import Dataset
import os


# Model and tokenizer
teacher_model_name = "swj0419/7b_iclm"
student_model_name = "muse-bench/MUSE-books_target"
tokenizer_name = "meta-llama/Llama-2-7b-hf"
file_path = "/n/home04/huandongchang/muse_bench/data/books/raw/forget_short.txt"
output_dir = "./distilled_student"

class DistillTrainer(Trainer):
    def __init__(self, teacher_model_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert teacher_model_path is not None, "[DistillTrainer] teacher model is not initialized. Ensure that the teacher model is passed to the DistillTrainer."

        # Prepare teacher model
        self._prepare_teacher_model(teacher_model_path)

    def _prepare_teacher_model(self, teacher_model_path):

        self.teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model_path)
        self.teacher_model.eval().half()  # Use half-precision for memory efficiency
        self.teacher_model = self._wrap_model(self.teacher_model)  # Wrapping for DDP compatibility
        self.teacher_model = self.accelerator.prepare_model(self.teacher_model, evaluation_mode=True)

    def compute_loss(self, model, inputs, return_outputs=False):

        # Compute student model outputs
        
        labels = inputs.get("input_ids").clone() if "labels" not in inputs else inputs.pop("labels")
        inputs["labels"] = labels
        
       
        
        student_outputs = model(**inputs)
        
        # if labels is not None:
        #     unwrapped_model = unwrap_model(model)
        
        if "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        
       
        # Calculate the student's primary loss
        # primary_loss = student_outputs["loss"] if isinstance(student_outputs, dict) else student_outputs[0]
        primary_loss = student_outputs.loss.mean() if hasattr(student_outputs, "loss") else None

        # Calculate distillation loss
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)
        
        _, _, logits_loss, distillation_loss = self._compute_distillation_loss(teacher_outputs, student_outputs)
        
        # breakpoint()
        # Log distillation losses
        if self.state.is_world_process_zero:
            self.log({
                "distillation/logits_loss": logits_loss.item(),
                "distillation/primary_loss": primary_loss.item(),
            })

        # Final loss combines primary loss and distillation loss
        total_loss = primary_loss + 0.5 * distillation_loss
        return (total_loss, student_outputs) if return_outputs else total_loss

    def _compute_distillation_loss(self, teacher_outputs, student_outputs):
        student_logits = student_outputs.logits
        teacher_logits = teacher_outputs.logits

        # Reverse KL divergence (student learns from teacher)
        teacher_log_prob = F.log_softmax(teacher_logits, dim=-1)
        student_prob = F.softmax(student_logits, dim=-1)
        reverse_kl = F.kl_div(teacher_log_prob, student_prob, reduction="batchmean")

        # Forward KL divergence (teacher signals to student)
        student_log_prob = F.log_softmax(student_logits, dim=-1)
        teacher_prob = F.softmax(teacher_logits, dim=-1)
        forward_kl = F.kl_div(student_log_prob, teacher_prob, reduction="batchmean")

        kl_loss = 0.5 * reverse_kl + 0.5 * forward_kl  # Average of forward and reverse KL

        return 0, 0, kl_loss, kl_loss



tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
tokenizer.pad_token = tokenizer.eos_token
student_model = AutoModelForCausalLM.from_pretrained(student_model_name).half()


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
full_dataset = TextDataset(file_path, tokenizer)
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, eval_dataset = random_split(full_dataset, [train_size, test_size])


# Data preparation (replace with an actual data loader for your dataset)
# train_dataset = load_dataset(file_path) # Ensure it's compatible

# Training arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=1,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=50,
    ddp_find_unused_parameters=False
    # fp16=True
)


# Initialize DistillTrainer with teacher model path
distill_trainer = DistillTrainer(
    teacher_model_path=teacher_model_name,
    model=student_model,
    args=training_args,
    train_dataset=train_dataset,  # Replace with actual dataset
    eval_dataset=eval_dataset,    # Replace with actual dataset
)


# Start distillation training
distill_trainer.train()
