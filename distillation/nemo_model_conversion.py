# convert_to_nemo.py
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from nemo.collections.nlp.models.language_modeling import MegatronGPTModel

# Parameters
teacher_model_path = "swj0419/7b_iclm"
student_model_path = "muse-bench/MUSE-books_target"
tokenizer_name = "meta-llama/Llama-2-7b-hf"
teacher_nemo_path = "teacher_model.nemo"
student_nemo_path = "student_model.nemo"

# Load Hugging Face models and tokenizer
teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model_path)
student_model = AutoModelForCausalLM.from_pretrained(student_model_path)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

# Convert models to NeMo-compatible format
# Create NeMo model instances and assign weights from Hugging Face models
teacher_nemo_model = MegatronGPTModel.restore_from_huggingface(teacher_model)
student_nemo_model = MegatronGPTModel.restore_from_huggingface(student_model)

# Save models in .nemo format
teacher_nemo_model.save_to(teacher_nemo_path)
student_nemo_model.save_to(student_nemo_path)

print("Conversion to .nemo format completed.")
