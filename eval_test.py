from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Paths to your base model, tokenizer, and LoRA checkpoint
base_model_path = "muse-bench/MUSE-books_target"
tokenizer_path = "meta-llama/Llama-2-7b-hf"
lora_checkpoint_path = "/n/home04/huandongchang/distillation_results_3epoch_job_llm/checkpoint-800"


# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
tokenizer.pad_token = tokenizer.eos_token 

# Load the base model
base_model = AutoModelForCausalLM.from_pretrained(base_model_path)

# Load the LoRA fine-tuned model
model = PeftModel.from_pretrained(base_model, lora_checkpoint_path)
# model.config.pad_token_id = tokenizer.eos_token_id



# Move the model to the GPU (if available)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)


for name, param in model.named_parameters():
    if torch.isnan(param).any().item():
        print(f"NaN found in {name}")
    if torch.isinf(param).any().item():
        print(f"Inf found in {name}")







# Set the model to evaluation mode
model.eval()

# Example input text
input_text = "What is the capital of France?"

# Tokenize the input text
inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to(device)

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    print("Logits:", logits)
    print("Logits contain NaN:", torch.isnan(logits).any().item())
    print("Logits contain Inf:", torch.isinf(logits).any().item())
    

assert isinstance(model, PeftModel)

# Generate output using the fine-tuned model
with torch.no_grad():
    outputs = model.generate(
        inputs.input_ids,
        max_length=50,  # Adjust as needed
        attention_mask=inputs.attention_mask, 
        do_sample=True,
        temperature=0.7,  # Adjust as needed
        top_k=50,        # Adjust as needed
        top_p=0.95       # Adjust as needed
    )

# Decode the generated output
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Generated Output:", generated_text)


