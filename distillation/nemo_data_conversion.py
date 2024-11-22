# preprocess_data.py
from transformers import AutoTokenizer
import os

# Paths and Parameters
tokenizer_name = "meta-llama/Llama-2-7b-hf"  # Hugging Face tokenizer name
input_file = "/n/home04/huandongchang/muse_bench/data/books/raw/forget_short.txt"
output_file = "/n/home04/huandongchang/muse_bench/data/books/tokenized_for_nemo.txt"
max_length = 512  # Adjust as needed

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

# Process and save the tokenized data
with open(input_file, "r", encoding="utf-8") as f_in, open(output_file, "w", encoding="utf-8") as f_out:
    for line in f_in:
        line = line.strip()
        if not line:
            continue  # Skip empty lines
        tokens = tokenizer.encode(line, truncation=True, max_length=max_length)
        tokenized_line = " ".join(map(str, tokens))  # Convert tokens to space-separated string
        f_out.write(tokenized_line + "\n")

print(f"Tokenized data saved to {output_file}")
