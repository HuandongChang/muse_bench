import json
import pandas as pd
import os
from typing import List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer


def read_json(fpath: str) -> Dict | List:
    with open(fpath, 'r') as f:
        return json.load(f)


def read_text(fpath: str) -> str:
    with open(fpath, 'r') as f:
        return f.read()


def write_json(obj: Dict | List, fpath: str):
    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    with open(fpath, 'w') as f:
        return json.dump(obj, f)


def write_text(obj: str, fpath: str):
    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    with open(fpath, 'w') as f:
        return f.write(obj)


# def write_csv(obj, fpath: str):
#     os.makedirs(os.path.dirname(fpath), exist_ok=True)
#     pd.DataFrame(obj).to_csv(fpath, index=False)

def write_csv(obj, fpath: str):
    # Ensure the directory exists only if the directory part is not empty
    dir_name = os.path.dirname(fpath)
    if dir_name:  # Avoid calling os.makedirs with an empty string
        os.makedirs(dir_name, exist_ok=True)
    pd.DataFrame(obj).to_csv(fpath, index=False)


def load_model(model_dir: str, **kwargs):
    return AutoModelForCausalLM.from_pretrained(model_dir, **kwargs).to('cuda')


def load_tokenizer(tokenizer_dir: str, **kwargs):
    return AutoTokenizer.from_pretrained(tokenizer_dir, **kwargs)
    