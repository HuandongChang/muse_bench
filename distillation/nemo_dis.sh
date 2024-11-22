#!/bin/bash

# Paths
STUDENT_CKPT="student_model.nemo"  # Path to converted student model
TEACHER_CKPT="teacher_model.nemo"  # Path to converted teacher model
TOKENIZER_PATH="path/to/tokenizer.model"  # Path to tokenizer file
DATA_PATHS="[1.0,/n/home04/huandongchang/muse_bench/data/books/tokenized_for_nemo.txt]"
FINAL_SAVE_FILE="./distilled_student/final_checkpoint.nemo"  # Output file for distilled model

# Tensor Parallelism (TP) degree
TP=4  # Number of GPUs or parallelism level

# Set number of processes
NPROC=$TP
launch_config="torchrun --nproc_per_node=$NPROC"

# Run distillation script
${launch_config} examples/nlp/language_modeling/megatron_gpt_distillation.py \
    model.restore_from_path=$STUDENT_CKPT \
    model.kd_teacher_restore_from_path=$TEACHER_CKPT \
    model.tensor_model_parallel_size=$TP \
    model.tokenizer.model=$TOKENIZER_PATH \
    model.data.data_prefix=$DATA_PATHS \
    model.nemo_path=$FINAL_SAVE_FILE \
    trainer.precision=bf16 \
    trainer.devices=$NPROC

echo "Distillation process completed."
