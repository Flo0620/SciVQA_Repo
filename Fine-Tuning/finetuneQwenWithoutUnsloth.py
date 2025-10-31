#Code kommt hauptsächlich von https://huggingface.co/learn/cookbook/en/fine_tuning_vlm_trl

import torch
import mlflow
from DatasetLoader import DatasetLoader
from datetime import datetime
#import mlflow
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from transformers import AutoTokenizer, DataCollatorWithPadding
from peft import LoraConfig, get_peft_model
import yaml

from dotenv import load_dotenv
import os

load_dotenv()

hf_token = os.getenv("HF_Token")

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--lora_r", type=int, default=64)
parser.add_argument("--lora_alpha", type=int, default=128)
parser.add_argument("--lora_dropout", type=float, default=0.2)
parser.add_argument("--learning_rate", type=float, default=2e-4)
parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-VL-32B-Instruct")
parser.add_argument("--lr_scheduler_type", type=str, default="linear")
parser.add_argument("--output_dir", type=str, default="Qwen2_5_32B-8bit_2Epochs")
parser.add_argument("--resume_training", action='store_true', help="Resume training from checkpoint")
parser.add_argument("--num_epochs", type=int, default=1)
parser.add_argument("--dataset_path", type=str, default="/ltstorage/home/9schleid/SciVQA/unsloth/shared_task/train_2025-03-27_18-34-44.json")

args = parser.parse_args()

print(args.lora_r)
print(args.lora_alpha)
print(args.lora_dropout)
print(args.learning_rate)
print(args.model_id)
print(args.lr_scheduler_type)
print(args.output_dir)
print(args.resume_training)
print(args.num_epochs)
print(args.dataset_path)

dataset = DatasetLoader(
    dataset_json_file_path = args.dataset_path,
    image_dir = "/ltstorage/home/9schleid/SciVQA/unsloth/SciVQAAndSpiQAAndArXivQA/SPIQA_And_SciVQA_And_ArXivQA_train_images"
)

validation_dataset = DatasetLoader(
    dataset_json_file_path = "/ltstorage/home/9schleid/SciVQA/unsloth/shared_task/validation_2025-03-27_18-34-44.json",
    image_dir = "/ltstorage/home/9schleid/SciVQA/unsloth/shared_task/images_validation/images_validation"
)

from transformers import BitsAndBytesConfig

# BitsAndBytesConfig int-8 config
bnb_config_args = {
    "load_in_8bit": True,
}

bnb_config = BitsAndBytesConfig(**bnb_config_args)

#model_id = "Qwen/Qwen2.5-VL-32B-Instruct"
#model_id = "Qwen/Qwen2.5-VL-7B-Instruct"

# Load model and tokenizer
model_config_args = {
    "pretrained_model_name_or_path": args.model_id,
    "device_map": "auto",
    "torch_dtype": torch.bfloat16,
    "quantization_config": bnb_config
}

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    **{key: value for key, value in model_config_args.items()}
)
processor = AutoProcessor.from_pretrained(args.model_id)


peft_config_args = {
    "lora_alpha": args.lora_alpha,
    "lora_dropout": args.lora_dropout,
    "r": args.lora_r,
    "bias": "none",
    "target_modules": ["q_proj", "v_proj"],
    #"target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
    "task_type": "CAUSAL_LM",
}

peft_config = LoraConfig(
    **{key: value for key, value in peft_config_args.items()}
)

# Apply PEFT model adaptation
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()


from trl import SFTConfig

# Configure training arguments
training_config = {
    "output_dir": args.output_dir,
    "num_train_epochs": args.num_epochs,
    "max_steps": 2,
    "per_device_train_batch_size": 1,
    "per_device_eval_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "max_seq_length": 4096,
    "gradient_checkpointing": True,
    "optim": "adamw_torch_fused",
    "learning_rate": args.learning_rate,
    "lr_scheduler_type": args.lr_scheduler_type,
    "logging_steps": 10,
    #"eval_steps": 1,  # Steps interval for evaluation
    #"eval_strategy": "steps",  # Strategy for evaluation
    "save_strategy": "epoch",
    #"save_strategy": "steps",
    "save_steps": 1,
    "metric_for_best_model": "eval_loss",
    "greater_is_better": False,
    "bf16": True,
    "tf32": True,
    "max_grad_norm": 0.3,
    "warmup_ratio": 0.03,
    "push_to_hub": True,
    "report_to": "mlflow",
    "gradient_checkpointing_kwargs": {"use_reentrant": False},
    "dataset_text_field": "",
    "dataset_kwargs": {"skip_prepare_dataset": True},
    "label_names": ["labels"],
}


training_args = SFTConfig(**training_config)
training_args.remove_unused_columns = False  # Keep unused columns in dataset

# Create a data collator to encode text and image pairs
def collate_fn(examples):
    # Get the texts and images, and apply the chat template
    texts = [
        processor.apply_chat_template(example, tokenize=False) for example in examples
    ]  # Prepare texts for processing
    image_inputs = [process_vision_info(example)[0] for example in examples]  # Process the images to extract inputs

    # Tokenize the texts and process the images
    batch = processor(
        text=texts, images=image_inputs, return_tensors="pt", padding=True
    )  # Encode texts and images into tensors

    # The labels are the input_ids, and we mask the padding tokens in the loss computation
    labels = batch["input_ids"].clone()  # Clone input IDs for labels
    labels[labels == processor.tokenizer.pad_token_id] = -100  # Mask padding tokens in labels

    image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]  # Convert image token to ID

    # Mask image token IDs in the labels
    for image_token_id in image_tokens:
        labels[labels == image_token_id] = -100  # Mask image token IDs in labels

    batch["labels"] = labels  # Add labels to the batch

    return batch  # Return the prepared batch


from trl import SFTTrainer
import numpy as np

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset.getConvertedDataset(),
    eval_dataset=validation_dataset.getConvertedDataset()[:10],
    data_collator=collate_fn,
    peft_config=peft_config,
    compute_metrics=compute_metrics,
    tokenizer=processor.tokenizer,
)

mlflow.set_tracking_uri("https://mlflow-g4k-serving-474827717259.europe-west3.run.app/")
mlflow.set_experiment("Sci-VQA")

cfg = {}
cfg['system_prompt'] = dataset.system_prompt
cfg['user_prompt'] = dataset.user_prompt
cfg.update(bnb_config_args)
cfg.update(model_config_args)
cfg.update(peft_config_args)
cfg.update(training_config)

with open('/ltstorage/home/9schleid/scivqa/conf/defaults.yaml', 'r') as cfg_file:
    defaults_config = yaml.safe_load(cfg_file)
    defaults_config = {
        "apply_few_shot": defaults_config.get("apply_few_shot", False),
        "few_shot_template": defaults_config.get("few_shot_template", ""),
        "few_shot_dataset_path": defaults_config.get("few_shot_dataset_path", ""),
        "few_shot_images_path": defaults_config.get("few_shot_images_path", ""),
        "adapter_path": defaults_config.get("adapter_path", ""),
        "model_id": args.model_id,
    }

    cfg.update(defaults_config)
mlflow.log_params(cfg)
mlflow.set_tag("mlflow.note.content", f"Fine-tuning {args.model_id} r={args.lora_r} a={args.lora_alpha} d={args.lora_dropout} lr={args.learning_rate} scheduler={args.lr_scheduler_type} output_dir={args.output_dir}")
if args.resume_training:
    trainer.train(resume_from_checkpoint=True)
else:
    trainer.train()
trainer.save_model(training_args.output_dir)

