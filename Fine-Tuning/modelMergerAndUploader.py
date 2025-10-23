from unsloth import FastVisionModel

from dotenv import load_dotenv
import os

# Load the .env file
load_dotenv()

# Access the variable
hf_token = os.getenv("HF_Token")

model, tokenizer = FastVisionModel.from_pretrained(
    model_name = "lora_model", # YOUR MODEL YOU USED FOR TRAINING
    load_in_4bit = True, # Set to False for 16bit LoRA
)

model.push_to_hub_merged("Flo0620/Qwen2.5-VL-7B-instruct-bnb-4bit-finetuned2", tokenizer, token = hf_token)