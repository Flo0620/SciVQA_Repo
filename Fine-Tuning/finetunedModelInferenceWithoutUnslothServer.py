import torch
from flask import Flask, request, jsonify
import yaml
from DatasetLoader import DatasetLoader
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import json
import hashlib
import time
import random 
import argparse
import traceback

# Load port from configuration file
config_path = "/ltstorage/home/9schleid/scivqa/conf/defaults.yaml"
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

parser = argparse.ArgumentParser()
parser.add_argument("--adapter_path", type=str, default=None)
parser.add_argument("--model_id", type=str, default=None)
parser.add_argument("--parse_json_answer", type=bool, default=False)
parser.add_argument("--port", type=str, default=config.get("vllm_port", 5000))


args = parser.parse_args()

vllm_port = args.port

# Initialize Flask app
app = Flask(__name__)

# Load model and processor
if args.adapter_path is None:
    adapter_path = config.get("adapter_path", "")
else:
    adapter_path = args.adapter_path
if args.model_id is None:
    model_id = config.get("model_id", "")
else:
    model_id = args.model_id

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
processor = AutoProcessor.from_pretrained(model_id)
model.load_adapter(adapter_path)

# Function to generate text
def generate_text_from_sample(model, processor, conversation, max_new_tokens=1024, device="cuda"):
    # Prepare the text input by applying the chat template
    text_input = processor.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )

    # Process the visual input (dummy placeholder for now)
    image_inputs, _ = process_vision_info(conversation)

    # Prepare the inputs for the model
    model_inputs = processor(
        text=[text_input],
        images=image_inputs,
        return_tensors="pt",
    ).to(device)  # Move inputs to the specified device

    # Generate text with the model
    generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens)

    # Trim the generated ids to remove the input ids
    trimmed_generated_ids = [out_ids[len(in_ids):] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)]

    # Decode the output text
    output_text = processor.batch_decode(
        trimmed_generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output_text[0]  # Return the first decoded output text

# Flask endpoint
@app.route('/v1/chat/completions', methods=['POST'])
def generate():
    try:
        data = json.loads(request.get_json())
        # Record the request arrival time
        processing_start_time = time.time()
        data["arrival_time"] = processing_start_time

        messages = data.get("conversation", [])
        print("Received messages:", messages)

        # Generate text
        output = generate_text_from_sample(model, processor, messages)

        # Add the generated response to the incoming JSON
        if args.parse_json_answer:
            output_json = json.loads(output)
            data["response"] = output_json.get("answer", "")
        else:
            data["response"] = output
        # Find the message in the messages list that has the role "system"
        system_message = next((msg for msg in messages if msg.get("role") == "system"), None)
        user_message = next((msg for msg in messages if msg.get("role") == "user"), None)

        data["sys_prompt"] = system_message.get("content") if system_message else ""
        data["user_prompt"] = user_message.get("content") if user_message else ""
        # Create a hash of the messages list
        messages_str = json.dumps(messages, sort_keys=True)
        data["prompt_id"] = hashlib.sha256(messages_str.encode()).hexdigest()
        data["request_id"] = "justRandomValue_" + str(hashlib.sha256(str(random.randint(0, int(1e6))).encode()).hexdigest())
        data["conversation_id"] = "justRandomValue_"+str(hashlib.sha256(str(random.randint(0, int(1e6))).encode()).hexdigest())

        # Record the processing end time
        data["processing_time"] = time.time() - processing_start_time

        # Record the request finish time
        data["finished_time"] = time.time()

        # Return the modified JSON as the response
        return jsonify(data)
    except Exception as e:
        print("Error:", str(e))
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/v1/health")
def health():
    return "OK", 200


# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=vllm_port)
    