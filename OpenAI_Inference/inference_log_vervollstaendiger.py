import json
import time
import uuid

# File paths
inference_log_path = "/ltstorage/home/9schleid/OpenAIServer/temp/inference_log.json"
validation_path = "/ltstorage/home/9schleid/OpenAIServer/shared_task/test_2025-07-03_09-01.json"
output_path = "/ltstorage/home/9schleid/OpenAIServer/temp/inference_log_32B_one_shot_auf_test.json"

# Load inference log (instance_id -> response)
with open(inference_log_path, "r") as f:
    inference_entries = json.load(f)
instance_to_response = {entry["meta_data"]["instance_id"]: entry["response"] for entry in inference_entries}

# Load validation entries
with open(validation_path, "r") as f:
    validation_entries = json.load(f)

output_entries = []
for entry in validation_entries:
    instance_id = entry.get("instance_id")
    response = instance_to_response.get(instance_id, None)
    response = response.removeprefix("Answer: ")
    # Compose new structure
    new_entry = {
        "arrival_time": 0,
        "context": {
            "documents": [],
            "prompt_vars": {
                "answer_options": entry.get("answer_options", None) if entry.get("answer_options") else None,
                "caption": entry.get("caption", None),
                "compound": entry.get("compound", None),
                "figure_type": entry.get("figure_type", None),
                "qa_pair_type": entry.get("qa_pair_type", None)
            }
        },
        "conversation": None,
        "conversation_id": None,
        "finished_time": 0,
        "id": None,
        "meta_data": {
            "caption": entry.get("caption", None),
            "categories": entry.get("categories", None),
            "figure_id": entry.get("figure_id", None),
            "image_path": entry.get("image_file", None),
            "instance_id": entry.get("instance_id", None),
            "qa_pair_type": entry.get("qa_pair_type", None),
            "question": entry.get("question", None),
            "reference_answer": entry.get("answer", None)
        },
        "model": "gpt-4o-mini",
        "processing_time": 0,
        "prompt_id": None,
        "request_id": None,
        "response": response,
        "sys_prompt": None,
        "user_prompt": None
    }
    output_entries.append(new_entry)

with open(output_path, "w") as f:
    json.dump(output_entries, f, indent=4)