import json
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_path", type=str, default=None)
parser.add_argument("--output_path", type=str, default=None)
args = parser.parse_args()
input_path = args.input_path
output_path = args.output_path

def reformat_entry(columns, values):
    entry = dict(zip(columns, values))

    # Build user_prompt
    user_prompt = []
    if entry.get("user_prompt0_type") == "image_url":
        user_prompt.append({
            "type": "image_url",
            "image_url": {"url": entry.get("user_prompt0_image_url_url")}
        })
    if entry.get("user_prompt1_type") == "text":
        user_prompt.append({
            "type": "text",
            "text": entry.get("user_prompt1_text")
        })

    # Build meta_data
    meta_data = {
        "figure_id": entry.get("meta_data_figure_id"),
        "image_path": entry.get("meta_data_image_path"),
        "question": entry.get("meta_data_question"),
        "reference_answer": entry.get("meta_data_reference_answer"),
        "caption": entry.get("meta_data_caption"),
        "qa_pair_type": entry.get("meta_data_qa_pair_type"),
        "categories": entry.get("meta_data_categories"),
        "instance_id": entry.get("meta_data_instance_id"),
    }

    # Build context
    context = {
        "documents": [],
        "prompt_vars": {
            "qa_pair_type": entry.get("context_prompt_vars_qa_pair_type"),
            "answer_options": entry.get("context_prompt_vars_answer_options") if entry.get("context_prompt_vars_answer_options") != "None" else None,
            "caption": entry.get("context_prompt_vars_caption"),
            "compound": True if entry.get("context_prompt_vars_compound")=="True" else False,
            "figure_type": entry.get("context_prompt_vars_figure_type"),
        }
    }

    # Compose final entry
    result = {
        "request_id": entry.get("request_id"),
        "prompt_id": entry.get("prompt_id"),
        "sys_prompt": entry.get("sys_prompt"),
        "user_prompt": user_prompt,
        "response": entry.get("response"),
        "conversation_id": entry.get("conversation_id"),
        "meta_data": meta_data,
        "context": context,
        "arrival_time": entry.get("arrival_time"),
        "finished_time": entry.get("finished_time"),
        "processing_time": entry.get("processing_time"),
    }
    return result

def main():
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    columns = data["columns"]
    rows = data["data"]

    reformatted = [reformat_entry(columns, row) for row in rows]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(reformatted, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()