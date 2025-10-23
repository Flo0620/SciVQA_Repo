import json

json_path = "/ltstorage/home/9schleid/SciVQA/unsloth/SpiQa/SPIQA_train_after_split.json"

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Number of entries: {len(data)}")