import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--json_path", type=str, default=None)
args = parser.parse_args()
json_path = args.json_path

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

original_SpiQA_data = False
if original_SpiQA_data:
    responses = [qa.get("response",None) for entry in data.values() for qa in entry.get("qa", "") if qa.get("response",None) is not None]
else:
    responses = [entry.get("response", "").removeprefix("Answer: ") for entry in data]

lengths = [len(response) for response in responses]

num_entries = len(responses)
average_length = sum(lengths) / num_entries if num_entries > 0 else 0

print(f"Number of entries: {num_entries}")
print(f"Average response length: {average_length:.2f}")