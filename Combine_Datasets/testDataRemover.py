import json

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--combined_file", type=str, default=None)
parser.add_argument("--spiqa_file_path", type=str, default=None)
parser.add_argument("--scivqa_file_path", type=str, default=None)
parser.add_argument("--arxivqa_file_path", type=str, default=None)
parser.add_argument("--output_file", type=str, default=None)
args = parser.parse_args()
combined_file = args.combined_file
spiqa_file_path = args.spiqa_file_path
scivqa_file_path = args.scivqa_file_path
arxivqa_file_path = args.arxivqa_file_path
output_file = args.output_file

# File paths
test_files = [
    spiqa_file_path,
    scivqa_file_path,
    arxivqa_file_path
]

# Load instance_ids from test files
test_instance_ids = set()
for test_file in test_files:
    with open(test_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        test_instance_ids.update(entry["instance_id"] for entry in data)

# Load combined file
with open(combined_file, "r", encoding="utf-8") as f:
    combined_data = json.load(f)

# Filter out entries present in test_instance_ids
filtered_data = [entry for entry in combined_data if entry["instance_id"] not in test_instance_ids]

# Save filtered data
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(filtered_data, f, ensure_ascii=False, indent=2)

print(f"Filtered data saved to {output_file}. Removed {len(combined_data) - len(filtered_data)} entries.")
print(f"Combined data contained {len(combined_data)} entries.")
print(f"Test sets contained {len(test_instance_ids)} entries.")