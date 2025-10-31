import json

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--file1_path", type=str, default=None)
parser.add_argument("--file2_path", type=str, default=None)
parser.add_argument("--output_path", type=str, default=None)
args = parser.parse_args()
file1_path = args.file1_path
file2_path = args.file2_path
output_path = args.output_path

# Load first file and filter out entries with 'exclude_from_training'
with open(file1_path, "r", encoding="utf-8") as f1:
    data1 = json.load(f1)
filtered_data1 = [entry for entry in data1 if "exclude_from_training" not in entry]
instance_ids_1 = set(entry["instance_id"] for entry in data1)



# Load second file and filter out entries with duplicate instance_id
with open(file2_path, "r", encoding="utf-8") as f2:
    data2 = json.load(f2)
filtered_data2 = [entry for entry in data2 if entry["instance_id"] not in instance_ids_1]

# Combine and save
combined_data = filtered_data1 + filtered_data2
with open(output_path, "w", encoding="utf-8") as fout:
    json.dump(combined_data, fout, ensure_ascii=False, indent=2)