import json

# File paths
combined_file = "/ltstorage/home/9schleid/SciVQA/unsloth/SciVQAAndSpiQAAndArXivQA/combinedOhneTestSplits.json"
test_files = [
    "/ltstorage/home/9schleid/SciVQA/unsloth/SpiQa/SPIQA_test_split.json",
    "/ltstorage/home/9schleid/SciVQA/unsloth/shared_task/test_without_answers_2025-04-14_15-30.json",
    "/ltstorage/home/9schleid/SciVQA/unsloth/arxivqa/filteredDatasetLCSS6ProcessedForHyperparametertuning_test_split.json"
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
output_file = "/ltstorage/home/9schleid/SciVQA/unsloth/SciVQAAndSpiQAAndArXivQA/combinedOhneTestSplits_filtered.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(filtered_data, f, ensure_ascii=False, indent=2)

print(f"Filtered data saved to {output_file}. Removed {len(combined_data) - len(filtered_data)} entries.")
print(f"Combined data contained {len(combined_data)} entries.")
print(f"Test sets contained {len(test_instance_ids)} entries.")