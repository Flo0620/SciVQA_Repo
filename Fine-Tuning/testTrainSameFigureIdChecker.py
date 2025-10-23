import json

test_json_path = "/ltstorage/home/9schleid/SciVQA/unsloth/arxivqa/filteredDatasetLCSS7_minus_6Processed_Test_set_ohne_figures_aus_train_set.json"
train_json_path = "/ltstorage/home/9schleid/SciVQA/unsloth/arxivqa/filteredDatasetLCSS6ProcessedForHyperparametertuning_train_split.json"

# Load JSON files
with open(test_json_path, "r") as f:
    test_entries = json.load(f)

with open(train_json_path, "r") as f:
    train_entries = json.load(f)

# Build a mapping from figure_id to instance_id for train entries
train_paper_map = {}
for entry in train_entries:
    pid = entry.get("figure_id")
    iid = entry.get("instance_id")
    if pid is not None and iid is not None:
        train_paper_map.setdefault(pid, []).append(iid)

# Check for matching figure_ids and print instance_id pairs
for test_entry in test_entries:
    test_pid = test_entry.get("figure_id")
    test_iid = test_entry.get("instance_id")
    if test_pid is not None and test_iid is not None:
        if test_pid in train_paper_map:
            for train_iid in train_paper_map[test_pid]:
                print(f"Test instance_id: {test_iid}, Train instance_id: {train_iid}, figure_id: {test_pid}")