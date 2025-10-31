import json

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--test_json_path", type=str, default=None)
parser.add_argument("--train_json_path", type=str, default=None)
args = parser.parse_args()
test_json_path = args.test_json_path
train_json_path = args.train_json_path

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