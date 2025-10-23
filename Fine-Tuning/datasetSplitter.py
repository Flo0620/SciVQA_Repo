import json
import random
import argparse

parser = argparse.ArgumentParser(description="Split a JSON dataset into train and test sets.")
parser.add_argument("--input_file", type=str, default="/ltstorage/home/9schleid/SciVQA/unsloth/input.json", help="Path to the input JSON file.")
parser.add_argument("--test_split_file", type=str, default="/ltstorage/home/9schleid/SciVQA/unsloth/test_split.json", help="Path to save the test split JSON file.")
parser.add_argument("--train_split_file", type=str, default="/ltstorage/home/9schleid/SciVQA/unsloth/train_split.json", help="Path to save the train split JSON file.")
parser.add_argument("--split_pos", type=int, default=4200, help="Index at which to split the dataset.")

args = parser.parse_args()

# Read the JSON data
with open(args.input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# Shuffle the data
random.shuffle(data)

# Split the data
test_split = data[:args.split_pos]
train_after_split = data[args.split_pos:]

# Save the test split
with open(args.test_split_file, "w", encoding="utf-8") as f:
    json.dump(test_split, f, ensure_ascii=False, indent=2)

# Save the remaining training data
with open(args.train_split_file, "w", encoding="utf-8") as f:
    json.dump(train_after_split, f, ensure_ascii=False, indent=2)