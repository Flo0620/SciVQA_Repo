import json
import re

# Load the first dataset (JSONL format)
def load_jsonl(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return [json.loads(line.strip()) for line in f]

# Load the second dataset (JSON format)
def load_json(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_id_from_image(image_path):
    # Assumes image path like "images/0904.0709_0.jpg"
    filename = image_path.split('/')[-1].split('_')[0]
    return re.sub(r'\D', '', filename)  # Remove all non-numeric characters

def preprocess_paper_id(paper_id):
    return re.sub(r'\D', '', paper_id)  # Remove all non-numeric characters

def longest_common_substring(s1, s2):
    m = len(s1)
    n = len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    lcs_length = 0

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                lcs_length = max(lcs_length, dp[i][j])

    return lcs_length

def find_matching_instances(first_dataset, second_dataset):
    paper_ids = {preprocess_paper_id(entry['paper_id']) for entry in second_dataset}

    matching_instances = []
    for i, entry in enumerate(first_dataset):
        if i % 1000 == 0:
            print(f"Processing entry {i} of {len(first_dataset)}")
        id_from_image = extract_id_from_image(entry['image'])
        for paper_id in paper_ids:
            if longest_common_substring(id_from_image, paper_id) > 7:
                matching_instances.append(id_from_image)
                break

    return matching_instances

def remove_and_save_duplicates(first_dataset, matches, duplicates_file):
    duplicates = [entry for entry in first_dataset if extract_id_from_image(entry['image']) in matches]
    remaining_entries = [entry for entry in first_dataset if extract_id_from_image(entry['image']) not in matches]

    # Save duplicates to a new JSON file
    with open(duplicates_file, 'w', encoding='utf-8') as f:
        json.dump(duplicates, f, ensure_ascii=False, indent=2)

    return remaining_entries

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--first_dataset_path", type=str, default=None)
parser.add_argument("--second_dataset_path", type=str, default=None)
parser.add_argument("--duplicates_file_path", type=str, default=None)
parser.add_argument("--filtered_dataset_path", type=str, default=None)
args = parser.parse_args()
first_dataset_path = args.first_dataset_path
second_dataset_path = args.second_dataset_path
duplicates_file_path = args.duplicates_file_path
filtered_dataset_path = args.filtered_dataset_path

first_dataset = load_jsonl(first_dataset_path)
second_dataset = load_json(second_dataset_path)

matches = find_matching_instances(first_dataset, second_dataset)

remaining_entries = remove_and_save_duplicates(first_dataset, matches, duplicates_file_path)
with open(filtered_dataset_path, 'w', encoding='utf-8') as f:
    json.dump(remaining_entries, f, ensure_ascii=False, indent=2)

for match in matches:
    print(json.dumps(match, indent=2))