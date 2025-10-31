import json
from collections import defaultdict


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--file_path", type=str, default=None)
args = parser.parse_args()

file_path = args.file_path

# Load the JSON data
with open(file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

stats_overall = {"total": 0, "correct": 0}
# Initialize a dictionary to store counts for each subcategory
subcategory_stats = defaultdict(lambda: {"total": 0, "correct": 0})

# Process each entry
for entry in data:
    qa_pair_type = entry.get("meta_data", {}).get("qa_pair_type", "unknown")
    if qa_pair_type == "unknown":
        qa_pair_type = entry.get("qa_pair_type", "unknown")
    if qa_pair_type == "unknown":
        print(entry)
    is_correct = entry.get("answer_is_correct", False)
    
    stats_overall["total"] += 1
    if is_correct:
        stats_overall["correct"] += 1
    # Update counts for the subcategory
    subcategory_stats[qa_pair_type]["total"] += 1
    if is_correct:
        subcategory_stats[qa_pair_type]["correct"] += 1

# Calculate and print the percentage of correct answers and total questions for each subcategory
for subcategory, stats in subcategory_stats.items():
    total = stats["total"]
    correct = stats["correct"]
    percentage_correct = (correct / total * 100) if total > 0 else 0
    print(f"Subcategory: {subcategory}, Total Questions: {total}, Percentage Correct: {percentage_correct:.2f}%")
total_number_of_questions = stats_overall["total"]
percentage_correct = (stats_overall["correct"] / total_number_of_questions * 100) if total_number_of_questions > 0 else 0
print(f"Overall: Total Questions: {total_number_of_questions}, Percentage Correct: {percentage_correct:.2f}%")