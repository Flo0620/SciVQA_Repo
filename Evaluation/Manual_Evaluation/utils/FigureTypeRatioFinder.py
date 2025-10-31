import json
from collections import Counter, defaultdict
import csv

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--json_path", type=str, default=None)
args = parser.parse_args()

json_path = args.json_path

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Count figure types, grouping those with "," into "Compound of different types"
figure_types = []
correct_counts = defaultdict(int)
total_counts = defaultdict(int)

for entry in data:
    fig_type = entry.get("context").get("prompt_vars").get("figure_type")
    #fig_type = entry.get("figure_type")
    if fig_type:
        fig_type = fig_type.lower()
        if "," in fig_type:
            fig_type_norm = "Compound Of Different Types"
        else:
            fig_type_norm = fig_type.title()
        figure_types.append(fig_type_norm)
        total_counts[fig_type_norm] += 1
        if entry.get("answer_is_correct") is True:
            correct_counts[fig_type_norm] += 1

counter = Counter(figure_types)

# Combine all figure types with 112 or less occurrences into "Other"
other_count = 0
other_correct = 0
filtered_counter = {}
filtered_correct = {}
for fig_type, count in counter.items():
    if count <= 7:
        other_count += count
        other_correct += correct_counts[fig_type]
    else:
        filtered_counter[fig_type] = count
        filtered_correct[fig_type] = correct_counts[fig_type]
if other_count > 0:
    filtered_counter["Other"] = other_count
    filtered_correct["Other"] = other_correct

total = sum(filtered_counter.values())

# Prepare sorted results
sorted_types = sorted(filtered_counter.items(), key=lambda x: x[1], reverse=True)

print("Figure Type\tCount\tPercentage\tCorrect Fraction")
for fig_type, count in sorted_types:
    percentage = (count / total) * 100 if total else 0
    correct_fraction = (filtered_correct[fig_type] / count) * 100 if count else 0
    print(f"{fig_type}\t{count}\t{percentage:.4f}%\t{correct_fraction:.4f}")

total_correct = sum(filtered_correct.values())
total_ratio = (total_correct / total) if total else 0
print(f"\nTotal Correct Answers: {total_correct}/{total} ({total_ratio:.4f})")

# Write results to CSV
csv_path = "figure_type_ratios_with_correct.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Figure Type", "Count", "Percentage", "Correct Fraction"])
    for fig_type, count in sorted_types:
        percentage = (count / total) * 100 if total else 0
        correct_fraction = (filtered_correct[fig_type] / count) * 100 if count else 0
        writer.writerow([fig_type, count, f"{percentage:.2f}%", f"{correct_fraction:.2f}"])
