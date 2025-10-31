import json
import re

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--json_path", type=str, default=None)
args = parser.parse_args()

json_path = args.json_path

# Load JSON data
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Helper function to extract unique options (A-D) from a string
def extract_options(text):
    return set(re.findall(r"([a-d])", text))

# Filter for 'finite answer set non-binary' questions
filtered = [
    entry for entry in data
    if 'finite answer set non-binary' in entry.get('meta_data', {}).get('qa_pair_type', '')
]

multi_option_response_count = 0
multi_option_reference_count = 0

for entry in filtered:
    response = entry.get('response', '').removeprefix("Answer: ").lower()
    reference = entry.get('meta_data', {}).get('reference_answer', '').lower()

    response_options = extract_options(response)
    reference_options = extract_options(reference)

    if len(response_options) > 1:
        multi_option_response_count += 1
        print(f"Response: {response} | Options: {response_options}")
    if len(reference_options) > 1:
        multi_option_reference_count += 1

print(f"Total filtered questions: {len(filtered)}")
print(f"Questions with multiple options in response: {multi_option_response_count}")
print(f"Questions with multiple options in reference_answer: {multi_option_reference_count}")