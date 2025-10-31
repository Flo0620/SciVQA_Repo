import json
import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str, default=None)
parser.add_argument("--output_file", type=str, default=None)
args = parser.parse_args()
input_file = args.input_file
output_file = args.output_file

# Load the JSON data
with open(input_file, 'r', encoding='utf-8') as file:
    data = json.load(file)

# Process the data
processed_data = []
for entry in data:
    # Filter out entries without exactly 4 answer options
    if "options" not in entry or len(entry["options"]) != 4:
        continue

    # Add "source_dataset" key
    entry["source_dataset"] = "arxivqa"
    
    # Modify "label" key
    if "label" in entry:
        match = re.search(r'[A-D]', entry["label"])
        if match:
            entry["answer"] = match.group(0)  # Assign the first occurrence of A, B, C, or D
        del entry["label"]  # Remove the old "label" key
    
    # Add "qa_pair_type" key
    entry["qa_pair_type"] = "closed-ended finite answer set non-binary"
    
    # Rename "options" key to "answer_options"
    entry["answer_options"] = entry["options"]
    del entry["options"]

    entry["instance_id"] = entry["id"]
    del entry["id"]

    entry["categories"] = "nan"

    # Rename "image" key to "image_file" and update its value
    if "image" in entry:
        entry["image_file"] = entry["image"].replace("images/", "arxivqa_images/")
        del entry["image"]
    
    entry["figure_id"] = entry["image_file"].split("/")[-1].rsplit(".", 1)[0] if "image_file" in entry else None

    entry["caption"] = None
    entry["compound"] = None
    entry["figure_type"] = None

    # Transform "answer_options" into a list of dictionaries with keys A, B, C, D
    transformed_options = []
    valid_format = True

    for i, option in enumerate(entry["answer_options"]):
        # Extract the first group of characters until the first space
        match = re.match(r'^(\S+)\s', option)
        if match:
            prefix = match.group(1).strip().lower()
            expected_letter = chr(97 + i)  # 'a', 'b', 'c', 'd'

            # Check if the prefix contains the expected letter
            if expected_letter in prefix and all(c.isalpha() or not c.isalnum() for c in prefix):
                # Remove the prefix and the space
                transformed_options.append({chr(65 + i): option[len(match.group(0)):].strip()})
            else:
                valid_format = False
                break
        else:
            valid_format = False
            break

    # If valid format, replace "answer_options" with the transformed list of dictionaries
    if valid_format:
        entry["answer_options"] = transformed_options
        processed_data.append(entry)

# Save the processed data
with open(output_file, 'w', encoding='utf-8') as file:
    json.dump(processed_data, file, indent=4, ensure_ascii=False)

print(f"Processed data saved to {output_file}")
print(f"Number of entries in the processed data: {len(processed_data)}")
