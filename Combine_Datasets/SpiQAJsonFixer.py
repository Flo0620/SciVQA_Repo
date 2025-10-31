import json
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--input_path", type=str, default=None)
parser.add_argument("--output_path", type=str, default=None)
args = parser.parse_args()
input_path = args.input_path
output_path = args.output_path

# Read and reformat the JSON file
with open(input_path, 'r', encoding='utf-8', errors='replace') as file:
	data = json.load(file)

# Write the properly indented JSON to a new file
with open(output_path, 'w', encoding='utf-8', errors='replace') as file:
	json.dump(data, file, indent=4)

print(f"Fixed JSON saved to {output_path}")
