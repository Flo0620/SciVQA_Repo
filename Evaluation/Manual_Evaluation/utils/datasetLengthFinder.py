import json

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--json_file_path", type=str, default=None)
args = parser.parse_args()

json_file_path = args.json_file_path

def count_json_entries(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            if isinstance(data, list):
                return len(data)
            elif isinstance(data, dict):
                return len(data.keys())
            else:
                print("Unsupported JSON structure.")
                return 0
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return 0
    except json.JSONDecodeError:
        print(f"Error decoding JSON in file: {file_path}")
        return 0

def count_questions_of_type(file_path, question_type):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            if isinstance(data, list):
                return sum(1 for entry in data if entry.get('qa_pair_type', '') is not None and question_type in entry.get('qa_pair_type', ''))
            elif isinstance(data, dict):
                return sum(1 for key, value in data.items() if value.get('qa_pair_type', '') is not None and question_type in value.get('qa_pair_type', ''))
            else:
                print("Unsupported JSON structure.")
                return 0
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return 0
    except json.JSONDecodeError:
        print(f"Error decoding JSON in file: {file_path}")
        return 0

if __name__ == "__main__":
    entry_count = count_json_entries(json_file_path)
    print(f"Number of entries in the JSON file: {entry_count}")
    mult_choice_count = count_questions_of_type(json_file_path, "non-binary")
    print(f"Number of multiple choice questions in the JSON file: {mult_choice_count}")