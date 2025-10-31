import json
import csv
import zipfile
import os


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--json_path", type=str, default=None)
args = parser.parse_args()

json_path = args.json_path

def generate_predictions_zip(json_file_path):
    # Load JSON data
    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)

    # Prepare CSV file
    csv_file_path = 'predictions.csv'
    with open(csv_file_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        # Write header
        csv_writer.writerow(['instance_id', 'answer_pred'])
        # Write rows
        for entry in data:
            instance_id = entry.get('meta_data', {}).get('instance_id', '')
            answer_pred = entry.get('response', '').removeprefix("Answer: ").replace('\n', ' ')
            csv_writer.writerow([instance_id, str(answer_pred)])

    # Create ZIP file
    zip_file_path = 'predictions.zip'
    with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(csv_file_path, os.path.basename(csv_file_path))

    # Clean up the CSV file
    os.remove(csv_file_path)

# Example usage
generate_predictions_zip(json_path)