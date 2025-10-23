import json
import csv
import zipfile
import os

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
            answer_pred = entry.get('response', '')
            csv_writer.writerow([instance_id, str(answer_pred)])

    # Create ZIP file
    zip_file_path = 'predictions.zip'
    with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(csv_file_path, os.path.basename(csv_file_path))

    # Clean up the CSV file
    os.remove(csv_file_path)

# Example usage
# Replace 'path_to_json_file.json' with the actual path to your JSON file
generate_predictions_zip('/ltstorage/home/9schleid/scivqa/outputs/25-04-16_14:43_Qwen2.5-VL-7B-instruct-bnb-4bit-finetuned-changedParams/inference_log.json')