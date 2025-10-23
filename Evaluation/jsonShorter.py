import json

# Input and output file paths
input_file = '/ltstorage/home/9schleid/scivqa/outputs/25-04-23_12:43_gpt-4o-mini/inference_log.json'
output_file = '/ltstorage/home/9schleid/scivqa/outputs/25-04-23_12:43_gpt-4o-mini/inference_log_shortend.json'

# Load the JSON data
with open(input_file, 'r') as f:
    data = json.load(f)

# Function to recursively replace "url" key content
def replace_url_content(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "url":
                obj[key] = "base64encodedImage"
            else:
                replace_url_content(value)
    elif isinstance(obj, list):
        for item in obj:
            replace_url_content(item)

# Apply the replacement
replace_url_content(data)

# Save the modified JSON
with open(output_file, 'w') as f:
    json.dump(data, f, indent=4)