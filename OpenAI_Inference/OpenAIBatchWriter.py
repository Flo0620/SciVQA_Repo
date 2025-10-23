from flask import Flask, request, jsonify
import json
import hashlib
import time
import random
import os
import base64
from dotenv import load_dotenv
import sys
import faulthandler
faulthandler.enable()

app = Flask(__name__)
load_dotenv()

images_path = "./shared_task/validation_images"
batch_dir = "batches_zero_shot_baseline"
max_batch_size_bytes = 190 * 1024 * 1024  # 90 MB safety limit
port = int(os.getenv("PORT", 5000))

os.makedirs(batch_dir, exist_ok=True)

def get_current_batch_file():
    """Return the path of the current batch file, or create a new one if needed."""
    existing = sorted([
        f for f in os.listdir(batch_dir) if f.startswith("batch_") and f.endswith(".jsonl")
    ])
    
    if not existing:
        return os.path.join(batch_dir, "batch_1.jsonl")

    last_file = os.path.join(batch_dir, existing[-1])
    if os.path.getsize(last_file) < max_batch_size_bytes:
        return last_file

    # Create a new file
    last_num = int(existing[-1].split("_")[1].split(".")[0])
    new_file = os.path.join(batch_dir, f"batch_{last_num + 1}.jsonl")
    open(new_file, "a").close()
    return new_file


def convert_images(messages):
    for message in messages:
        if message.get("role") == "user":
            content = message.get("content", [])
            if isinstance(content, list):
                for item in content:
                    if item.get("type") == "image":
                        image_path = item.get("image")
                        image_file = os.path.basename(image_path)
                        image_full_path = os.path.join(images_path, image_file)
                        try:
                            with open(image_full_path, "rb") as img_f:
                                encoded = base64.b64encode(img_f.read()).decode("utf-8")
                                item.clear()
                                item["type"] = "image_url"
                                item["image_url"] = {"url": f"data:image/png;base64,{encoded}"}
                        except Exception as e:
                            print(f"Error encoding image {image_path}: {e}")


@app.route('/v1/chat/completions', methods=['POST'])
def collect_for_batch():
    try:
        data = json.loads(request.get_json())
        arrival_time = time.time()

        messages = data.get("conversation", [])
        model = data.get("model", "gpt-4o-mini")
        convert_images(messages)

        # Metadata
        prompt_id = hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()
        request_id = data.get("meta_data", {}).get("instance_id", arrival_time)

        # Format for batch file
        batch_entry = {
            "custom_id": request_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "messages": messages
            }
        }

        # Determine which batch file to write to
        batch_file_path = get_current_batch_file()
        with open(batch_file_path, "a") as f:
            f.write(json.dumps(batch_entry) + "\n")

        response = {
            "status": "queued",
            "request_id": request_id,
            "prompt_id": prompt_id,
            "batch_file": os.path.abspath(batch_file_path),
            "timestamp": arrival_time
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/v1/health")
def health():
    return "OK", 200



def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        # Allow Ctrl+C to exit normally
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    print("Uncaught exception", file=sys.stderr)
    import traceback
    traceback.print_exception(exc_type, exc_value, exc_traceback)
    sys.exit(1)  # Exit the process on unhandled exception

sys.excepthook = handle_exception

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=port)
    except Exception as e:
        print(f"Critical error running Flask server: {e}", file=sys.stderr)
        sys.exit(1)
