import os
import json
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

parser = argparse.ArgumentParser(description="Submit and monitor OpenAI batches in parallel.")
parser.add_argument("--batch-dir", type=str, required=True, help="Directory containing batch files.")
args = parser.parse_args()

batch_dir = args.batch_dir
log_file = os.path.join(batch_dir, "inference_log.json")
poll_interval = 15  # seconds
max_workers = 8  # adjust based on system/API limits

def upload_batch_file(file_path):
    with open(file_path, "rb") as f:
        uploaded_file = client.files.create(file=f, purpose="batch")
    print(f"Uploaded file: {file_path} → file_id: {uploaded_file.id}")
    return uploaded_file.id

def submit_batch(file_id):
    batch = client.batches.create(
        input_file_id=file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )
    print(f"Submitted batch: {batch.id}")
    return batch.id

def poll_until_complete(batch_id):
    while True:
        batch = client.batches.retrieve(batch_id)
        print(f"[{batch_id}] Status: {batch.status}")
        if batch.status in ("completed", "failed", "expired", "canceled"):
            return batch
        time.sleep(poll_interval)

def fetch_results(batch):
    if not batch.output_file_id:
        print(f"[{batch.id}] No output file.")
        return []

    output = client.files.content(batch.output_file_id)
    results = []
    for line in output.text.splitlines():
        try:
            parsed = json.loads(line)
            results.append(parsed)
        except json.JSONDecodeError:
            print(f"Failed to parse line: {line}")
    return results

def write_inference_log(results):
    log_entries = []
    for entry in results:
        custom_id = entry.get("custom_id")
        response = entry.get("response", {}).get("body", {}).get("choices", [{}])[0].get("message", {}).get("content", "")
        log_entries.append({
            "instance_id": custom_id,
            "response": response
        })

    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            try:
                existing_logs = json.load(f)
            except json.JSONDecodeError:
                existing_logs = []
    else:
        existing_logs = []

    existing_logs.extend(log_entries)

    with open(log_file, "w") as f:
        json.dump(existing_logs, f, indent=4)

def handle_batch(file_path):
    try:
        file_id = upload_batch_file(file_path)
        batch_id = submit_batch(file_id)
        batch = poll_until_complete(batch_id)

        if batch.status == "completed":
            print(f"[{batch.id}] completed. Fetching results...")
            results = fetch_results(batch)
            write_inference_log(results)
            print(f"[{batch.id}] Wrote {len(results)} entries to log.")
        else:
            print(f"[{batch.id}] Failed with status: {batch.status}")
    except Exception as e:
        print(f"[{file_path}] Error: {e}")

def main():
    batch_files = sorted([os.path.join(batch_dir, f) for f in os.listdir(batch_dir) if f.endswith(".jsonl")])

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(handle_batch, file_path) for file_path in batch_files]
        for future in as_completed(futures):
            future.result()  # to propagate exceptions if any

if __name__ == "__main__":
    main()
