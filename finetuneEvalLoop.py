import subprocess
import time
import requests
import yaml
import random
server_proc = None
try:
    for lora_r, lora_alpha, lora_dropout, lr, lr_scheduler_type, output_dir, resume_training, num_epochs, model_id, onlyEval, run_checkpoint_number, run_final_model, dataset_path in reversed([
       (64, 128, 0.2, 2e-4, "linear", "/ltstorage/home/9schleid/SciVQA/unsloth/Qwen2_5_7B_r64_a128_d0_2_12096TrainSize_SameSteps", False, 2, "Qwen/Qwen2.5-VL-7B-Instruct", False, None, True, "/ltstorage/home/9schleid/SciVQA/unsloth/shared_task/smaller_train_sets/train_12096.json"),
        (64, 128, 0.2, 2e-4, "linear", "/ltstorage/home/9schleid/SciVQA/unsloth/Qwen2_5_7B_r64_a128_d0_2_9072TrainSize_SameSteps", False, 2, "Qwen/Qwen2.5-VL-7B-Instruct", False, None, True, "/ltstorage/home/9schleid/SciVQA/unsloth/shared_task/smaller_train_sets/train_9072.json"),
        (64, 128, 0.2, 2e-4, "linear", "/ltstorage/home/9schleid/SciVQA/unsloth/Qwen2_5_7B_r64_a128_d0_2_6048TrainSize_SameSteps", False, 2, "Qwen/Qwen2.5-VL-7B-Instruct", False, None, True, "/ltstorage/home/9schleid/SciVQA/unsloth/shared_task/smaller_train_sets/train_6048.json"),
      ]):

        if not onlyEval:
            # Step 1: Finetune the model
            print("Starting finetuning with params: lora_r={}, lora_alpha={}, lora_dropout={}, output_dir={}, resuming_training={}, num_epochs={}".format(
                lora_r, lora_alpha, lora_dropout, output_dir, resume_training, num_epochs))
    #
            argumentlist = ["python", "/ltstorage/home/9schleid/SciVQA/unsloth/finetuneQwenWithoutUnsloth.py", 
                                    "--lora_r", str(lora_r),
                                        "--lora_alpha", str(lora_alpha),
                                        "--lora_dropout", str(lora_dropout),
                                        "--learning_rate", str(lr),
                                        "--lr_scheduler_type", str(lr_scheduler_type),
                                        "--output_dir", str(output_dir),
                                        "--num_epochs", str(num_epochs),
                                        "--model_id", str(model_id),
                                        "--dataset_path", str(dataset_path)]
            if resume_training:
                argumentlist.append("--resume_training")
            finetune = subprocess.run(argumentlist, check=True)
            print("Finetuning completed.")

# Load base_url from defaults.yaml
        with open("/ltstorage/home/9schleid/scivqa/conf/defaults.yaml", "r") as file:
            config = yaml.safe_load(file)
            port = str(config.get("vllm_port", 5000) +random.randint(0, 1000))  # Randomize port to avoid conflicts

        if run_checkpoint_number:
            # Step 2: Start the Flask server
            print("Starting server...")
            server_proc = subprocess.Popen(["python", "/ltstorage/home/9schleid/SciVQA/unsloth/finetunedModelInferenceWithoutUnslothServer.py",
                                            "--model_id", model_id,
                                            "--adapter_path", f"{output_dir}/checkpoint-{run_checkpoint_number}",
                                            "--port", port])

            # Step 3: Wait for server to be ready
            def wait_for_server(url="http://localhost:18125/v1/health", timeout=300):
                print("Waiting for server to be ready...")
                for _ in range(timeout):
                    try:
                        response = requests.get(url)
                        if response.status_code == 200:
                            print("Server is ready!")
                            return True
                    except requests.exceptions.ConnectionError:
                        pass
                    time.sleep(5)
                raise RuntimeError("Server didn't become ready in time.")

            wait_for_server(url = f"http://localhost:{port}/v1/health")

            # Step 4: Run evaluation
            print("Running evaluation...")
            evaluate = subprocess.run(["uv", "run", "-m", "scivqa.evaluation.execution",
                                    f"model_id={model_id}",
                                    f"adapter_path={output_dir}/checkpoint-{run_checkpoint_number}",
                                    f"+num_epochs={num_epochs}",
                                    f"++base_url=http://localhost:{port}/v1/",
                                    f"+hyperparameterTuning=True"]
                                    , cwd="/ltstorage/home/9schleid/scivqa", check=True)
            print("Evaluation completed.")

            # Optional: Kill the server after evaluation
            server_proc.terminate()
            print("Server terminated.")

        if run_final_model:
            # Step 2: Start the Flask server
            print("Starting server...")
            server_proc = subprocess.Popen(["python", "/ltstorage/home/9schleid/SciVQA/unsloth/finetunedModelInferenceWithoutUnslothServer.py",
                                            "--model_id", model_id,
                                            "--adapter_path", output_dir,
                                            "--port", port])

            # Step 3: Wait for server to be ready
            def wait_for_server(url="http://localhost:18125/v1/health", timeout=300):
                print("Waiting for server to be ready...")
                for _ in range(timeout):
                    try:
                        response = requests.get(url)
                        if response.status_code == 200:
                            print("Server is ready!")
                            return True
                    except requests.exceptions.ConnectionError:
                        pass
                    time.sleep(5)
                raise RuntimeError("Server didn't become ready in time.")

            wait_for_server(url = f"http://localhost:{port}/v1/health")

            # Step 4: Run evaluation
            print("Running evaluation...")
            evaluate = subprocess.run(["uv", "run", "-m", "scivqa.evaluation.execution",
                                    f"model_id={model_id}",
                                    f"adapter_path={output_dir}",
                                    f"+num_epochs={num_epochs}",
                                    f"++base_url=http://localhost:{port}/v1/",
                                    f"+hyperparameterTuning=True"]
                                    , cwd="/ltstorage/home/9schleid/scivqa", check=True)
            print("Evaluation completed.")

            # Optional: Kill the server after evaluation
            server_proc.terminate()
            print("Server terminated.")
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    # Ensure the server is terminated in case of an error
    if server_proc:
        server_proc.terminate()
        print("Server terminated.")
        try:
            server_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            print("⚠️ Server did not shut down cleanly. Killing it.")
            server_proc.kill()