"""Module for evaluation of QA datasets."""

from pathlib import Path

import hydra
import litellm
import mlflow
import mlflow.data.pandas_dataset
import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv
from encourage.llm.inference_runner import BatchInferenceRunner
from encourage.prompts import PromptCollection
from encourage.prompts.context import Context
from encourage.prompts.meta_data import MetaData
from vllm import SamplingParams

from scivqa.evaluation.config import Config
from scivqa.evaluation.evaluation_Copy import main as evaluation
from scivqa.utils.file_manager import FileManager
from scivqa.utils.flatten_dict import flatten_dict
import requests
import json
from tqdm import tqdm
from jinja2 import Template


config_path = str((Path(__file__).parents[3] / "conf").resolve())


def prepare_data(cfg, df: pd.DataFrame) -> tuple[list, list]:
    """Prepare data for the inference runner."""
    meta_datas = []
    contexts = []
    few_shot_image_files = []
    if cfg.apply_few_shot:
        with open(cfg.few_shot_dataset_path, "r") as file:
            few_shot_dict = json.load(file)
    for _, row in df.iterrows():
        meta_data = MetaData(
            {
                "figure_id": row["figure_id"],
                "image_path": row["image_file"],
                "question": row["question"],
                "reference_answer": row["answer"],
                "caption": row["caption"],
                "qa_pair_type": row["qa_pair_type"],
                "categories": row["categories"],
                "instance_id": row["instance_id"],
            }
        )
        meta_datas.append(meta_data)

        if cfg.apply_few_shot:
            # Determine whether to find a matching row with or without answer options
            if len(row["answer_options"]) > 0:
                # Find the first entry in few_shot_dict with answer options and matching figure_type
                matching_row = next(
                item for item in few_shot_dict
                #if len(item["answer_options"]) > 0 and item["figure_type"] == row["figure_type"]
                if len(item["answer_options"]) > 0
                )
            else:
                # Find the first entry in few_shot_dict without answer options and matching figure_type
                matching_row = next(
                item for item in few_shot_dict
                #if len(item["answer_options"]) == 0 and item["figure_type"] == row["figure_type"]
                if len(item["answer_options"]) == 0
                )

            # Prepare the values for rendering the j2 template
            few_shot_image_files.append(matching_row["image_file"])
            template_values = {
                "question": matching_row["question"],
                "caption": matching_row["caption"],
                "image_path": matching_row["image_file"],
                "answer_options": "\n".join(
                [f"{k}: {v}" for d in matching_row["answer_options"] for k, v in d.items() if v is not None]
                )
                if len(matching_row["answer_options"]) != 0
                else None,
                "answer": matching_row["answer"],
            }

            # Render the j2 template
            few_shot_template = FileManager(cfg.few_shot_template).read()
            rendered_few_shot = Template(few_shot_template).render(**template_values)

        # Create the context
        context_params = {
            "qa_pair_type": row["qa_pair_type"],
            "answer_options": "\n".join(
            [f"{k}: {v}" for d in row["answer_options"] for k, v in d.items() if v is not None]
            )
            if len(row["answer_options"]) != 0
            else None,
            "caption": row["caption"],
            "compound": row["compound"],
            "figure_type": row["figure_type"],
        }
        if cfg.apply_few_shot:
            context_params["few_shot"] = rendered_few_shot

        context = Context.from_prompt_vars(context_params)
        contexts.append(context)
    return meta_datas, contexts, few_shot_image_files


@hydra.main(version_base=None, config_path=config_path, config_name="defaults2")
def main(cfg: Config) -> None:
    """Main function for evaluation of QA datasets."""
    # Load dataset from Huggingface
    load_dotenv(".env")
    qa_dataset = (
        load_dataset(cfg.dataset.name, split=cfg.dataset.split)
        .to_pandas()
        #pd.read_json("/ltstorage/home/9schleid/scivqa/data/arxiv_qa/filteredDatasetLCSS7_minus_6Processed_Test_set_ohne_figures_aus_train_set.json")
        #pd.read_json("/ltstorage/home/9schleid/scivqa/data/spiqa/SPIQA_test_split.json")
        #pd.read_json("/ltstorage/home/9schleid/scivqa/data/test_2025-07-03_09-01.json")
        .reset_index(drop=True)
    )
    #qa_dataset = qa_dataset[qa_dataset['answer_options'].apply(lambda x: len(x) > 0)].reset_index(drop=True)
    #qa_dataset = qa_dataset[qa_dataset['qa_pair_type'].apply(lambda x: "infinite answer set non-visual" in x)].reset_index(drop=True)
    #qa_dataset = qa_dataset[qa_dataset['qa_pair_type'].apply(lambda x: "infinite answer set visual" in x)].reset_index(drop=True)
    #qa_dataset = qa_dataset[qa_dataset['qa_pair_type'].apply(lambda x: "non-binary non-visual" in x)].reset_index(drop=True)
    #qa_dataset = qa_dataset[qa_dataset['qa_pair_type'].apply(lambda x: "non-binary visual" in x)].reset_index(drop=True)
    #qa_dataset = qa_dataset[qa_dataset['qa_pair_type'].apply(lambda x: " binary non-visual" in x)].reset_index(drop=True)
    #qa_dataset = qa_dataset[qa_dataset['qa_pair_type'].apply(lambda x: " binary visual" in x)].reset_index(drop=True)
    #qa_dataset = qa_dataset[qa_dataset['qa_pair_type'].apply(lambda x: "unanswerable" in x)].reset_index(drop=True)
    #qa_dataset = qa_dataset[qa_dataset['qa_pair_type'].apply(lambda x: " visual" in x)].reset_index(drop=True)



    litellm._logging._disable_debugging()
    mlflow.openai.autolog()

    if cfg.use_vllm:
        sampling_params = SamplingParams(
            temperature=cfg.model.temperature, max_tokens=cfg.model.max_tokens
        )
        runner = BatchInferenceRunner(sampling_params, cfg.model.model_name, base_url=cfg.base_url)
    sys_prompt = FileManager(cfg.dataset.sys_prompt_path).read()

    ## Run the Inference
    mlflow.set_tracking_uri(cfg.mlflow.uri)
    mlflow.set_experiment(experiment_name=cfg.mlflow.experiment_id)

    with mlflow.start_run():
        mlflow.log_params(flatten_dict(cfg))
        mlflow.log_params({"dataset_size": len(qa_dataset)})
        mlflow.log_input(
            mlflow.data.pandas_dataset.from_pandas(
                qa_dataset.drop(columns=["answer_options"]), name=cfg.dataset.name
            ),
            context="inference",
        )

        with mlflow.start_span(name="root"):
            meta_datas, contexts, few_shot_image_files= prepare_data(cfg, qa_dataset)

            image_paths = [
                [f"{cfg.few_shot_images_path}/{few_shot_image_files[i]}", f"{cfg.dataset.input_dir}/{cfg.dataset.split}/{qa_dataset['image_file'][i]}"]
                if cfg.apply_few_shot else
                [f"{cfg.dataset.input_dir}/{cfg.dataset.split}/{qa_dataset['image_file'][i]}"]
                for i in range(len(qa_dataset))
            ]

            prompt_collection = PromptCollection.create_image_prompts(
                    sys_prompts=sys_prompt,
                    user_prompts=qa_dataset["question"].tolist(),
                    image_paths=image_paths,
                    meta_datas=meta_datas,
                    contexts=contexts,
                    template_name=cfg.template_name,
                    use_image_path=not cfg.use_vllm,
                )
            if cfg.use_vllm:
                responses = runner.run(prompt_collection)

                json_dump = [response.to_dict(truncated = False) for response in responses.response_data]
                for response in json_dump:
                    response["response"] = response["response"].removeprefix("Answer: ")

            else:
                responses = []
                for i, prompt in enumerate(tqdm(prompt_collection, desc="Processing Prompts")):
                    url = f"{cfg.base_url}/chat/completions"
                    headers = {"Content-Type": "application/json"}
                    data = prompt.to_json()
                    
                    # Correct the data format to clean json and remove the unnecessary dialog key 
                    data = json.loads(data)
                    messages = json.loads(data.get("conversation")).get("dialog")
                    data["conversation"] = messages
                    data["model"] = cfg.model.model_name
                    data = json.dumps(data)

                    response = requests.post(url, headers=headers, json=data)
                    if response.status_code == 200:
                        responses.append(response.json())
                    else:
                        print(f"Error: {response.status_code}, {response.text}")
                json_dump= responses
                    

        # Save the output to hydra folder0
        FileManager(cfg.output_folder + "/inference_log.json").dump_json(
            json_dump, pydantic_encoder=True
        )
        if cfg.use_vllm:
            json_dump = [flatten_dict(response.to_dict(truncated = False)) for response in responses.response_data]
            for response in json_dump:
                    response["response"] = response["response"].removeprefix("Answer: ")
        else:
            json_dump = [flatten_dict(response) for response in responses]

        active_run = mlflow.active_run()
        run_name = active_run.info.run_name if active_run else "responses"
        mlflow.log_table(data=pd.DataFrame(json_dump), artifact_file=f"{run_name}.json")

        # Evaluate the retrieval
        evaluation(cfg)


if __name__ == "__main__":
    main()
