This repository contains the code for the master thesis "Visual Question Answering on Scientific Charts Using Fine-Tuned Vision-Language Models".

The repository contains multiple subfolders whose contents are explained in the following, mentioning the most important scripts in each subfolder.
It also contains [encourage](https://github.com/uhh-hcds/encourage) as a sub-repository, which was used as a base in this thesis.

## Combine Datasets
This folder contains the scripts that were used to combine the ArXivQA and the SpiQA datasets with the SciVQA dataset.  
[ArxivDatasetAligner.py](https://github.com/Flo0620/SciVQA_Repo/blob/main/Combine_Datasets/ArxivDatasetAligner.py) and [SpiQaDatasetAligner.py](https://github.com/Flo0620/SciVQA_Repo/blob/main/Combine_Datasets/SpiQaDatesetAligner.py) restructure the datasets and filter out questions that do not align with the SciVQA dataset.  
[DatasetDuplicatePaperFinder.py](https://github.com/Flo0620/SciVQA_Repo/blob/main/Combine_Datasets/DatasetDuplicatePaperFinder.py) finds papers that were scraped in two of the datasets, such that these questions can be filtered out to prevent that images from the training dataset could leak into the test dataset.  

## Evaluation
This folder contains the configurations in the [conf folder](https://github.com/Flo0620/SciVQA_Repo/tree/main/Evaluation/conf).  
The [src folder](https://github.com/Flo0620/SciVQA_Repo/tree/main/Evaluation/src) contains the code to run inference on a model that runs on a server, together with the code for the automatic evaluation.
The code to deploy this server for a fine-tuned model is part of the Fine-Tuning folder and introduced later. For models from Huggingface also [start_vllm_server_as_process.py](https://github.com/Flo0620/SciVQA_Repo/blob/main/Evaluation/start_vllm_server_as_process.py) can be used.  
Furthermore, the Evaluation folder contains the code and the results of the manual evaluation in [the corresponding subfolder](https://github.com/Flo0620/SciVQA_Repo/tree/main/Evaluation/Manual_Evaluation). [manualAnnotation.py](https://github.com/Flo0620/SciVQA_Repo/blob/main/Evaluation/Manual_Evaluation/manualAnnotation.py) is the code that starts the annotation process which shows the annotator the image together with the question,the generated answer, and the reference answer. [accuracyCalculator.py](https://github.com/Flo0620/SciVQA_Repo/blob/main/Evaluation/Manual_Evaluation/accuracyCalculator.py) then calculates the accuracy per question type from those annotations.    
The subfolder [Inference_logs](https://github.com/Flo0620/SciVQA_Repo/tree/main/Evaluation/Manual_Evaluation/Inference_logs/validation%20set) contains the answers together with their annotations for the manually evaluated results.

## Fine-Tuning
This folder contains the code and the datasets for the fine-tuning of the models (the images were left away due to memory reasons, but can be downloaded from the original datasets: [SciVQA](https://huggingface.co/datasets/katebor/SciVQA), [ArXivQA](https://huggingface.co/datasets/MMInstruction/ArxivQA), [SpiQA](https://huggingface.co/datasets/google/spiqa)).  
Notably, it also contains the [version of the SciVQA dataset that was manually improved](https://github.com/Flo0620/SciVQA_Repo/blob/main/Fine-Tuning/shared_task/SciVQA_train_combined_manually_labeled.json) during this thesis.  
[finetuneQwenWithoutUnsloth.py](https://github.com/Flo0620/SciVQA_Repo/blob/main/Fine-Tuning/finetuneQwenWithoutUnsloth.py) contains the code to fine-tune a model.  
[finetunedModelInferenceWithoutUnslothServer.py](https://github.com/Flo0620/SciVQA_Repo/blob/main/Fine-Tuning/finetunedModelInferenceWithoutUnslothServer.py) starts the server that runs a fine-tuned model in order to perform inference.  

## OpenAI Inference
In this folder, the code to create the batches of questions that should be prompted to GPT-4o mini is located in [OpenAIBatchWriter.py](https://github.com/Flo0620/SciVQA_Repo/blob/main/OpenAI_Inference/OpenAIBatchWriter.py). [submit_and_monitor_batches.py](https://github.com/Flo0620/SciVQA_Repo/blob/main/OpenAI_Inference/submit_and_monitor_batches.py) then contains the code to submit the batches to OpenAI and to collect the responses.
