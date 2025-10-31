import json
import base64
from PIL import Image
import random
from jinja2 import Template
import yaml
from types import SimpleNamespace

class DatasetLoader:
    def __init__(self, dataset_json_file_path, filter_for_dataset=None, image_dir=None):
        with open(dataset_json_file_path, 'r') as file:
            self.data = json.load(file)
        if filter_for_dataset is not None:
            print(f"Filtering dataset for {filter_for_dataset}")
            self.data = [item for item in self.data if item.get('source_dataset','') == filter_for_dataset]
        with open('/ltstorage/home/9schleid/scivqa/conf/defaults.yaml', 'r') as cfg_file:
            self.cfg = yaml.safe_load(cfg_file)
        if self.cfg['apply_few_shot']:
            with open(self.cfg['few_shot_dataset_path'], "r") as file:
                self.few_shot_dict = json.load(file)
        self.index = 0
        self.image_dir = image_dir

        dataset_name = next(item['dataset'] for item in self.cfg['defaults'] if 'dataset' in item)
        dataset_config_path = f"/ltstorage/home/9schleid/scivqa/conf/dataset/{dataset_name}.yaml"
        with open(dataset_config_path, 'r') as dataset_cfg_file:
            self.dataset_cfg = yaml.safe_load(dataset_cfg_file)
        with open(self.dataset_cfg['sys_prompt_path'], 'r') as sys_prompt_file:
            self.system_prompt = sys_prompt_file.read()
        with open(self.cfg["template_folder"] + self.cfg['template_name'], 'r') as user_prompt_file:
            self.user_prompt = user_prompt_file.read()
            
    def getConvertedDataset(self, addAnswerPrefix = False):
        converted_dataset = []
        for i, sample in enumerate(self.data): 
            if (i+1) % 5000 == 0:
                print(f"Processing sample {i+1}/{len(self.data)}")
            conversation = self.convertToConversation(sample, addAnswerPrefix)
            converted_dataset.append(conversation)
        random.seed(42)
        random.shuffle(converted_dataset)

        return converted_dataset

    def getNextSample(self):
        if self.index < len(self.data):
            sample = self.data[self.index]
            self.index += 1
            return sample
        else:
            raise StopIteration("No more samples available.")
        
    def getNthSample(self, index):
        if 0 <= index < len(self.data):
            return self.data[index]
        else:
            raise IndexError("Index out of range.")
        
    def getSampleImage(self, sample):
        try:
            image_path = f"{self.image_dir}/{sample['image_file']}" if self.image_dir else sample['image_file']
            return Image.open(image_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Image file {sample['image_file']} not found.")
        
    def convertToConversation(self, sample, addAnswerPrefix = False):
        try:
            image_path = f"{self.image_dir}/{sample['image_file']}" if self.image_dir else sample['image_file']
            image_data = Image.open(image_path)

            if self.cfg['apply_few_shot']:
                # Determine whether to find a matching row with or without answer options
                if len(sample["answer_options"]) > 0:
                    # Find the first entry in few_shot_dict with answer options 
                    matching_row = next(
                    item for item in self.few_shot_dict
                    if len(item["answer_options"]) > 0)
                else:
                    # Find the first entry in few_shot_dict without answer options
                    matching_row = next(
                    item for item in self.few_shot_dict
                    if len(item["answer_options"]) == 0)

                # Prepare the values for rendering the j2 template
                few_shot_image_file_name = matching_row["image_file"]
                template_values = {
                    "question": matching_row["question"],
                    "caption": matching_row["caption"],
                    "answer_options": "\n".join(
                    [f"{k}: {v}" for d in matching_row["answer_options"] for k, v in d.items() if v is not None]
                    )
                    if len(matching_row["answer_options"]) != 0
                    else None,
                    "answer": f"Answer: {matching_row["answer"]}" if addAnswerPrefix else matching_row["answer"],
                }

                # Render the j2 template
                with open(self.cfg['few_shot_template'], "r") as template_file:
                    few_shot_template = template_file.read()
                rendered_few_shot = Template(few_shot_template).render(**template_values)

            answer_options = ""
            if sample.get('qa_pair_type', '') is not None and "finite answer set non-binary" in sample.get('qa_pair_type', ''):
                answer_options = "\n".join(
                    [f"{key}: {value}" for d in sample['answer_options'] for key, value in d.items()]
                )

            template = Template(self.user_prompt)
            question_content = template.render(caption = sample['caption'], user_prompt=[SimpleNamespace(text = sample['question'])], answer_options=answer_options, few_shot = rendered_few_shot if self.cfg['apply_few_shot'] else None)
            user_content = [
                {"type": "text", "text": question_content}
            ]
            if self.cfg['apply_few_shot']:
                user_content.append({"type": "image", "image": Image.open(f"{self.cfg['few_shot_images_path']}/{few_shot_image_file_name}")})
            user_content.append({"type": "image", "image": image_data})

            conversation = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": self.system_prompt}
                    ]
                },
                {
                    "role": "user",
                    "content": user_content
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": sample['answer']}
                    ]
                }
            ]
            return conversation
        except FileNotFoundError:
            raise FileNotFoundError(f"Image file {sample['image_file']} not found.")