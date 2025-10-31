import json
import uuid
import argparse

def filter_json(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    filtered_data = {}

    for key, entry in data.items():
        if 'qa' in entry:
            entry['qa'] = [qa for qa in entry['qa'] if len(qa.get('answer', '')) <= 50]

            if len(entry['qa']) > 0:  # Only keep the entry if 'qa' is not empty
                filtered_data[key] = entry

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=4)

def filter_all_figures(data):
    for entry in data.values():
        if 'qa' in entry and 'all_figures' in entry:
            referenced_ids = {qa.get('reference') for qa in entry['qa'] if 'reference' in qa}
            entry['all_figures'] = {fig_id: fig for fig_id, fig in entry['all_figures'].items() if fig_id in referenced_ids}
    return data

#filtered_data = filter_all_figures(data)
def remove_non_figure_references(data):
    for key, entry in list(data.items()):
        if 'qa' in entry and 'all_figures' in entry:
            # Filter figures with content_type 'figure'
            entry['all_figures'] = {fig_id: fig for fig_id, fig in entry['all_figures'].items() if fig.get('content_type') == 'figure'}
            
            # Get valid figure IDs
            valid_figure_ids = set(entry['all_figures'].keys())
            
            # Filter questions referencing valid figures
            entry['qa'] = [qa for qa in entry['qa'] if qa.get('reference') in valid_figure_ids]
            
            # Remove entry if no valid figures or questions remain
            if not entry['all_figures'] or not entry['qa']:
                del data[key]
    return data

def transform_to_question_grouped(data):
    new_entries = []
    for paper_id, entry in data.items():
        if 'qa' in entry and 'all_figures' in entry:
            for qa in entry['qa']:
                figure_id = qa.get('reference')
                figure_data = entry['all_figures'].get(figure_id, {})
                new_entry = {
                    "instance_id": str(uuid.uuid4().hex),
                    "figure_id": figure_id.rsplit('.', 1)[0] if figure_id else None,  # Remove only the last file extension
                    "image_file": figure_id,
                    "caption": figure_data.get("caption"),
                    "figure_type": figure_data.get("figure_type"),
                    "compound": None,
                    "figs_numb": None,
                    "qa_pair_type": None,
                    "question": qa.get("question"),
                    "answer": qa.get("answer"),
                    "rationale": qa.get("explanation",""),
                    "answer_options": [],
                    "venue": None,
                    "categories": None,
                    "source_dataset": "spiqa",
                    "paper_id": paper_id,
                    "pdf_url": None
                }
                new_entries.append(new_entry)
    return new_entries

def remove_similar_paper_ids(data, other_dataset_path):
    with open(other_dataset_path, 'r', encoding='utf-8') as f:
        other_dataset = json.load(f)

    other_paper_ids = {entry['paper_id'] for entry in other_dataset}

    counter = 0
    filtered_data = []
    for entry in data:
        paper_id = entry['paper_id'].rsplit('v', 1)[0]  # Remove the version suffix (e.g., "v1")
        if not any(paper_id in other_id for other_id in other_paper_ids):
            filtered_data.append(entry)
        else:
            counter +=1
    print(f"Filtered out {counter} entries with similar paper IDs.")

    return filtered_data


parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str, default=None)
parser.add_argument("--output_file", type=str, default=None)
parser.add_argument("--other_dataset_path", type=str, default=None)
args = parser.parse_args()
input_file = args.input_file
output_file = args.output_file
other_dataset_path = args.other_dataset_path

with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

data = filter_all_figures(data)
data = remove_non_figure_references(data)
data = transform_to_question_grouped(data)
filtered_data = remove_similar_paper_ids(data, other_dataset_path)


with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=4)