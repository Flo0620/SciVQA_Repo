import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import json
import pathlib


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--image_folder", type=str, default=None)
parser.add_argument("--predictions_file_path", type=str, default=None)
parser.add_argument("--old_annotated_file_path", type=str, default=None)
parser.add_argument("--annotated_file_path", type=str, default=None)
args = parser.parse_args()

image_folder = args.image_folder
old_annotated_file_path = args.old_annotated_file_path
annotated_file_path = args.annotated_file_path
predictions_file_path = args.predictions_file_path

# Load the JSON file
with open(predictions_file_path, "r", encoding="utf-8") as file:
    data = json.load(file)

only_multiple_choice_questions = True

old_gpt_format = False

# Create the annotated file if it does not exist
if not pathlib.Path(annotated_file_path).exists():
    with open(annotated_file_path, "w", encoding="utf-8") as f:
        json.dump([], f, ensure_ascii=False, indent=4)
with open(old_annotated_file_path, "r", encoding="utf-8") as file:
    annotated_data = json.load(file)

def wrap_text(text, max_width=50):
    words = text.split()
    lines = []
    current_line = []
    current_length = 0
    for word in words:
        if current_length + len(word) + 1 > max_width:
            lines.append(" ".join(current_line))
            current_line = [word]
            current_length = len(word)
        else:
            current_line.append(word)
            current_length += len(word) + 1
    if current_line:
        lines.append(" ".join(current_line))
    return "\n".join(lines)

labeled_data = []
number_of_images = len(data)
# Extract the relevant question and response
for i, entry in enumerate(data):
    qa_pair_type = entry.get("meta_data", {}).get("qa_pair_type", "unknown")
    if qa_pair_type == "unknown":
        qa_pair_type = entry.get("qa_pair_type", "unknown")

    if not old_gpt_format:
        if ("finite answer set non-binary" not in qa_pair_type or not only_multiple_choice_questions) and any(annotated_entry["meta_data"]["instance_id"] == entry["meta_data"]["instance_id"] and annotated_entry["answer_is_correct"]!=None for annotated_entry in annotated_data):
            labeled_data.append([annotated_entry for annotated_entry in annotated_data if annotated_entry["meta_data"]["instance_id"] == entry["meta_data"]["instance_id"] and annotated_entry["answer_is_correct"]!=None][0])
            continue  # Skip already annotated entries
    else:
        if ("finite answer set non-binary" not in qa_pair_type or not only_multiple_choice_questions) and any(annotated_entry["instance_id"] == entry["instance_id"] and annotated_entry["answer_is_correct"]!=None for annotated_entry in annotated_data):
            labeled_data.append([annotated_entry for annotated_entry in annotated_data if annotated_entry["instance_id"] == entry["instance_id"] and annotated_entry["answer_is_correct"]!=None][0])
            continue  # Skip already annotated entries

    print(f"Entry {i + 1}/{number_of_images}")


    if old_gpt_format:
        image_file = entry["image_file"]
        question = entry["question"]
        reference_answer = entry["answer"]
        response = entry["response"].removeprefix("Answer: ")
    else:
        image_file = entry["meta_data"]["image_path"]
        question = entry["meta_data"]["question"]
        reference_answer = entry["meta_data"]["reference_answer"]
        #image_file = entry["image_file"]
        #question = entry["question"]
        #reference_answer = entry["answer"]
        response = entry["response"].removeprefix("Answer: ")
        #if not "infinite" in entry["qa_pair_type"]:
        #    continue

    if response.lower() == reference_answer.lower():
        entry["answer_is_correct"] = True
        labeled_data.append(entry)
        continue
    
    if reference_answer.lower() == response.rstrip('.').lower() and reference_answer.lower() in ["yes", "no"]:
        entry["answer_is_correct"] = True
        print("Correct Yes/No with '.'")
        labeled_data.append(entry)
        continue
    if reference_answer.lower() in ["yes", "no"] and response.lower() in ["yes", "no"]:
        entry["answer_is_correct"] = False
        labeled_data.append(entry)
        continue

    if reference_answer.lower() == "yes" and response.lower().startswith("yes") or reference_answer.lower() == "no" and response.lower().startswith("no"):
        entry["answer_is_correct"] = True
        print("Yes/No with explanation starting with correct yes/no value.")
        labeled_data.append(entry)
        continue

    if reference_answer.lower() in ["a", "b", "c", "d"] and response.lower() in ["a", "b", "c", "d"]:
        entry["answer_is_correct"] = False
        labeled_data.append(entry)
        continue
    
    if "finite answer set non-binary" in qa_pair_type:
        # Normalize by removing spaces and commas, then sort letters
        def normalize_and_sort(s):
            normalized = s.replace(' ', '').replace(',', '').lower()
            return ''.join(sorted(normalized))
        if normalize_and_sort(response) == normalize_and_sort(reference_answer):
            print("Correct multiple choice: ", response, reference_answer)
            entry["answer_is_correct"] = True
            labeled_data.append(entry)
            continue

    if reference_answer.lower() == "It is not possible to answer this question based only on the provided data.".lower():
        entry["answer_is_correct"] = False
        labeled_data.append(entry)
        continue

    if response.lower() == "It is not possible to answer this question based only on the provided data.".lower() and reference_answer.lower() != "It is not possible to answer this question based only on the provided data.".lower():
        entry["answer_is_correct"] = False
        print("Falsely classified as unanswerable.")
        labeled_data.append(entry)
        continue

    if not old_gpt_format:
        answer_options = entry["context"]["prompt_vars"]["answer_options"]
    #answer_options = entry["answer_options"]
    image_path = os.path.join(image_folder, image_file)
    
    # Load and display the image
    img = mpimg.imread(image_path)

    # Create the plot
    plt.figure(figsize=(10, 8))
    plt.imshow(img)
    plt.axis('off')  # Hide axes

    # Add the question and response as text
    # Function to add line breaks for long text
    def wrap_text(text, max_width=100):
        words = text.split()
        lines = []
        current_line = []
        current_length = 0
        for word in words:
            if current_length + len(word) + 1 > max_width:
                lines.append(" ".join(current_line))
                current_line = [word]
                current_length = len(word)
            else:
                current_line.append(word)
                current_length += len(word) + 1
        if current_line:
            lines.append(" ".join(current_line))
        return "\n".join(lines)

    # Wrap the question and response text
    wrapped_question = wrap_text(question)
    plt.title(wrapped_question)

    wrapped_response = wrap_text(response)

    # Wrap the reference answer text
    wrapped_reference_answer = wrap_text(reference_answer)

    # Adjust vertical spacing dynamically based on the length of the wrapped response
    response_y_position = -0.1 - 0.05 * wrapped_response.count('\n')
    reference_answer_y_position = response_y_position - 0.1 - 0.05 * wrapped_reference_answer.count('\n')

    plt.text(0.5, response_y_position, f"Response: {wrapped_response}", fontsize=12, ha='center', transform=plt.gca().transAxes)
    plt.text(0.5, reference_answer_y_position, f"Reference Answer: {wrapped_reference_answer}", fontsize=12, ha='center', transform=plt.gca().transAxes)
    print("-------------------------------------------------------------------------------")
    print(f"Response: {response}")
    print(f"Reference Answer: {reference_answer}")

    #print(entry["raw_response"])

    # Show the plot
    plt.tight_layout()
    def on_key(event):
        if event.key == 'left':
            entry["answer_is_correct"] = False
            labeled_data.append(entry)
            plt.close()
        elif event.key == 'right':
            entry["answer_is_correct"] = True
            labeled_data.append(entry)
            plt.close()

    # Connect the key press event to the handler
    fig = plt.gcf()
    fig.canvas.mpl_connect('key_press_event', on_key)

    plt.show()

    # Save the modified data to a new JSON file after all entries are processed
    with open(annotated_file_path, "w", encoding="utf-8") as output_file:
        json.dump(labeled_data, output_file, ensure_ascii=False, indent=4)

# Save the modified data to a new JSON file after all entries are processed
with open(annotated_file_path, "w", encoding="utf-8") as output_file:
    json.dump(labeled_data, output_file, ensure_ascii=False, indent=4)