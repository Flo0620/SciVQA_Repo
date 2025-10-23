import json
import os
from PIL import Image

#json_path = "/ltstorage/home/9schleid/SciVQA/unsloth/SciVQAAndSpiQAAndArXivQA/combined.json"
#
#with open(json_path, "r", encoding="utf-8") as f:
#    data = json.load(f)
#
#longest_entry = max(data, key=lambda x: len(x.get("question", "")))
#longest_question = longest_entry.get("question", "")
#
#print("Longest question (length {}):".format(len(longest_question)))
#print(longest_question)
#print("\nFull entry with longest question:")
#print(json.dumps(longest_entry, indent=2, ensure_ascii=False))

json_path = "/ltstorage/home/9schleid/SciVQA/unsloth/SciVQAAndSpiQAAndArXivQA/combined.json"
image_base_path = "/ltstorage/home/9schleid/SciVQA/unsloth/SciVQAAndSpiQAAndArXivQA/SPIQA_And_SciVQA_And_ArXivQA_train_images"

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

image_sizes = []
count_large_images = 0

for entry in data:
    image_file = entry.get("image_file", "")
    image_path = os.path.join(image_base_path, image_file)
    try:
        with Image.open(image_path) as img:
            size = img.size  # (width, height)
            area = size[0] * size[1]
            image_sizes.append((image_file, size, area))
            if area > 500_000:
                count_large_images += 1
    except Exception as e:
        continue  # Skip files that can't be opened


# Sort by area (smallest to largest)
image_sizes.sort(key=lambda x: x[2])

# Print as list of (filename, (width, height))
for image_file, size, area in image_sizes:
    print((image_file, size))

print(f"Number of images larger than 500k pixels: {count_large_images}")
