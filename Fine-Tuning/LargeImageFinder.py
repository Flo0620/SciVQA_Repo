import json
import os
from PIL import Image

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--json_path", type=str, default=None)
parser.add_argument("--image_base_path", type=str, default=None)
args = parser.parse_args()
json_path = args.json_path
image_base_path = args.image_base_path

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
