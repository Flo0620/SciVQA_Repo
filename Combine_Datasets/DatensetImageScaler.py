import os
from PIL import Image

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", type=str, default=None)
parser.add_argument("--output_dir", type=str, default=None)
args = parser.parse_args()
input_dir = args.input_dir
output_dir = args.output_dir

# Maximum number of pixels
max_pixels = 500000

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

def resize_image(image_path, output_path):
    with Image.open(image_path) as img:
        # Calculate the scaling factor to ensure the image has at most max_pixels
        width, height = img.size
        current_pixels = width * height
        if current_pixels > max_pixels:
            scale_factor = (max_pixels / current_pixels) ** 0.5
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Save the resized image
        img.save(output_path)

# Process all images in the input directory
counter = 0
for filename in os.listdir(input_dir):
    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, filename)
    
    # Check if the file is an image
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
        resize_image(input_path, output_path)
        counter += 1
        if counter % 1000 == 0:
            print(f"Processed {counter} images so far.")
    else:
        print(f"Skipped non-image file: {filename}")