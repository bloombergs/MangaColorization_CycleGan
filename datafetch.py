import os
import shutil
from datasets import load_dataset
from PIL import Image
import random

dataset = load_dataset("MichaelP84/manga-colorization-dataset", split="train")

base_dir = "/kaggle/working" 
bw_dir = os.path.join(base_dir, "bw_images")
color_dir = os.path.join(base_dir, "color_images")

os.makedirs(bw_dir, exist_ok=True)
os.makedirs(color_dir, exist_ok=True)
def save_image(image, path):
    image.save(path)

num_images = 500

indices = list(range(len(dataset)))
random.shuffle(indices)

selected_indices = indices[:num_images]

for i, index in enumerate(selected_indices):
    example = dataset[index]

    bw_image = example["bw_image"]
    color_image = example["color_image"]

    bw_filename = f"bw_{i}.png"
    color_filename = f"color_{i}.png"

    bw_path = os.path.join(bw_dir, bw_filename)
    color_path = os.path.join(color_dir, color_filename)

    save_image(bw_image, bw_path)
    save_image(color_image, color_path)

print(f"Saved {num_images} black and white images to {bw_dir}")
print(f"Saved {num_images} color images to {color_dir}")
print("Completed!")
