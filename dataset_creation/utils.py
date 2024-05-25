import random
import os

def select_slides(x,total):
    selected_slides = []
    while len(selected_slides)<x:
        id = random.randint(0, total-1)
        if id not in selected_slides:
            selected_slides.append(id)
    return(selected_slides)

def save_patches_in_batches(tissue_patches, tissue_names, output_folder, batch_size=100):
    for i in range(0, len(tissue_patches), batch_size):
        batch_patches = tissue_patches[i:i + batch_size]
        batch_names = tissue_names[i:i + batch_size]
        for patch, name in zip(batch_patches, batch_names):
            patch.save(os.path.join(output_folder, name))