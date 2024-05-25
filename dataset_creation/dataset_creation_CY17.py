import h5py
import numpy as np
import pandas as pd
from PIL import Image
import openslide
from tqdm import tqdm
import os
from tiatoolbox.wsicore.wsireader import WSIReader
from utils import save_patches_in_batches, select_slides
import logging
logging.basicConfig(level=logging.INFO)


def save_patches_from_h5(h5_file_path, wsi_path, save_dir, stage="negative"):
    # Open the H5 file
    with h5py.File(h5_file_path, 'r') as h5_file:
        coords = h5_file['coords']
        logging.info(f"Number of patches: {coords.shape[0]}")

        # Open the WSI file
        wsi = openslide.OpenSlide(wsi_path)
        #slide = WSIReader.open(wsi_path)
        
        #INFO_WSI = slide.info.as_dict()
        
        if stage == "negative":
            total_patches = coords.shape[0]
            per_slide = int(args.NC/args.NWN)
            selected_slides = select_slides(per_slide,total_patches)

        else:
            selected_slides = np.arange(0, coords.shape[0])

        logging.info(f"Number of selected patches: {len(selected_slides)} for patient {wsi_path}")

        tissue_patches = []
        tissue_names = []
        # Iterate through the coordinates and visualize the patches
        for i in tqdm(selected_slides, desc="Processing slides"):  # Visualize first 5 patches
            x, y = coords[i]
            patch = wsi.read_region((x, y), 0, (256, 256))  # Assuming patch size of 256x256
            patch = patch.convert('RGB')
            
            tissue_patches.append(patch)
            tissue_names.append(f"{save_dir}/patch_{i+1}_{x}_{y}.png")
    
    save_patches_in_batches(tissue_patches, tissue_names, save_dir)


def main(args):
    stages = pd.read_csv(args.stage_csv)
    logging.info("Stages dataframe loaded")
    # filter all the files that have .tff extension in patient column of the stages dataframe
    stages = stages[stages['patient'].str.contains('.tif',case=False)]
    stages = stages[stages['stage']=="macro"]

    for i, row in tqdm(stages.iterrows(), desc="Processing stages"):
        patient = row['patient'].split('.')[0]
        stage = row['stage']
        wsi_path = f"{args.wsi_dir}/{patient}.tif"
        h5_file_path = f"{args.h5_dir}/{patient}.h5"
        
        save_dir = f"{args.save_dir}/{stage}/"

        os.makedirs(save_dir, exist_ok=True)
        logging.info(f"Processing: {patient}")

        save_patches_from_h5(h5_file_path, wsi_path, save_dir)
        #logging.info(INFO_WSI)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Visualize patches from H5 files")
    parser.add_argument("--stage_csv", type=str, help="Path to the CSV file containing the stages of the patients")
    parser.add_argument("--wsi_dir", type=str, help="Path to the directory containing the WSI files")
    parser.add_argument("--h5_dir", type=str, help="Path to the directory containing the H5 files")
    parser.add_argument("--save_dir", type=str, help="Path to the directory where the patches will be saved")
    parser.add_argument('--NWN',default=2,type=int)
    parser.add_argument('--NC',default=100,type=int)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)