import h5py
import numpy as np
import pandas as pd

import openslide
from tqdm import tqdm
import os
import yaml
from tiatoolbox.wsicore.wsireader import WSIReader
from utils import save_patches_in_batches, \
    select_slides, \
    parse_xml, \
    extract_polygon_image, \
    is_within_patch, \
    calculate_iou
import logging
logging.basicConfig(level=logging.INFO)

def get_args():
    # open the config.yaml file
    with open("config.yaml", 'r') as stream:
        try:
            args = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            logging.info(exc)
    #logging.info(args)
    return args


def validate_patch(xml_path, patch_coords, patch_size=256, iou_threshold=0.1):
    if os.path.exists(xml_path):
        annotations = parse_xml(xml_path)
    else:
        logging.info(f"XML file {xml_path} does not exist.")
        return False

    for annotation in annotations:
        annotation_coords = annotation['coordinates']
        iou = calculate_iou(patch_coords, annotation_coords, patch_size)
        if iou > iou_threshold:
            logging.info(f"Patch {patch_coords} has IoU {iou:.2f} with annotation and is considered valid.")
            return True
        else:
            continue
    return False


def save_patches_from_h5(h5_file_path, wsi_path, patient, save_dir, stage, args):
    xml = f"{args['xml_dir']}/{patient}.xml"

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
            per_slide = int(args['NC']/args['NWN'])
            selected_slides = select_slides(per_slide,total_patches)

        else:
            selected_slides = np.arange(0, coords.shape[0])

        logging.info(f"Number of selected patches: {len(selected_slides)} for patient {wsi_path}")

        tissue_patches = []
        tissue_names = []
        # Iterate through the coordinates and visualize the patches
        for i in tqdm(selected_slides, desc="Processing slides"):  # Visualize first 5 patches
            x, y = coords[i]
            patch = wsi.read_region((x, y), 0, (256, 256)) 
            patch = patch.convert('RGB')
            
            if stage != "negative":
                if os.path.exists(xml):
                    if validate_patch(xml, (x,y), patch_size=256):
                        tissue_patches.append(patch)
                        tissue_names.append(f"{save_dir}/patch_{patient}_{x}_{y}.png")
                    else:
                        continue

                else:
                    continue
            else:
                tissue_patches.append(patch)
                tissue_names.append(f"{save_dir}/patch_{patient}_{x}_{y}.png")
    
    # if stage != "negative":
    #     results = validate_patches(f"{args.xml_dir}/{patient}.xml", tissue_patches, patient, patch_size=256)
    # else:
    #     results = []
    logging.info(f"Saving {len(tissue_patches)} patches")
    save_patches_in_batches(tissue_patches, tissue_names, save_dir)



def main(args):
    stages = pd.read_csv(args['stage_csv'])
    logging.info("Stages dataframe loaded")
    # filter all the files that have .tff extension in patient column of the stages dataframe
    stages = stages[stages['patient'].str.contains('.tif',case=False)]
    stages = stages[stages['stage']!='negative']
    
    # TEST: take only 3 rows of each stage for testing
    stages = stages.groupby('stage').head(3)

    validation_results = pd.DataFrame()

    for i, row in tqdm(stages.iterrows(), desc="Processing stages"):
        patient = row['patient'].split('.')[0]
        stage = row['stage']
        wsi_path = f"{args['wsi_dir']}/{patient}.tif"
        h5_file_path = f"{args['h5_dir']}/{patient}.h5"
        
        save_dir = f"{args['save_dir']}/{stage}/"

        os.makedirs(save_dir, exist_ok=True)
        logging.info(f"Processing: {stage} and {patient}")

        save_patches_from_h5(h5_file_path, wsi_path, patient, save_dir, stage, args)
        #validation_results = pd.concat([validation_results, pd.DataFrame(results)], ignore_index=True)
        #logging.info(INFO_WSI)


if __name__ == "__main__":
    args = get_args()
    main(args)