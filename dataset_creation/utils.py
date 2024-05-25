import random
import os
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw
import numpy as np
from shapely.geometry import Polygon

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


def parse_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    annotations = []
    for annotation in root.findall('.//Annotation'):
        name = annotation.attrib['Name']
        part_of_group = annotation.attrib['PartOfGroup']
        coordinates = []
        for coord in annotation.find('Coordinates').findall('Coordinate'):
            x = float(coord.attrib['X'])
            y = float(coord.attrib['Y'])
            coordinates.append((x, y))
        annotations.append({'name': name, 'part_of_group': part_of_group, 'coordinates': coordinates})
    
    return annotations


def extract_polygon_image(slide, coordinates, output_path):
    # Calculate the bounding box
    x_coords = [coord[0] for coord in coordinates]
    y_coords = [coord[1] for coord in coordinates]
    min_x = int(min(x_coords))
    max_x = int(max(x_coords))
    min_y = int(min(y_coords))
    max_y = int(max(y_coords))
    
    # Extract the region from the WSI
    region = slide.read_region((min_x, min_y), 0, (max_x - min_x, max_y - min_y))
    
    # Convert to RGB
    region = region.convert('RGB')
    
    # Create a mask image
    mask = Image.new('L', (max_x - min_x, max_y - min_y), 0)
    draw = ImageDraw.Draw(mask)
    
    # Draw the polygon on the mask
    polygon = [(x - min_x, y - min_y) for x, y in coordinates]
    draw.polygon(polygon, outline=1, fill=1)
    mask = np.array(mask)
    
    # Apply the mask to the region
    region = np.array(region)
    region[mask == 0] = 0
    
    # Save the result
    Image.fromarray(region).save(output_path)

def is_within_patch(patch_coords, annotation_coords, patch_size):
    patch_x, patch_y = patch_coords
    patch_x_max, patch_y_max = patch_x + patch_size, patch_y + patch_size

    for x, y in annotation_coords:
        if not (patch_x <= x <= patch_x_max and patch_y <= y <= patch_y_max):
            return False
    return True


def calculate_iou(patch_coords, annotation_coords, patch_size):
    patch_polygon = Polygon([
        patch_coords,
        (patch_coords[0] + patch_size, patch_coords[1]),
        (patch_coords[0] + patch_size, patch_coords[1] + patch_size),
        (patch_coords[0], patch_coords[1] + patch_size)
    ])
    
    annotation_polygon = Polygon(annotation_coords)
    
    intersection = patch_polygon.intersection(annotation_polygon).area
    union = patch_polygon.union(annotation_polygon).area
    return intersection / union