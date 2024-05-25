from utils import parse_xml, \
    is_within_patch, \
    calculate_iou

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Validate patches")
    parser.add_argument("--xml_path", type=str, help="Path to the XML file containing the annotations")
    parser.add_argument("--patches_path", type=str, help="Path to the JSON file containing the patches")
    parser.add_argument("--patch_size", type=int, help="Size of the patches")
    return parser.parse_args()

def validate_patches(xml_path, patches, patch_size):
    annotations = parse_xml(xml_path)
    results = []

    for patch_id, patch_data in patches.items():
        patch_coords = patch_data['coordinates']
        patch_image = patch_data['image']

        for annotation in annotations:
            annotation_coords = annotation['coordinates']
            if is_within_patch(patch_coords, annotation_coords, patch_size):
                iou = calculate_iou(patch_coords, annotation_coords, patch_size)
                results.append({'patch_id': patch_id, 'annotation_name': annotation['name'], 'iou': iou})

    return results

def main(args):
    # Example usage
    xml_path = 'path/to/annotations.xml'
    patch_size = 256  # Assuming patches are 256x256
    validation_results = validate_patches(xml_path, patches, patch_size)

    for result in validation_results:
        print(f"Patch ID: {result['patch_id']}, Annotation: {result['annotation_name']}, IoU: {result['iou']:.2f}")

if __name__ == '__main__':
    args = parse_args()
    main(args)