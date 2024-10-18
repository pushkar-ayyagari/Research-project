import os
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np

# Directories
dataset_dir = 'D:\\Data\\VirajaProject\\pythonProject1'  # Base directory for the dataset
images_dir = os.path.join(dataset_dir, 'images')
annotations_dir = os.path.join(dataset_dir, 'annotations')

# Image size
IMG_SIZE = (128, 128)


# Image processing function (Resizes and converts RGBA to RGB if necessary)
def process_image(image_path):
    """Resizes and processes the image, converting RGBA to RGB if necessary."""
    image = Image.open(image_path)

    # Convert RGBA to RGB if the image has 4 channels
    if image.mode == 'RGBA':
        image = image.convert('RGB')

    image = image.resize(IMG_SIZE)
    return np.array(image)


# Bounding box extraction function
def extract_bounding_boxes(xml_path):
    """Extracts bounding boxes and class labels from the XML annotation file."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    bounding_boxes = []

    for obj in root.findall('object'):
        label = obj.find('name').text
        bbox = obj.find('bndbox')

        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)

        # Append the label and bounding box coordinates as a tuple
        bounding_boxes.append((label, (xmin, ymin, xmax, ymax)))

    return bounding_boxes


# Normalize bounding box coordinates relative to the image size
def normalize_bounding_boxes(image, bounding_boxes):
    """Normalizes bounding boxes relative to the image dimensions."""
    height, width, _ = image.shape
    normalized_boxes = []

    for label, (xmin, ymin, xmax, ymax) in bounding_boxes:
        xmin_norm = xmin / width
        ymin_norm = ymin / height
        xmax_norm = xmax / width
        ymax_norm = ymax / height

        normalized_boxes.append((label, (xmin_norm, ymin_norm, xmax_norm, ymax_norm)))

    return normalized_boxes


# Load dataset function with bounding box extraction and normalization
def load_dataset_with_bounding_boxes(images_dir, annotations_dir):
    """Loads images and bounding boxes, processes them into Numpy arrays."""
    images = []
    all_bounding_boxes = []

    for img_file in os.listdir(images_dir):
        if img_file.endswith(".jpg") or img_file.endswith(".png"):
            img_path = os.path.join(images_dir, img_file)
            xml_file = img_file.replace('.jpg', '.xml').replace('.png', '.xml')
            xml_path = os.path.join(annotations_dir, xml_file)

            if os.path.exists(xml_path):
                # Process image
                img = process_image(img_path)

                # Extract bounding boxes from the XML file
                bboxes = extract_bounding_boxes(xml_path)

                # Normalize bounding boxes relative to the image size
                normalized_bboxes = normalize_bounding_boxes(img, bboxes)

                # Append image and its corresponding bounding boxes
                images.append(img)
                all_bounding_boxes.append(normalized_bboxes)

    return np.array(images), all_bounding_boxes


# Main execution
if __name__ == "__main__":
    # Load the dataset with bounding boxes
    images, bounding_boxes = load_dataset_with_bounding_boxes(images_dir, annotations_dir)

    print(f"Loaded {len(images)} images.")
    print(f"Bounding boxes for the first image: {bounding_boxes[0]}")
    print(f"Total images with multiple bounding boxes: {sum(len(bboxes) > 1 for bboxes in bounding_boxes)}")
