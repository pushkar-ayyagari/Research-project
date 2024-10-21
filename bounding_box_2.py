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


# Image processing function
def process_image(image_path):
    """Resizes and processes the image."""
    image = Image.open(image_path)
    image = image.resize(IMG_SIZE)
    return np.array(image)


# Bounding box extraction function
def extract_bounding_boxes(xml_path):
    """Extracts bounding box coordinates from the XML annotation file."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    bounding_boxes = []
    for obj in root.findall('object'):
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        
        # Append the bounding box as a tuple: (xmin, ymin, xmax, ymax)
        bounding_boxes.append((xmin, ymin, xmax, ymax))
    
    return bounding_boxes


# Load dataset function
def load_dataset(images_dir, annotations_dir):
    """Loads images and bounding boxes, processes them into Numpy arrays."""
    images = []
    bounding_boxes_list = []
    multiple_bboxes_count = 0  # Counter for images with multiple bounding boxes
    
    for img_file in os.listdir(images_dir):
        if img_file.endswith(".jpg") or img_file.endswith(".png"):
            img_path = os.path.join(images_dir, img_file)
            xml_file = img_file.replace('.jpg', '.xml').replace('.png', '.xml')
            xml_path = os.path.join(annotations_dir, xml_file)

            if os.path.exists(xml_path):
                img = process_image(img_path)
                bounding_boxes = extract_bounding_boxes(xml_path)

                images.append(img)
                bounding_boxes_list.append(bounding_boxes)
                
                # Count if the image has more than one bounding box
                if len(bounding_boxes) > 1:
                    multiple_bboxes_count += 1

    return np.array(images), bounding_boxes_list, multiple_bboxes_count


# Example usage
if __name__ == "__main__":
    images, bounding_boxes_list, multiple_bboxes_count = load_dataset(images_dir, annotations_dir)
    
    print(f"Loaded {len(images)} images.")
    print(f"{multiple_bboxes_count} images have multiple bounding boxes.")
    
    # Example to print bounding box information
    for i, bboxes in enumerate(bounding_boxes_list[:5]):  # Display first 5 bounding boxes as a sample
        print(f"Image {i+1} has {len(bboxes)} bounding box(es): {bboxes}")
