import os
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau

# Directories
dataset_dir = 'D:\\Data\\VirajaProject\\pythonProject1'  # Base directory for the dataset
images_dir = os.path.join(dataset_dir, 'images')
annotations_dir = os.path.join(dataset_dir, 'annotations')

# Image size
IMG_SIZE = (128, 128)

# Load images and bounding boxes from XML files
def process_image(image_path):
    """Resizes and processes the image, converting RGBA to RGB if necessary."""
    image = Image.open(image_path)

    # Convert RGBA to RGB if the image has 4 channels
    if image.mode == 'RGBA':
        image = image.convert('RGB')

    image = image.resize(IMG_SIZE)
    return np.array(image)

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

        bounding_boxes.append((label, (xmin, ymin, xmax, ymax)))

    return bounding_boxes

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
                img = process_image(img_path)
                bboxes = extract_bounding_boxes(xml_path)
                normalized_bboxes = normalize_bounding_boxes(img, bboxes)

                images.append(img)
                all_bounding_boxes.append(normalized_bboxes)

    return np.array(images), all_bounding_boxes

# Handle dataset issues (removing images with multiple classes, etc.)
def filter_single_object_images(images, bounding_boxes):
    """Filter images that contain multiple classes."""
    filtered_images = []
    filtered_bboxes = []

    for img, bboxes in zip(images, bounding_boxes):
        # Check if all objects in this image belong to the same class
        labels = [bbox[0] for bbox in bboxes]
        if len(set(labels)) == 1:  # Keep only images with a single class
            filtered_images.append(img)
            filtered_bboxes.append(bboxes)

    return np.array(filtered_images), filtered_bboxes

# Convert bounding boxes and labels to fixed-size output format

def convert_to_fixed_output(bounding_boxes, num_classes):
    """Converts the bounding boxes into a fixed output format for the model."""
    fixed_labels = []
    fixed_bboxes = []

    for bboxes in bounding_boxes:
        # Use the first bounding box for simplicity (handle only one object per image for now)
        if bboxes:
            label, (xmin, ymin, xmax, ymax) = bboxes[0]  # Select the first bounding box
            label_index = class_to_index(label, num_classes)  # Convert label to index
            fixed_labels.append(label_index)
            fixed_bboxes.append([xmin, ymin, xmax, ymax])
        else:
            # If there are no bounding boxes, append zeros
            fixed_labels.append(0)
            fixed_bboxes.append([0, 0, 0, 0])

    return np.array(fixed_labels), np.array(fixed_bboxes)

# One-hot encode class labels
def one_hot_encode_labels(labels, num_classes):
    """Converts class labels to one-hot encoded vectors."""
    return to_categorical(labels, num_classes=num_classes)


def class_to_index(label, num_classes):
    """Converts a class label to an integer index."""
    classes = ['speedlimit', 'stop', 'trafficlight', 'crosswalk']  # Add your actual classes here
    return classes.index(label) if label in classes else 0  # Default to class '0' if label not found

# CNN Model for Bounding Box Prediction
def create_cnn_model(input_shape, num_classes):
    """Defines a CNN model for bounding box prediction."""
    model = Sequential()

    # Convolutional layers
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Fully connected layers for bounding box prediction and class prediction
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes + 4, activation='linear'))  # +4 for bounding box coordinates

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    return model

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Main execution
if __name__ == "__main__":
    # Load the dataset with bounding boxes
    images, bounding_boxes = load_dataset_with_bounding_boxes(images_dir, annotations_dir)

    # Filter out images with multiple objects (simplify the task for now)
    images, bounding_boxes = filter_single_object_images(images, bounding_boxes)

    print(f"Filtered {len(images)} images with single-class objects.")

    # Convert bounding boxes and class labels to fixed output format
    y_labels, y_bboxes = convert_to_fixed_output(bounding_boxes, num_classes=10)

    # One-hot encode the labels (for 10 classes)
    y_labels_one_hot = one_hot_encode_labels(y_labels, num_classes=10)

    # Combine one-hot encoded class labels and bounding boxes into a single output
    y_train_combined = np.hstack([y_labels_one_hot, y_bboxes])

    # Split the dataset into training and validation sets
    x_train, x_val, y_train, y_val = train_test_split(images, y_train_combined, test_size=0.2, random_state=42)

    # Fit the generator to the training data
    datagen.fit(x_train)

    # Assuming 10 classes and bounding boxes (x_min, y_min, x_max, y_max)
    input_shape = (128, 128, 3)
    num_classes = 10  # Adjust based on the number of classes in your dataset

    # Create the CNN model
    cnn_model = create_cnn_model(input_shape, num_classes)

    # Print model summary
    cnn_model.summary()

    # Add learning rate reduction callback
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=0.00001)

    # Train the CNN model
    cnn_model.fit(datagen.flow(x_train, y_train, batch_size=32),
                  validation_data=(x_val, y_val),
                  epochs=50,
                  callbacks=[reduce_lr])

    # Evaluate the model
    val_loss, val_accuracy = cnn_model.evaluate(x_val, y_val)
    print(f"Validation Loss: {val_loss}")
    print(f"Validation Accuracy: {val_accuracy}")

    # Save the model in Keras format
    cnn_model.save('bounding_box_cnn_model.keras')
