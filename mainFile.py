import os
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

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


# Label extraction function
def extract_labels(xml_path):
    """Extracts labels from the XML annotation file."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    labels = []
    for obj in root.findall('object'):
        label = obj.find('name').text
        labels.append(label)
    return labels


# Load dataset function
def load_dataset(images_dir, annotations_dir):
    """Loads images and labels, processes them into Numpy arrays."""
    images = []
    labels = []
    for img_file in os.listdir(images_dir):
        if img_file.endswith(".jpg") or img_file.endswith(".png"):
            img_path = os.path.join(images_dir, img_file)
            xml_file = img_file.replace('.jpg', '.xml').replace('.png', '.xml')
            xml_path = os.path.join(annotations_dir, xml_file)

            if os.path.exists(xml_path):
                img = process_image(img_path)
                label = extract_labels(xml_path)

                images.append(img)

                # If multiple labels exist, pick the first one for now (adjust based on your needs)
                if isinstance(label, list) and len(label) > 0:
                    labels.append(label[0])
                else:
                    labels.append(label)

    return np.array(images), np.array(labels)


# CNN Model definition
def create_cnn_model(input_shape, num_classes):
    """Defines a simple CNN model."""
    model = Sequential()

    # Convolutional layers
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Fully connected layers
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


# Example usage
if __name__ == "__main__":
    images, labels = load_dataset(images_dir, annotations_dir)
    print(f"Loaded {len(images)} images and {len(labels)} labels.")

    # Assuming 10 classes and 128x128x3 image size
    input_shape = (128, 128, 3)
    num_classes = 10

    cnn_model = create_cnn_model(input_shape, num_classes)
    cnn_model.summary()

    # You can now train the model with cnn_model.fit()
    # For example: cnn_model.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val))
