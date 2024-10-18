import os
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Directories
dataset_dir = 'D:\\Data\\VirajaProject\\pythonProject1'
images_dir = os.path.join(dataset_dir, 'images')
annotations_dir = os.path.join(dataset_dir, 'annotations')

# Image size for input to the model
IMG_SIZE = (128, 128)

# Load and process images with normalized bounding boxes
def process_image(image_path):
    """Resizes and processes the image, converting RGBA to RGB if necessary."""
    image = Image.open(image_path)

    if image.mode == 'RGBA':
        image = image.convert('RGB')

    image = image.resize(IMG_SIZE)
    return np.array(image)

def extract_bounding_boxes(xml_path):
    """Extracts bounding boxes from the XML annotation file."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    bounding_boxes = []
    for obj in root.findall('object'):
        bbox = obj.find('bndbox')

        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)

        bounding_boxes.append((xmin, ymin, xmax, ymax))
    return bounding_boxes

def normalize_bounding_boxes(image, bounding_boxes):
    """Normalizes bounding boxes relative to the image dimensions."""
    height, width, _ = image.shape
    normalized_boxes = []
    for (xmin, ymin, xmax, ymax) in bounding_boxes:
        xmin_norm = xmin / width
        ymin_norm = ymin / height
        xmax_norm = xmax / width
        ymax_norm = ymax / height
        normalized_boxes.append((xmin_norm, ymin_norm, xmax_norm, ymax_norm))
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
                img = Image.open(img_path)
                original_width, original_height = img.size
                img = process_image(img_path)

                bboxes = extract_bounding_boxes(xml_path)
                normalized_bboxes = [(xmin / original_width, ymin / original_height, xmax / original_width, ymax / original_height) for (xmin, ymin, xmax, ymax) in bboxes]

                images.append(img)
                all_bounding_boxes.append(normalized_bboxes[0])

    return np.array(images), np.array(all_bounding_boxes)

# Fine-Tuned CNN Model for Bounding Box Regression using MobileNet
def create_finetuned_mobilenet_model(input_shape):
    """Defines a fine-tuned CNN model with MobileNet as a feature extractor for bounding box prediction."""
    base_model = MobileNet(weights='imagenet', include_top=False, input_shape=input_shape)

    # Unfreeze the last few layers of MobileNet for fine-tuning
    for layer in base_model.layers[-20:]:  # Unfreeze the last 20 layers
        layer.trainable = True

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)  # Increased units for higher precision
    x = Dropout(0.5)(x)
    outputs = Dense(4, activation='linear')(x)

    model = Model(inputs=base_model.input, outputs=outputs)

    # Compile the model with mean squared error loss for bounding box regression
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

    return model

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=10,  # Reduced augmentation ranges for higher precision
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.05,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Visualization of bounding boxes
def visualize_bounding_boxes(image, true_bbox, pred_bbox):
    """Visualizes ground-truth and predicted bounding boxes on the image."""
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    # Draw ground-truth box in green
    xmin, ymin, xmax, ymax = true_bbox
    rect = patches.Rectangle((xmin * IMG_SIZE[0], ymin * IMG_SIZE[1]),
                             (xmax - xmin) * IMG_SIZE[0], (ymax - ymin) * IMG_SIZE[1],
                             linewidth=2, edgecolor='g', facecolor='none')
    ax.add_patch(rect)

    # Draw predicted box in red
    xmin, ymin, xmax, ymax = pred_bbox
    rect = patches.Rectangle((xmin * IMG_SIZE[0], ymin * IMG_SIZE[1]),
                             (xmax - xmin) * IMG_SIZE[0], (ymax - ymin) * IMG_SIZE[1],
                             linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

    plt.show()

# Main execution
if __name__ == "__main__":
    images, bounding_boxes = load_dataset_with_bounding_boxes(images_dir, annotations_dir)

    x_train, x_val, y_train, y_val = train_test_split(images, bounding_boxes, test_size=0.2, random_state=42)

    datagen.fit(x_train)

    input_shape = (128, 128, 3)
    cnn_model = create_finetuned_mobilenet_model(input_shape)

    cnn_model.summary()

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=0.00001)

    cnn_model.fit(datagen.flow(x_train, y_train, batch_size=32),
                  validation_data=(x_val, y_val),
                  epochs=50,  # Increased number of epochs for more training
                  callbacks=[reduce_lr])

    val_loss, val_mae = cnn_model.evaluate(x_val, y_val)
    print(f"Validation Loss: {val_loss}")
    print(f"Validation MAE: {val_mae}")

    index = 0
    true_bbox = y_val[index]
    pred_bbox = cnn_model.predict(x_val[index].reshape(1, 128, 128, 3))[0]

    visualize_bounding_boxes(x_val[index], true_bbox, pred_bbox)

    cnn_model.save('finetuned_mobilenet_bounding_box_regression_model.keras')
