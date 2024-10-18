import os
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np
from sklearn.preprocessing import LabelEncoder
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

# Image processing function
def process_image(image_path):
    """Resizes and processes the image, converting RGBA to RGB if necessary."""
    image = Image.open(image_path)

    # Convert RGBA to RGB if the image has 4 channels
    if image.mode == 'RGBA':
        image = image.convert('RGB')

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

# CNN Model definition with Dropout layers
def create_cnn_model(input_shape, num_classes):
    """Defines a CNN model with reduced dropout layers."""
    model = Sequential()

    # Convolutional layers with reduced dropout
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))  # Reduced dropout

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))  # Reduced dropout

    # Fully connected layers
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.4))  # Reduced dropout before final layer
    model.add(Dense(num_classes, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Main execution
if __name__ == "__main__":
    # Load the dataset
    images, labels = load_dataset(images_dir, annotations_dir)
    print(f"Loaded {len(images)} images and {len(labels)} labels.")

    # Convert class labels from strings to integers
    label_encoder = LabelEncoder()
    integer_labels = label_encoder.fit_transform(labels)

    # One-hot encode the labels for classification
    one_hot_labels = to_categorical(integer_labels, num_classes=10)

    # Split the dataset into training and validation sets
    x_train, x_val, y_train, y_val = train_test_split(images, one_hot_labels, test_size=0.2, random_state=42)

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

    # Fit the generator to the training data
    datagen.fit(x_train)

    # Assuming 10 classes and 128x128x3 image size
    input_shape = (128, 128, 3)
    num_classes = 10

    # Create the CNN model
    cnn_model = create_cnn_model(input_shape, num_classes)

    # Print model summary
    cnn_model.summary()

    # Add a learning rate reduction callback
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=0.00001)

    # Train the CNN model using augmented data for 50 epochs
    cnn_model.fit(datagen.flow(x_train, y_train, batch_size=32),
                  validation_data=(x_val, y_val),
                  epochs=50,  # Increased epochs
                  callbacks=[reduce_lr])

    # Evaluate the model
    val_loss, val_accuracy = cnn_model.evaluate(x_val, y_val)
    print(f"Validation Loss: {val_loss}")
    print(f"Validation Accuracy: {val_accuracy}")

    # Save the model in Keras format
    cnn_model.save('cnn_model.keras')
