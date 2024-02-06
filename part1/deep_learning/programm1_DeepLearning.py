import csv
import cv2
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import os

def get_primary_color(image):
    # Convert the image to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Reshape the image to be a list of RGB pixels
    pixels = image.reshape(-1, 3)
    # Find the most common pixel value
    primary_color = np.bincount(pixels.astype(int).tostring()).argmax()
    # Convert the pixel value back to RGB
    primary_color = np.fromstring(primary_color, dtype=int)
    return primary_color

def load_images(image_paths):
    # Load images and convert them to the RGB color space
    images = []
    primary_colors = []
    for image_path in image_paths:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)
        primary_colors.append(get_primary_color(image))
    return np.array(images), np.array(primary_colors)

def build_model():
    # Build a simple CNN model
    model = Sequential()
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(256, 256, 3)))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(3, (3, 3), activation='softmax', padding='same'))
    return model

def train_model(model, images, primary_colors):
    # Split the images and primary colors into a training set and a validation set
    train_images, val_images, train_colors, val_colors = train_test_split(images, primary_colors, test_size=0.2, random_state=42)

    # Compile and train the model
    model.compile(optimizer=Adam(), loss='mse')
    model.fit(train_images, train_colors, epochs=10, validation_data=(val_images, val_colors))

if __name__ == "__main__":
    input_csv_path = 'part1/image_paths_Right.csv'

    if not os.path.exists(input_csv_path):
        print(f"Error: {input_csv_path} does not exist")
        exit(1)

    with open(input_csv_path, 'r') as input_csv:
        reader = csv.reader(input_csv)
        image_paths = [row[0] for row in reader]

    images, primary_colors = load_images(image_paths)
    model = build_model()
    train_model(model, images, primary_colors)