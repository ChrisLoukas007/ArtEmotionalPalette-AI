import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os
import csv
from kneed import KneeLocator
from collections import Counter
from matplotlib.colors import rgb2hex


# Function to load and preprocess image
def load_and_preprocess_image(image_path):
    # Read the image
    image = cv2.imread(image_path)
    # Convert the image from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Resize the image to 600x400
    image = cv2.resize(image, (600, 400), interpolation = cv2.INTER_AREA)
    # Reshape the image to a 2D array
    image = image.reshape(-1, 3)
    return image

# Function to extract dominant colors from an image using the elbow method
def extract_colors_elbow(image):
    distortions = []
    K = range(1,10)
    # Loop over different values of k to find the optimal number of clusters
    for k in K:
        kmeanModel = KMeans(n_clusters=k, n_init=10)
        kmeanModel.fit(image)
        distortions.append(kmeanModel.inertia_)

    # Use the KneeLocator function to find the elbow point in the plot
    kn = KneeLocator(K, distortions, curve='convex', direction='decreasing')
    number_of_colors = kn.knee

    # Fit the KMeans model to the image with the optimal number of clusters
    clf = KMeans(n_clusters = number_of_colors, n_init=10)
    labels = clf.fit_predict(image)

    # Count the number of pixels in each cluster
    counts = Counter(labels)
    counts = dict(sorted(counts.items()))

    # Get the center color of each cluster
    center_colors = clf.cluster_centers_
    # Order the colors according to the sorted keys of the counts dictionary
    ordered_colors = [center_colors[i] for i in counts.keys()]
    # Get the RGB colors
    rgb_colors = [ordered_colors[i] for i in counts.keys()]

    # Convert RGB colors to hexadecimal
    hex_colors = [rgb2hex(color/255.0) for color in rgb_colors]  # Divide by 255 because rgb2hex expects RGB values to be in the range [0, 1]
    return hex_colors

# Get the list of image paths
image_paths = [os.path.join('images_jpg', f) for f in os.listdir('images_jpg') if f.endswith('.jpg')]

dominant_colors = {}

# Open the CSV file in write mode
with open('dominant_colors.csv', 'w', newline='') as csvfile:
    # Create a CSV writer
    writer = csv.writer(csvfile)

    # Loop over all images
    for image_path in image_paths:
        # Load and preprocess the image
        image = load_and_preprocess_image(image_path)
        # Extract the dominant colors from the image
        dominant_colors = extract_colors_elbow(image)
        # Write the image path and the dominant colors to the CSV file
        writer.writerow([image_path] + dominant_colors)