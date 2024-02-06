import csv # for reading and writing csv files
import cv2 # for reading images
import numpy as np # for numerical operations
from sklearn.cluster import KMeans # for K-means algorithm
import time # for measuring the execution time
import psutil # for getting CPU and memory usage
import os # for file  path operations
from multiprocessing import Pool # for parallel processing

def get_primary_color(image_path): # Function to get the primary color of an image
    try: 
        image = cv2.imread(image_path) # Load the image
        if image is None: 
            print(f"Warning: Could not load image at {image_path}")
            return None
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convert the image from BGR to RGB
        image = image.reshape(-1, 3) # Reshape the image to be a list of RGB pixels . Each pixel is a 3-element array containing the RGB values
        
        np.random.seed(0) # This line makes sure that if we run the code multiple times, we get the same result each time.
        kmeans = KMeans(n_clusters=1, n_init=10) #I set n_clusters=1 because the K-means algorithm will find the average color of the image that is the primary color, 
        #n_init=10 means the algorithm will be run 10 times with different initial centroids, and the best one in terms of inertia will be chosen.
        kmeans.fit(image) # Fit the K-means model to the image
        
        # Gets the main color that kmeans found . It;s a group of 3 values that represent the RGB color
        primary_color = tuple(map(int, kmeans.cluster_centers_[0])) 
        # I wrote 0 because I want to get the first color that kmeans found. If you want to get more colors, you can change the value of n_clusters in the KMeans function.
        
        return primary_color
    except Exception as e:
        print(f"Error processing image at {image_path}: {e}") 
        return None

def process_image(image_path): 
    primary_color = get_primary_color(image_path)
    if primary_color is not None:
        # Write the RGB values in the form (r, g, b)
        return [image_path, f"({primary_color[0]}, {primary_color[1]}, {primary_color[2]})"]

if __name__ == "__main__":
    input_csv_path = 'image_paths_Right.csv'
    output_csv_path = 'part1\k_menas\image_paths_color_k-means.csv'

    if not os.path.exists(input_csv_path):
        print(f"Error: {input_csv_path} does not exist")
        exit(1)

    with open(input_csv_path, 'r') as input_csv:
        reader = csv.reader(input_csv)
        image_paths = [row[0] for row in reader]

    start_time = time.time()

    with Pool() as p:
        results = p.map(process_image, image_paths)

    with open(output_csv_path, 'w', newline='') as output_csv:
        writer = csv.writer(output_csv)
        writer.writerow(['image_path', 'color'])  # Write header
        writer.writerows(result for result in results if result is not None)

    end_time = time.time()

    print(f"Total execution time: {end_time - start_time} seconds")

    # CPU usage
    print(f"CPU usage: {psutil.cpu_percent()}%")

    # Memory usage
    memory_usage = psutil.virtual_memory()
    print(f"Memory usage: {memory_usage.percent}%")