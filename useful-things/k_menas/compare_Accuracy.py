import csv
import math

def calculate_distance(rgb1, rgb2):
    return math.sqrt(sum((e1-e2)**2 for e1, e2 in zip(rgb1, rgb2)))

# Read the output of programm1.py
with open('part1/k_menas/predicted_rgb_values.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader)  # Skip the header
    programm1_output = [(row[0], tuple(map(int, row[1].strip("()").split(", ")))) for row in reader]

# Read the actual primary colors
with open('part1/k_menas/real_values_rgb.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader)  # Skip the header
    real_values = [(row[0], tuple(map(int, row[1].strip("()").split(", ")))) for row in reader]

# Compare the primary colors and calculate the accuracy
correct = 0
threshold = 10  # You can adjust this value based on your requirements
for programm1_row, real_row in zip(programm1_output, real_values):
    distance = calculate_distance(programm1_row[1], real_row[1])
    if distance <= threshold:
        correct += 1

accuracy = correct / len(programm1_output)
print(f'Accuracy: {accuracy * 100}%')