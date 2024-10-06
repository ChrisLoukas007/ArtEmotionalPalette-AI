import tensorflow as tf
import numpy as np
import pickle
import os

# Load the model
model_path = './model/emotion_model.h5'
model = tf.keras.models.load_model(model_path)

# Load the scaler
with open('./model/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load the label encoder
with open('./model/label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

# Sample inputs (RGB values)
sample_inputs = [
    np.array([255, 0, 0, 0, 255, 0, 0, 0, 255]),  # Red, Green, Blue
    np.array([128, 128, 128, 64, 64, 64, 192, 192, 192]),  # Grayscale variations
    np.array([0, 0, 0, 255, 255, 255, 128, 128, 128]),  # Black, White, Gray
]

# Predict emotions for each sample input
for idx, input_data in enumerate(sample_inputs):
    input_scaled = scaler.transform(input_data.reshape(1, -1))
    prediction = model.predict(input_scaled)
    top_indices = prediction.argsort()[0][::-1][:2]
    top_emotions = le.inverse_transform(top_indices)
    top_probabilities = prediction[0][top_indices]
    print(f"Sample {idx + 1}:")
    for emotion, probability in zip(top_emotions, top_probabilities):
        print(f"  {emotion}: {probability * 100:.2f}%")
