import numpy as np
import pickle
import os
import tensorflow as tf
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Define the paths to the model files
svm_model_path = os.path.join(os.path.dirname(__file__), 'model', 'svm_model.pkl')
rf_model_path = os.path.join(os.path.dirname(__file__), 'model', 'random_forest_model.pkl')
mlp_model_path = os.path.join(os.path.dirname(__file__), 'model', 'emotion_model.h5')
scaler_path = os.path.join(os.path.dirname(__file__), 'model', 'mlp_scaler.pkl')
le_path = os.path.join(os.path.dirname(__file__), 'model', 'label_encoder.pkl')

# Load the SVM model
with open(svm_model_path, 'rb') as f:
    svm_model = pickle.load(f)

# Load the Random Forest model
with open(rf_model_path, 'rb') as f:
    rf_model = pickle.load(f)

# Load the MLP model
mlp_model = tf.keras.models.load_model(mlp_model_path)

# Load the scaler for the MLP model
with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

# Load the label encoder
with open(le_path, 'rb') as f:
    le = pickle.load(f)

# Sample inputs (combinations of three primary colors)
sample_inputs = [
    np.array([126, 115, 85, 218, 217, 200, 53, 53, 39])  
]

# Normalize the inputs to [0, 1] range
sample_inputs_normalized = [sample_input / 255.0 for sample_input in sample_inputs]

# Predict emotions for each sample input using SVM, Random Forest, and MLP models
for idx, input_data in enumerate(sample_inputs_normalized):
    input_reshaped = input_data.reshape(1, -1)
    
    # Predict using SVM model
    svm_prediction = svm_model.predict_proba(input_reshaped)
    svm_top_indices = svm_prediction.argsort()[0][::-1][:2]
    svm_top_emotions = le.inverse_transform(svm_top_indices)
    svm_top_probabilities = svm_prediction[0][svm_top_indices]
    
    logging.debug(f"Sample {idx + 1} (SVM Model):")
    for emotion, probability in zip(svm_top_emotions, svm_top_probabilities):
        logging.debug(f"  {emotion}: {probability * 100:.2f}%")
    
    # Predict using Random Forest model
    rf_prediction = rf_model.predict_proba(input_reshaped)
    rf_top_indices = rf_prediction.argsort()[0][::-1][:2]
    rf_top_emotions = le.inverse_transform(rf_top_indices)
    rf_top_probabilities = rf_prediction[0][rf_top_indices]
    
    logging.debug(f"Sample {idx + 1} (Random Forest Model):")
    for emotion, probability in zip(rf_top_emotions, rf_top_probabilities):
        logging.debug(f"  {emotion}: {probability * 100:.2f}%")
    
    # Scale the input for the MLP model
    mlp_input_scaled = scaler.transform(input_reshaped)
    
    # Predict using MLP model
    mlp_prediction = mlp_model.predict(mlp_input_scaled)
    mlp_top_indices = mlp_prediction.argsort()[0][::-1][:2]
    mlp_top_emotions = le.inverse_transform(mlp_top_indices)
    mlp_top_probabilities = mlp_prediction[0][mlp_top_indices]
    
    logging.debug(f"Sample {idx + 1} (MLP Model):")
    for emotion, probability in zip(mlp_top_emotions, mlp_top_probabilities):
        logging.debug(f"  {emotion}: {probability * 100:.2f}%")
    
    logging.debug("\n")