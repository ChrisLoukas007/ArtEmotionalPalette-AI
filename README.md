ArtEmotionalPalette-AI

Overview

ArtEmotionalPalette-AI is an innovative project designed to analyze the emotional dimensions of artworks based on their dominant colors. This system bridges art and technology by employing machine learning algorithms to predict emotional responses evoked by paintings. Our research is grounded in Kobayashi's (1991) framework, which associates specific color combinations with words describing emotions.

Thesis Objectives

The primary objective of this project was to develop a system capable of:

- Extracting dominant colors from paintings using advanced techniques.
- Predicting emotions based on color combinations through machine learning classifiers.
- Providing a web-based application for users to analyze the emotional content of paintings.

Methodology

1. Dataset Development

- Source: Kobayashi's research (1991).
- Composition:
  - 161 words describing positive emotions.
  - 130 distinct colors.
  - 1,040 unique combinations of three colors paired with an emotion.
- Augmentation: The dataset was expanded to ~17,200 samples using permutation and SMOTE Tomek techniques to handle class imbalance.

2. Dominant Color Extraction

- Evaluated methods: Color Histogram, Image Quantification, and Deep Learning.
- Selected method: K-means clustering, chosen for its effectiveness in identifying dominant colors.

3. Classifier Development

- Evaluated algorithms: Support Vector Machines (SVM), Multilayer Perceptron (MLP), and Random Forest.
- Optimization techniques:
  - Hyperparameter tuning using Grid Search and Randomized Search.
  - Stratified K-Fold cross-validation to avoid overfitting.
- Results:
  - MLP: Best performance with 95.44% test accuracy.
  - SVM: Close second with 94.48% test accuracy.
  - Random Forest: Achieved 87.82% test accuracy.

4. Web Application Development

- Emotion Predictor: A user-friendly web application that allows users to upload images, extract their dominant colors, and predict corresponding emotions using the trained models.

Folder Descriptions

- ColorExtraction: Contains the code and data for extracting dominant colors from images.
- ModelTraining: Includes scripts and data for training, testing, and optimizing machine learning models.
- Application: The web-based Emotion Predictor application.
- Images: Sample images used for testing the application.
- Miscellaneous: Additional scripts and files for data processing and exploration.

Usage Instructions
Prerequisites

- Install Python and the required libraries listed in requirements.txt.

Running the Project

1. Extract Colors:

- Navigate to the ColorExtraction/code/ directory.
- Run the Jupyter notebooks to extract dominant colors.

2. Train Models:

- Navigate to the ModelTraining/code/ directory.
- Train models using the provided notebooks.

3. Run the Application:

- Navigate to the Application/ directory.
- Run docker-compose up to start the web application.

Results

- The MLP model achieved the highest accuracy (95.44%) in emotion prediction.
- The Emotion Predictor application provides a seamless way to analyze the emotional dimensions of paintings.

Contribution
We welcome contributions from the community. Feel free to fork the repository, raise issues, and submit pull requests.

## Large Files Not Included

The large model files have been removed from the Git history due to GitHub's file size limits. If you need these files, please contact me or download them from <external link>.
