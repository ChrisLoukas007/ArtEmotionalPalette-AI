# ArtEmotionalPalette-AI 🎨🧠

### Master Thesis – University of Patras, CEID

📌 **Author:** Christos Loukas Ntais  
📌 **Supervisor:** Prof. Ioannis Vasilopoulos

---

## Overview

**ArtEmotionalPalette-AI** is a **machine learning** project that aims to **predict the emotional content of paintings** based on their representative colors. This work is part of my **Master Thesis** at the **University of Patras, Computer Engineering & Informatics Department (CEID)**.

Leveraging **Kobayashi's (1991) findings** on **three-color combinations** and their associated emotions, the system **extracts key colors** from a painting and uses **academically validated** machine learning models to **predict the painting’s emotional label**.

---

## Project Goals & Objectives

1. **Extract Representative Colors**: Use **K-Means clustering** to identify the three most defining colors of a given painting.
2. **Predict Emotions**: Train **SVM, MLP, and Random Forest** classifiers to map color combinations to emotional labels with a **rigorous evaluation** process.
3. **Build a Web Application**: Provide an **interactive platform** for users to upload images and instantly **analyze** their emotional dimensions.

---

## Methodology

### 1. Dataset Development

- **Foundation**: Built on **Kobayashi’s research (1991)**, linking color combos to emotional words.
- **Composition**:
  - 161 words describing **positive emotions**
  - 130 distinct colors
  - 1,040 unique **three-color combinations** mapped to emotions
- **Data Augmentation**:
  - Expanded to ~17,200 samples using **SMOTETomek** for **class balance**

### 2. Representative Color Extraction

- **Explored Methods**: Color Histogram, Image Quantification, Deep Learning
- **Chosen Method**: **K-Means Clustering** (offered the most consistent accuracy in identifying key colors)

### 3. Machine Learning & Automated Search

#### 3.1 Manual Model Training

- Trained **multi-class** classification models:
  - **Support Vector Machine (SVM)**
  - **Multilayer Perceptron (MLP)**
  - **Random Forest Classifier**
- **Optimization**:
  - **Grid Search & Randomized Search** for hyperparameter tuning
  - **Stratified K-Fold Cross-Validation** to reduce overfitting and enhance reliability
- **Performance**:
  | Model | Test Accuracy |
  |-------------------|----------------------|
  | **MLP** | **95.44%** ✅ _(Best)_ |
  | **SVM** | **94.48%** |
  | **Random Forest** | **87.82%** |

#### 3.2 auto-sklearn Approach

- Used **auto-sklearn** (in the `auto-sklearn/` folder) to automatically search for the best performing model
- Resulted in an **SVC model** with ~**91% test accuracy**, which was still **lower** than the **MLP** approach but demonstrates an alternative automated pipeline

### 4. Web Application

A **user-friendly web app** enables:

1. **Image Upload**: Users upload a painting
2. **Color Extraction**: Automatically extracts its **three representative colors**
3. **Emotion Prediction**: Returns **emotional labels** using the trained ML models

---

## 🗂 Folder Structure & Contents

Below is a brief overview of the current repository structure:

| Folder / File                      | Description                                                           |
| ---------------------------------- | --------------------------------------------------------------------- |
| `Color-Extraction/`                | Scripts & notebooks for **color extraction** from images.             |
| `emotion-predictor/`               | The **web-based** Emotion Predictor application.                      |
| `images_jpg/`                      | Sample images for **testing** the application.                        |
| `PartA-model/`                     | Main ML approach notebooks & data (see details below).                |
| &emsp; ├─ `auto-sklearn/`          | Contains **auto-sklearn** experiment notebooks.                       |
| &emsp; ├─ `datasets/`              | Processed datasets used in the model **training**.                    |
| &emsp; ├─ `experiments/`           | Exploratory experiments and **research evaluations**.                 |
| &emsp; ├─ `Data_preparation.ipynb` | Notebook to **prepare** and **clean** data before training.           |
| &emsp; ├─ `final_dataset.csv`      | Final compiled dataset for **model input**.                           |
| &emsp; ├─ `MLP.ipynb`              | Jupyter notebook for training the **MLP** model.                      |
| &emsp; ├─ `RandomForest.ipynb`     | Jupyter notebook for training the **Random Forest** model.            |
| &emsp; └─ `SVM.ipynb`              | Jupyter notebook for training the **SVM** model.                      |
| `Abstract.md`                      | Contains the **thesis abstract** in Markdown format.                  |
| `README.md`                        | The file you are reading now.                                         |
| `.gitattributes`                   | Git LFS configuration or attributes for handling specific file types. |
| `.gitignore`                       | Specifies intentionally untracked files to ignore.                    |

---

## 🚀 Getting Started

### 1. Extract Colors

1. Go to the `Color-Extraction/` folder.
2. Run the provided **Jupyter notebooks** to perform **K-Means** clustering on your images.

### 2. Train Models

1. Navigate to the `PartA-model/` folder.
2. Choose any of the model notebooks:
   - **MLP.ipynb**
   - **SVM.ipynb**
   - **RandomForest.ipynb**
3. Or explore the **auto-sklearn** approach in the `auto-sklearn/` subfolder.
4. Run the **hyperparameter tuning** (Grid Search / Randomized Search) and **Stratified K-Fold Cross-Validation** cells to get **optimal** models.

### 3. Run the Web Application

1. Move to the `emotion-predictor/` directory.
2. Execute:
   ```bash
   docker-compose up
   ```

### Model Files (.pkl / .h5) Notice

For repository size constraints, the final trained model files (`.pkl` and `.h5`) are not included in this GitHub repo. If you’d like to:

- **Retrain the models yourself**, simply follow the steps in the `Model-Training/` notebooks.
- **Obtain the pre-trained models**, please email me at:  
  📧 **chris.dais1201@gmail.com**

I will be happy to send them to you or answer any additional questions regarding the research, implementation, or hyperparameter tuning.

### Further Documentation

All research details, model comparisons, hyperparameter choices, and the complete bibliography used in this thesis are available in an accompanying **130-page document** named  
**"ΣΥΝΑΙΣΘΗΜΑΤΙΚΗ ΑΝΑΛΥΣΗ ΠΙΝΑΚΩΝ ΜΕ ΧΡΗΣΗ ΜΗΧΑΝΙΚΗΣ ΜΑΘΗΣΗΣ"** (`.doc` / `.pdf`). This includes:

- **In-depth algorithm explanations**
- **Justification of parameter settings**
- **Comprehensive references** to academic papers and textbooks that informed this study

To access this document, you can:  
📩 **Email me at:** **chris.dais1201@gmail.com**
