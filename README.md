# **ArtEmotionalPalette-AI** 🎨🧠

### **Master Thesis - University of Patras, CEID**

📌 **Author:** [Your Name]  
📌 **Supervisor:** Prof. **Ioannis Vasilopoulos**

## **Overview**

**ArtEmotionalPalette-AI** is a research-driven machine learning system designed to **predict the emotional content of paintings based on their dominant colors**. This project is part of my **Master Thesis** at the **University of Patras, Computer Engineering & Informatics Department (CEID)**.

By combining **color theory, psychology, and AI**, this system analyzes the **relationship between colors and emotions** using machine learning techniques. Our approach is based on **Kobayashi's (1991) research**, which links **three-color combinations** to specific **emotional labels**.

## **Project Goals & Objectives**

This research aims to **bridge art and technology** by developing a system capable of:
✔️ **Extracting dominant colors** from paintings using advanced clustering techniques.  
✔️ **Predicting emotions** based on color combinations via machine learning models.  
✔️ **Providing an interactive web application** for users to analyze the emotional dimensions of paintings.

## **Methodology**

### 1️⃣ **Dataset Development**

- **Source:** Based on **Kobayashi’s research (1991)**.
- **Composition:**
  - 161 words describing **positive emotions**.
  - 130 distinct colors.
  - 1,040 unique **three-color combinations** mapped to emotions.
- **Augmentation:**
  - Expanded to **~17,200 samples** using **SMOTETomek oversampling** to balance classes.

### 2️⃣ **Dominant Color Extraction**

✔️ Evaluated **multiple methods**: Color Histogram, Image Quantification, Deep Learning.  
✔️ **Selected Method**: **K-Means Clustering** (best accuracy in identifying dominant colors).

### 3️⃣ **Machine Learning Models for Emotion Prediction**

We trained three **multi-class classification models**:  
🔹 **Support Vector Machines (SVM)**  
🔹 **Multilayer Perceptron (MLP)**  
🔹 **Random Forest Classifier**

✔️ **Optimization Techniques Applied:**

- **Grid Search & Randomized Search** for hyperparameter tuning.
- **Stratified K-Fold Cross-Validation** to prevent overfitting.

✔️ **Final Model Results:**  
| Model | Test Accuracy |
|--------|-------------|
| **MLP** | **95.44%** ✅ _(Best Performance)_ |
| **SVM** | **94.48%** |
| **Random Forest** | **87.82%** |

### 4️⃣ **Web Application Development**

We developed a **user-friendly web application** that allows users to:

- Upload images 🎨
- Extract dominant colors 🎭
- Predict emotions based on **trained AI models** 🤖

---

## **🗂 Folder Structure & Contents**

| Folder                                          | Description                                                         |
| ----------------------------------------------- | ------------------------------------------------------------------- |
| **`Color-Extraction`**                          | Code & data for extracting dominant colors from images.             |
| **`Model-Training`** _(Previously PartA-model)_ | Scripts & data for **training, testing, and optimizing** ML models. |
| **`emotion-predictor`**                         | The web-based Emotion Predictor application.                        |
| **`images_jpg`**                                | Sample images used for testing the application.                     |
| **`datasets`**                                  | Contains the processed datasets used in model training.             |
| **`experiments`**                               | Contains exploratory experiments and research evaluations.          |

---

## **🚀 How to Run the Project**

### **1️⃣ Extract Colors**

1. Navigate to the **`Color-Extraction/`** directory.
2. Run the **Jupyter notebooks** to extract dominant colors from images.

### **2️⃣ Train Models**

1. Navigate to the **`Model-Training/`** directory.
2. Run the **MLP, SVM, or Random Forest notebooks** to train the models.

### **3️⃣ Run the Web Application**

1. Navigate to the **`emotion-predictor/`** directory.
2. Run the command:
   ```bash
   docker-compose up
   ```
