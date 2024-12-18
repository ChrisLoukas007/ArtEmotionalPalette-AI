# 1. Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (precision_score, recall_score, confusion_matrix, 
                             accuracy_score, f1_score, roc_curve, auc, classification_report)
from sklearn.preprocessing import label_binarize

import autosklearn.classification

# Set random seed for reproducibility
np.random.seed(42)

# 2. Load and preprocess data
data = pd.read_csv('final_dataset.csv')

# Separate features and labels
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Normalize input features to [0, 1] range
X = X / 255.0

# 3. Label encoding
le = LabelEncoder()
y = le.fit_transform(y)

# 4. Split the data into training and test sets with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# 5. Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 6. Use auto-sklearn for model selection and hyperparameter tuning
automl = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=3600,     # Total time in seconds for optimization
    per_run_time_limit=300,           # Time limit for each model training run in seconds
    tmp_folder="/app/tmp",
)

print("Fitting auto-sklearn classifier...")
automl.fit(X_train, y_train)
print("Done.")

# 7. Evaluate the best model found by auto-sklearn
y_pred = automl.predict(X_test)

test_accuracy = accuracy_score(y_test, y_pred)
test_f1 = f1_score(y_test, y_pred, average='weighted')
test_precision = precision_score(y_test, y_pred, average='weighted')
test_recall = recall_score(y_test, y_pred, average='weighted')

print(f"\nTest Accuracy: {test_accuracy:.4f}")
print(f"Test F1-score: {test_f1:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall: {test_recall:.4f}")

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# 8. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Confusion Matrix (Auto-sklearn)', pad=20)
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.tight_layout()
plt.show()

# 9. ROC Curve
# Binarize the labels for ROC curve
y_bin = label_binarize(y_test, classes=np.unique(y_test))
n_classes = y_bin.shape[1]

# Get the predicted probabilities from automl
y_score = automl.predict_proba(X_test)

fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_bin.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

plt.figure(figsize=(8, 6))
plt.plot(fpr["micro"], tpr["micro"],
         label='Micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve (Auto-sklearn)', fontsize=14)
plt.legend(loc="lower right", fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 10. Save the final pipeline (model) and preprocessing steps
model_dir = '../emotion-predictor/backend/app/model'
os.makedirs(model_dir, exist_ok=True)

# Save the automl model
model_path = os.path.join(model_dir, 'emotion_model_autosklearn.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(automl, f)
print(f"Auto-sklearn model saved successfully to {model_path}")

# Save the scaler
scaler_path = os.path.join(model_dir, 'mlp_scaler.pkl')
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
print(f"Scaler saved successfully to {scaler_path}")

# Save the label encoder
le_path = os.path.join(model_dir, 'label_encoder.pkl')
with open(le_path, 'wb') as f:
    pickle.dump(le, f)
print(f"Label Encoder saved successfully to {le_path}")

# Print the leaderboard of models considered by auto-sklearn
print("\nAuto-sklearn Leaderboard:")
print(automl.leaderboard())