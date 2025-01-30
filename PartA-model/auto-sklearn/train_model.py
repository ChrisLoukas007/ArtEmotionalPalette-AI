import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import json

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (precision_score, recall_score, confusion_matrix, 
                             accuracy_score, f1_score, roc_curve, auc, classification_report)
from sklearn.preprocessing import label_binarize

import autosklearn.classification

# Write a Custom Serialization Function which converts NumPy int/float to regular Python int/float
def custom_encoder(obj):
    """
    Custom JSON serializer.
    - Converts NumPy int/float to regular Python int/float
    - Converts unknown objects to string
    """
    import numpy as np
    if isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    # Convert any other object to string
    return str(obj)


# 1. Set random seed for reproducibility
np.random.seed(42)

# 2. Load and preprocess data
data = pd.read_csv('final_dataset.csv')  # Ensure final_dataset.csv is in the same directory or update the path

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
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 5. Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 6. Use auto-sklearn for model selection and hyperparameter tuning
automl = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=3600,   # Total time in seconds for optimization
    per_run_time_limit=300,         # Time limit for each model training
    tmp_folder="/app/tmp",          # Temporary directory for intermediate files
)

print("Fitting auto-sklearn classifier...")
automl.fit(X_train, y_train)
print("Done fitting the classifier.")

# 7. Create a results folder to save the best model and metrics
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

# 8. Save the best model (pickle)
model_path = os.path.join(results_dir, "best_model.pkl")
with open(model_path, "wb") as f:
    pickle.dump(automl, f)
print(f"Best model saved to {model_path}")

# 9. Extract and print best model name and hyperparameters
model_description = automl.show_models()
if isinstance(model_description, dict) and len(model_description) > 0:
    best_model_name = list(model_description.keys())[0]  # first model is usually the best
    best_model_config = model_description[best_model_name]
    print(f"\nBest Model: {best_model_name}")
    print(f"Model Configuration: {best_model_config}")

    # Save the best model description (JSON) to a text file
    description_path = os.path.join(results_dir, "best_model_description.txt")
    with open(description_path, "w") as f:
        f.write(json.dumps(model_description, indent=4, default=custom_encoder))
    print("Best model description (with hyperparameters) saved to", description_path)
else:
    print("No models were found by auto-sklearn.")
    best_model_name = "No model"
    best_model_config = {}

# 10. Evaluate the best model
y_pred = automl.predict(X_test)

test_accuracy = accuracy_score(y_test, y_pred)
test_f1 = f1_score(y_test, y_pred, average='weighted')
test_precision = precision_score(y_test, y_pred, average='weighted')
test_recall = recall_score(y_test, y_pred, average='weighted')

print(f"\nTest Accuracy: {test_accuracy:.4f}")
print(f"Test F1-score: {test_f1:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall: {test_recall:.4f}")

# 11. Classification report
print("\nClassification Report:")
classification_rep = classification_report(y_test, y_pred, target_names=le.classes_)
print(classification_rep)

# Save classification report to file
classification_report_path = os.path.join(results_dir, "classification_report.txt")
with open(classification_report_path, "w") as f:
    f.write(classification_rep)
print("Classification report saved to", classification_report_path)

# 12. Confusion Matrix
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

cm_plot_path = os.path.join(results_dir, "confusion_matrix_autosklearn.png")
plt.savefig(cm_plot_path, dpi=300)
print("Confusion matrix plot saved to", cm_plot_path)
plt.show()

# 13. ROC Curve
y_bin = label_binarize(y_test, classes=np.unique(y_test))
n_classes = y_bin.shape[1]
y_score = automl.predict_proba(X_test)

fpr = {}
tpr = {}
roc_auc = {}

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve/area
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

roc_plot_path = os.path.join(results_dir, "roc_curve_autosklearn.png")
plt.savefig(roc_plot_path, dpi=300)
print("ROC curve plot saved to", roc_plot_path)
plt.show()
