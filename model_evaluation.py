from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.utils import to_categorical
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
url = "dataset/pima-indians-diabetes-database.csv"  # Ensure correct path
data = pd.read_csv(url)

# Display dataset information
print(data.head())
print(f'Total number of records: {len(data)}')
print("Columns are: ", data.columns)
data.info()

# Check for missing values
if data.isnull().values.any():
    print("Dataset contains missing values.")
else:
    print("No missing values in the dataset.")

# Split data into features (X) and target (Y)
X = data[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
Y = data['Outcome']

# Split into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

# Function for KFold cross-validation for each model
def cross_val_metrics(model, X, Y):
    # Initialize KFold
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)

    # Accumulators for predictions and true labels to calculate confusion matrix after all folds
    all_true_labels = []
    all_pred_labels = []
    
    # Perform KFold cross-validation
    accuracy, precision, recall, f1 = [], [], [], []
    for i, (train_index, test_index) in enumerate(kfold.split(X), start=1):
        X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
        Y_train_fold, Y_test_fold = Y.iloc[train_index], Y.iloc[test_index]
        
        # Train the model
        model.fit(X_train_fold, Y_train_fold)
        
        # Predict and calculate metrics
        Y_pred_fold = model.predict(X_test_fold)
        
        # Append true labels and predictions to accumulators
        all_true_labels.extend(Y_test_fold)
        all_pred_labels.extend(Y_pred_fold)
        
        # Calculate individual fold metrics
        accuracy.append(accuracy_score(Y_test_fold, Y_pred_fold))
        precision.append(precision_score(Y_test_fold, Y_pred_fold))
        recall.append(recall_score(Y_test_fold, Y_pred_fold))
        f1.append(f1_score(Y_test_fold, Y_pred_fold))

    # Calculate confusion matrix once after all folds
    conf_matrix = confusion_matrix(all_true_labels, all_pred_labels)

    # Visualize confusion matrix once after all folds
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix for {model.__class__.__name__}')
    plt.show()

    # Return average scores
    return np.mean(accuracy), np.mean(precision), np.mean(recall), np.mean(f1)

# ---------------------------
# Random Forest Model
# ---------------------------
print("\nPerforming 10-Fold Cross Validation for Random Forest...")
rf = RandomForestClassifier(n_estimators=200, random_state=42)
accuracy_rf, precision_rf, recall_rf, f1_rf = cross_val_metrics(rf, X_train, Y_train)

# ---------------------------
# K-Nearest Neighbors Model
# ---------------------------
print("\nPerforming 10-Fold Cross Validation for KNN...")
knn = KNeighborsClassifier(n_neighbors=5)
accuracy_knn, precision_knn, recall_knn, f1_knn = cross_val_metrics(knn, X_train, Y_train)

# ---------------------------
# GRU Model
# ---------------------------
print("\nPerforming 10-Fold Cross Validation for GRU Model...")
# Convert labels to categorical (one-hot encoding) for GRU
Y_categorical = to_categorical(Y_train, num_classes=2)

# Reshape for GRU input format
X_train_reshaped = X_train.values.reshape(X_train.shape[0], 1, X_train.shape[1])  # Reshaping for GRU input format

# Initialize KFold for GRU model
kf = KFold(n_splits=10, shuffle=True, random_state=42)
gru_accuracies, gru_precisions, gru_recalls, gru_f1s = [], [], [], []
all_true_labels_gru = []
all_pred_labels_gru = []

for i, (train_index, test_index) in enumerate(kf.split(X_train_reshaped), start=1):
    X_train_gru, X_test_gru = X_train_reshaped[train_index], X_train_reshaped[test_index]
    Y_train_gru, Y_test_gru = Y_categorical[train_index], Y_categorical[test_index]

    # Build and compile GRU model
    gru_model = Sequential()
    gru_model.add(GRU(16, input_shape=(X_train_gru.shape[1], X_train_gru.shape[2]), activation='relu'))
    gru_model.add(Dense(2, activation='softmax'))  # Output layer for binary classification
    gru_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the GRU model on the training fold
    gru_model.fit(X_train_gru, Y_train_gru, epochs=10, batch_size=32, verbose=0)

    # Predict and calculate metrics
    Y_pred_gru = gru_model.predict(X_test_gru)
    Y_pred_classes_gru = np.argmax(Y_pred_gru, axis=1)
    Y_test_classes_gru = np.argmax(Y_test_gru, axis=1)

    # Append true labels and predictions for GRU
    all_true_labels_gru.extend(Y_test_classes_gru)
    all_pred_labels_gru.extend(Y_pred_classes_gru)

    # Store metrics for GRU
    gru_accuracies.append(accuracy_score(Y_test_classes_gru, Y_pred_classes_gru))
    gru_precisions.append(precision_score(Y_test_classes_gru, Y_pred_classes_gru))
    gru_recalls.append(recall_score(Y_test_classes_gru, Y_pred_classes_gru))
    gru_f1s.append(f1_score(Y_test_classes_gru, Y_pred_classes_gru))

# Calculate confusion matrix for GRU
conf_matrix_gru = confusion_matrix(all_true_labels_gru, all_pred_labels_gru)

# Visualize confusion matrix for GRU
sns.heatmap(conf_matrix_gru, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(f'Confusion Matrix for GRU Model')
plt.show()

# Average the metrics for GRU
accuracy_gru = np.mean(gru_accuracies)
precision_gru = np.mean(gru_precisions)
recall_gru = np.mean(gru_recalls)
f1_gru = np.mean(gru_f1s)

# ---------------------------
# Model Comparison (Cross-Validation)
# ---------------------------
print("\nModel Comparison (10-Fold Cross Validation):")
print(f"Random Forest - Accuracy: {accuracy_rf:.2f}, Precision: {precision_rf:.2f}, Recall: {recall_rf:.2f}, F1-Score: {f1_rf:.2f}")
print(f"KNN           - Accuracy: {accuracy_knn:.2f}, Precision: {precision_knn:.2f}, Recall: {recall_knn:.2f}, F1-Score: {f1_knn:.2f}")
print(f"GRU           - Accuracy: {accuracy_gru:.2f}, Precision: {precision_gru:.2f}, Recall: {recall_gru:.2f}, F1-Score: {f1_gru:.2f}")
model_comparison = pd.DataFrame({
    'Model': ['Random Forest', 'KNN', 'GRU'],
    'Accuracy': [accuracy_rf, accuracy_knn, accuracy_gru],
    'Precision': [precision_rf, precision_knn, precision_gru],
    'Recall': [recall_rf, recall_knn, recall_gru],
    'F1-Score': [f1_rf, f1_knn, f1_gru]
})

# Display the model comparison in a tabular format
print("\nTabular format Model Comparison (10-Fold Cross Validation):")
print(model_comparison.to_string(index=False))
