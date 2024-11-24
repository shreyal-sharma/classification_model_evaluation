# Classification Model Evaluation for Diabetes Prediction
This project performs the evaluation of different machine learning models to predict diabetes using the Pima Indians Diabetes Database. The models include Random Forest, K-Nearest Neighbors (KNN), and a GRU-based neural network model. The dataset is split into training and test sets, and 10-fold cross-validation is applied to evaluate the models. The project also includes confusion matrix visualizations and comparison of the models based on metrics like accuracy, precision, recall, and F1-score.

## Dataset
The dataset used for this project is the Pima Indians Diabetes Database. It contains medical records of patients and is used to predict whether a patient has diabetes. The features of the dataset are:

- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age
The target variable, Outcome, represents whether the patient has diabetes (1) or not (0).

## Data Loading
Ensure the dataset is available at the correct path:
```
url = "dataset/pima-indians-diabetes-database.csv"
```
You can download the dataset from various sources such as Kaggle.

## Installation
### Prerequisites
Ensure that the following Python libraries are installed:

- pandas
- numpy
- scikit-learn
- tensorflow
- matplotlib
- seaborn

You can install the necessary dependencies by running:
```
pip install pandas numpy scikit-learn tensorflow matplotlib seaborn
```

## Running the Project
1. Clone the repository or download the files to your local machine.
2. Ensure the dataset is located at the correct path (dataset/pima-indians-diabetes-database.csv).
3. Run the script: python model_evaluation.py

# Conclusion
This project compares different machine learning models for diabetes prediction using 10-fold cross-validation. The models are evaluated based on important metrics and their performance is compared. The GRU model showed the best performance in terms of accuracy and other evaluation metrics.
