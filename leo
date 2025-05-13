import pandas as pd

# Create a dictionary with data
data = {
    'Age': [45, 50, 60, 30, 70, 55, 40, 25, 65, 50],
    'Gender': ['Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female'],
    'Symptom1': ['Cough', 'Fever', 'Headache', 'Nausea', 'Chest Pain', 'Cough', 'Fever', 'Cough', 'Chest Pain', 'Headache'],
    'Symptom2': ['Fever', 'Cough', 'Fatigue', 'Vomiting', 'Fatigue', 'Chest Pain', 'Headache', 'Fatigue', 'Fever', 'Nausea'],
    'Disease': ['Disease1', 'Disease1', 'Disease2', 'Disease3', 'Disease2', 'Disease1', 'Disease3', 'Disease1', 'Disease2', 'Disease3']
}

# Create a DataFrame
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df.to_csv('patient_data.csv', index=False)

print("Dataset saved as 'patient_data.csv'.")
# Importing required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load the dataset
# Replace 'patient_data.csv' with the path to your CSV file containing the patient data
df = pd.read_csv('patient_data.csv')

# Step 2: Data Preprocessing
# Let's look at the first few rows to understand the data structure
print("First few rows of the dataset:")
print(df.head())

# Handle missing values if any (for simplicity, we'll drop rows with missing values)
df = df.dropna()  # Drops rows with missing values (NaN)

# Check columns and data types
print("\nData types of columns:")
print(df.dtypes)

# Encoding categorical data (if necessary)
label_encoder = LabelEncoder()

# List of categorical columns
categorical_columns = ['Gender', 'Symptom1', 'Symptom2']  # Add more categorical columns if necessary

# Check if categorical columns exist in the dataset before encoding
for col in categorical_columns:
    if col in df.columns:
        df[col] = label_encoder.fit_transform(df[col])
    else:
        print(f"Warning: Column '{col}' not found in the dataset. Skipping encoding for this column.")

# Step 3: Splitting the data into features (X) and target (y)
# Assuming 'Disease' is the target variable (output column)
if 'Disease' not in df.columns:
    raise ValueError("The 'Disease' column is not present in the dataset.")

X = df.drop(columns=['Disease'])  # Features (input data)
y = df['Disease']  # Target (output - disease)

# Step 4: Train-Test Split
# Splitting the data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Model Selection - Using Random Forest Classifier for disease prediction
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 6: Make predictions
y_pred = model.predict(X_test)

# Step 7: Evaluate the model
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'\nAccuracy: {accuracy:.2f}')

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

# Display confusion matrix using seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Step 8: Feature Importance (Optional: To see which features are most important)
importances = model.feature_importances_
features = X.columns
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance_df)

# Optional: Visualize feature importance using bar plot
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importance')
plt.show()
