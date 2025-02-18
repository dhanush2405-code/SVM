import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os

# Set Streamlit page configuration
st.set_page_config(page_title="Water Quality Prediction", layout="wide")

st.title("💧 Water Quality Prediction using SVM")

# File path
file_path = 'water_quality_dataset.csv'

# Load or generate dataset
def load_data():
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        np.random.seed(42)
        num_samples = 500
        data = {
            'pH': np.random.uniform(6.0, 9.0, num_samples),
            'Dissolved Oxygen': np.random.uniform(3.0, 10.0, num_samples),
            'BOD': np.random.uniform(1.0, 8.0, num_samples),
            'Nitrates': np.random.uniform(0.1, 50.0, num_samples),
            'Turbidity': np.random.uniform(0.1, 10.0, num_samples),
        }
        data['Quality'] = [
            1 if (6.5 <= pH <= 8.5 and do >= 5 and bod <= 3 and nitrates < 40 and turb < 5) else
            (0 if (pH < 6.0 or pH > 9.0 or do < 3 or bod > 8 or nitrates >= 50 or turb >= 10) else 2)  # Moderate (2)
            for pH, do, bod, nitrates, turb in zip(data['pH'], data['Dissolved Oxygen'], data['BOD'], data['Nitrates'], data['Turbidity'])
        ]
        df = pd.DataFrame(data)
        df.to_csv(file_path, index=False)
        return df

# Load dataset
df = load_data()
st.write("### Dataset Preview")
st.dataframe(df.head())

# Data Preprocessing
df = df.dropna()
target_column = df.columns[-1]
X = df.drop(columns=[target_column])
y = df[target_column]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train SVM model
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_model.fit(X_train, y_train)
y_pred = svm_model.predict(X_test)

# Model Evaluation
accuracy = accuracy_score(y_test, y_pred)
st.write(f"### Model Accuracy: {accuracy:.2f}")
st.write("### Classification Report")
st.text(classification_report(y_test, y_pred))

# Confusion Matrix
st.write("### Confusion Matrix")
fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
st.pyplot(fig)

