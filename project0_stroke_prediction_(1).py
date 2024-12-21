import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Title of the Streamlit App
st.title('Stroke Prediction Model')
st.write("This app predicts whether a person is likely to have a stroke based on various health metrics.")

# Load the dataset
st.subheader("Dataset Preview")
data = pd.read_csv('https://raw.githubusercontent.com/AfEtaiwi02/FirstProject/main/healthcare-dataset-stroke-data.csv')  # Use the correct file path or URL
st.dataframe(data.head())

# Handling null values and preprocessing
data['bmi'] = data['bmi'].fillna(data['bmi'].mean())
data = data.dropna(subset=['bmi'])

# Label encoding for categorical columns
label_columns = ['gender', 'ever_married', 'smoking_status']
label_encoder = LabelEncoder()

for col in label_columns:
    data[col] = label_encoder.fit_transform(data[col])

# Feature scaling
scaler = StandardScaler()
data[['age', 'avg_glucose_level', 'bmi']] = scaler.fit_transform(data[['age', 'avg_glucose_level', 'bmi']])

# Feature engineering (one-hot encoding)
data = pd.get_dummies(data, drop_first=True)

# Preparing X and y
X = data.drop(['stroke', 'id'], axis=1)
y = data['stroke']

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Choose model via streamlit sidebar
model_choice = st.sidebar.selectbox("Choose a Model", ["Logistic Regression", "Random Forest"])

if model_choice == "Logistic Regression":
    model = LogisticRegression(class_weight='balanced')
elif model_choice == "Random Forest":
    model = RandomForestClassifier(class_weight='balanced', n_estimators=200, max_depth=10, random_state=42)

# Training the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Display results
accuracy = accuracy_score(y_test, y_pred)
st.subheader(f'Accuracy: {accuracy * 100:.2f}%')

# Classification report
st.subheader("Classification Report:")
st.text(classification_report(y_test, y_pred))

# Confusion Matrix
st.subheader("Confusion Matrix:")
conf_matrix = confusion_matrix(y_test, y_pred)
st.write(conf_matrix)

# ROC Curve
y_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

st.subheader("ROC Curve")
fig, ax = plt.subplots()
ax.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.2f})')
ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Receiver Operating Characteristic Curve')
ax.legend(loc='lower right')
st.pyplot(fig)

# Cross-validation
from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
st.subheader(f"Cross-validation Scores: {cv_scores}")
st.subheader(f"Mean accuracy: {np.mean(cv_scores):.2f}")
