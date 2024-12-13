import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# --- Load Data ---
@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)

# --- Clean Data ---
def clean_sm(x):
    return np.where(x == 1, 1, 0)

def preprocess_data(data):
    # Create a clean copy
    ss = data.copy()
    ss['sm_li'] = clean_sm(ss['web1h'])

    # Feature engineering
    ss['income'] = ss['income'].where(ss['income'] <= 9)
    ss['educ2'] = ss['educ2'].where(ss['educ2'] <= 8)
    ss['age'] = ss['age'].where(ss['age'] <= 98)
    ss['female'] = ss['gender'].apply(lambda x: 1 if x == 2 else 0)
    ss['parent'] = ss['par'].apply(lambda x: 1 if x == 1 else 0)
    ss['married'] = ss['marital'].apply(lambda x: 1 if x == 1 else 0)
    ss.dropna(inplace=True)

    return ss

# --- App Layout ---
st.title("Social Media Usage Prediction App")
st.markdown("Predict if someone uses LinkedIn based on demographic features.")

# File uploader
uploaded_file = st.file_uploader("Upload your dataset (.csv)", type=["csv"])
if uploaded_file:
    # Load and preprocess data
    data = load_data(uploaded_file)
    st.write("Data Loaded:")
    st.dataframe(data.head())
    ss = preprocess_data(data)
    
    st.write("Preprocessed Data:")
    st.dataframe(ss.head())
    
    # Split data
    features = ['income', 'educ2', 'parent', 'married', 'female', 'age']
    X = ss[features]
    y = ss['sm_li']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the model
    model = LogisticRegression(class_weight='balanced', random_state=42, max_iter=500)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Model Accuracy: {accuracy:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    st.write("Confusion Matrix:")
    st.dataframe(pd.DataFrame(cm, index=["Non-User", "User"], columns=["Predicted Non-User", "Predicted User"]))

    # Classification report
    st.write("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    # Predictions
    st.header("Make a Prediction")
    with st.form("prediction_form"):
        income = st.slider("Income (1-9)", 1, 9, value=5)
        educ2 = st.slider("Education Level (1-8)", 1, 8, value=4)
        parent = st.selectbox("Are you a parent?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        married = st.selectbox("Are you married?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        female = st.selectbox("Gender", [0, 1], format_func=lambda x: "Female" if x == 1 else "Male")
        age = st.slider("Age", 18, 98, value=30)
        submitted = st.form_submit_button("Predict")

        if submitted:
            new_data = pd.DataFrame([[income, educ2, parent, married, female, age]], columns=features)
            new_data_scaled = scaler.transform(new_data)
            prediction = model.predict(new_data_scaled)[0]
            probability = model.predict_proba(new_data_scaled)[0]
            st.write(f"Prediction: {'LinkedIn User' if prediction == 1 else 'Non-User'}")
            st.write(f"Probability of LinkedIn User: {probability[1]:.4f}")
