#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import os
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Get the current directory
current_dir = os.path.dirname(__file__)

# Load the model and scaler
model_path = os.path.join(current_dir, "logistic_model.pkl")
scaler_path = os.path.join(current_dir, "scaler.pkl")

model = pickle.load(open(model_path, "rb"))
scaler = pickle.load(open(scaler_path, "rb"))

# Load the trained model and scaler
model = pickle.load(open("logistic_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Title of the app
# st.title("LinkedIn Usage Prediction App")
# st.write("Enter the demographic information below to predict LinkedIn usage and probability.")

# Collect user input
# income = st.slider("Household Income Level (1-9)", 1, 9, 5)
# education = st.slider("Education Level (1-8)", 1, 8, 4)
# parent = st.selectbox("Are you a parent?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
# married = st.selectbox("Marital Status", [0, 1], format_func=lambda x: "Not Married" if x == 0 else "Married")
# female = st.selectbox("Gender", [0, 1], format_func=lambda x: "Male" if x == 0 else "Female")
# age = st.number_input("Age (years)", min_value=1, max_value=98, value=30)

# Create input data as a DataFrame
# input_data = pd.DataFrame([[income, education, parent, married, female, age]],
                          columns=['income', 'educ2', 'parent', 'married', 'female', 'age'])

# Scale the input data
# input_data_scaled = pd.DataFrame(scaler.transform(input_data), columns=input_data.columns)

# Make predictions
# prediction = model.predict(input_data_scaled)
# probability = model.predict_proba(input_data_scaled)

# Display the result
# st.write(f"Prediction: {'LinkedIn User' if prediction[0] == 1 else 'Not a LinkedIn User'}")
# st.write(f"Probability of using LinkedIn: {probability[0][1]:.2f}")

In[ ]:




