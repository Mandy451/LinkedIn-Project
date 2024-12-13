import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import streamlit as st

# Streamlit App Title
st.title("LinkedIn Usage Prediction")

# Input GitHub Raw URL
default_url = "https://raw.githubusercontent.com/Mandy451/LinkedIn-Project/main/social_media_usage.csv"
github_url = st.text_input("GitHub Raw URL:", value=default_url)

if github_url:
    try:
        # Read dataset from GitHub
        s = pd.read_csv(github_url)
        st.write("### Dataset Overview")
        st.write(s.head())

        # Clean data function
        def clean_sm(x):
            return np.where(x == 1, 1, 0)

        # Data cleaning and preparation
        ss = s.copy()
        ss['sm_li'] = clean_sm(ss['web1h'])

        features = ['income', 'educ2', 'par', 'marital', 'gender', 'age']
        ss = ss[features + ['sm_li']]

        ss['income'] = ss['income'].where(ss['income'] <= 9)
        ss['educ2'] = ss['educ2'].where(ss['educ2'] <= 8)
        ss['age'] = ss['age'].where(ss['age'] <= 98)
        ss['female'] = ss['gender'].apply(lambda x: 1 if x == 2 else 0)
        ss['parent'] = ss['par'].apply(lambda x: 1 if x == 1 else 0)
        ss['married'] = ss['marital'].apply(lambda x: 1 if x == 1 else 0)

        ss.dropna(inplace=True)

        # Exploratory Data Analysis
        st.write("### Exploratory Data Analysis")
        for feature in features:
            st.write(f"#### {feature} vs LinkedIn Usage")
            fig, ax = plt.subplots()
            sns.boxplot(x='sm_li', y=feature, data=ss, ax=ax)
            plt.close(fig)  # Close figure after rendering
            st.pyplot(fig)

        # Feature and target variable
        y = ss['sm_li']
        X = ss.drop(columns=['sm_li'])

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale data
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

        # Train logistic regression model
        model = LogisticRegression(class_weight='balanced', random_state=42, max_iter=500)
        model.fit(X_train_scaled, y_train)

        # Evaluate model
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"### Model Accuracy: {accuracy:.4f}")

        cm = confusion_matrix(y_test, y_pred)
        cm_df = pd.DataFrame(cm, 
                             index=['Actual: Non-LinkedIn User', 'Actual: LinkedIn User'], 
                             columns=['Predicted: Non-LinkedIn User', 'Predicted: LinkedIn User'])
        st.write("### Confusion Matrix")
        st.write(cm_df)

        st.write("### Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        st.write(pd.DataFrame(report).transpose())

        # Prediction Section
        st.write("### Make Predictions")
        with st.form("prediction_form"):
            st.write("Input the following details:")
            income = st.number_input("Income (1-9):", min_value=1, max_value=9, step=1)
            educ2 = st.number_input("Education Level (1-8):", min_value=1, max_value=8, step=1)
            parent = st.radio("Are you a parent?", ("Yes", "No"))
            marital = st.radio("Marital Status:", ("Married", "Not Married"))
            female = st.radio("Gender:", ("Female", "Male"))
            age = st.slider("Age (0-98):", min_value=0, max_value=98, step=1)

            submitted = st.form_submit_button("Predict")

            if submitted:
                # Prepare input data
                input_data = pd.DataFrame([[income, educ2, 1 if parent == "Yes" else 0, 
                                             1 if marital == "Married" else 0, 
                                             1 if female == "Female" else 0, age]],
                                           columns=X_train.columns)
                input_data_scaled = pd.DataFrame(scaler.transform(input_data), columns=X_train.columns)

                # Make prediction
                prediction = model.predict(input_data_scaled)
                probability = model.predict_proba(input_data_scaled)

                st.write(f"### Prediction: {'LinkedIn User' if prediction[0] == 1 else 'Non-LinkedIn User'}")
                st.write(f"### Probability: {probability[0][1]:.2f} for LinkedIn User, {probability[0][0]:.2f} for Non-LinkedIn User")

    except Exception as e:
        st.error(f"An error occurred: {e}")
