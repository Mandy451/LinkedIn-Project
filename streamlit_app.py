import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

@st.cache_data
def load_data_from_github():
    url = "https://raw.githubusercontent.com/Mandy451/LinkedIn-Project/refs/heads/main/social_media_usage.csv"
    return pd.read_csv(url)

# --- Helper Functions ---
def clean_sm(x):
    """Convert non-1 values to 0."""
    return np.where(x == 1, 1, 0)

def preprocess_data(data):
    """Preprocess the dataset for training."""
    ss = data.copy()
    ss['sm_li'] = clean_sm(ss['web1h'])
    
    # Feature selection and cleaning
    features = ['income', 'educ2', 'par', 'marital', 'gender', 'age']
    ss['income'] = ss['income'].where(ss['income'] <= 9)
    ss['educ2'] = ss['educ2'].where(ss['educ2'] <= 8)
    ss['age'] = ss['age'].where(ss['age'] <= 98)
    ss['female'] = ss['gender'].apply(lambda x: 1 if x == 2 else 0)
    ss['parent'] = ss['par'].apply(lambda x: 1 if x == 1 else 0)
    ss['married'] = ss['marital'].apply(lambda x: 1 if x == 1 else 0)
    ss = ss.dropna()
    return ss, features

# --- App Layout ---
st.title("LinkedIn Usage Prediction App")
st.markdown("This app predicts whether someone uses LinkedIn based on demographic data.")

# --- Step 1: Upload Data ---
uploaded_file = st.file_uploader("Upload your dataset (.csv)", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("### Original Dataset")
    st.dataframe(data.head())

    # Preprocess data
    ss, features = preprocess_data(data)
    st.write("### Preprocessed Data")
    st.dataframe(ss.head())

    # Define X and y
    X = ss[features]
    y = ss['sm_li']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Step 2: Train Model ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(class_weight='balanced', random_state=42, max_iter=500)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)

    st.write(f"### Model Accuracy: {accuracy:.2f}")

    # Display confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    st.write("### Confusion Matrix")
    st.dataframe(pd.DataFrame(cm, index=["Non-User", "User"], columns=["Predicted Non-User", "Predicted User"]))

    # Display classification report
    st.write("### Classification Report")
    st.text(classification_report(y_test, y_pred))

    # --- Step 3: Visualize Data ---
    st.write("### Feature Distributions by LinkedIn Usage")
    for feature in features:
        st.write(f"**{feature} vs LinkedIn Usage**")
        fig, ax = plt.subplots()
        sns.boxplot(x='sm_li', y=feature, data=ss, ax=ax)
        ax.set_title(f'{feature} vs LinkedIn Usage')
        ax.set_xlabel('LinkedIn Usage (0 = No, 1 = Yes)')
        ax.set_ylabel(feature)
        st.pyplot(fig)

    # --- Step 4: Make Predictions ---
    st.write("### Make a Prediction")
    with st.form("prediction_form"):
        income = st.slider("Income Level (1-9)", 1, 9, value=5)
        educ2 = st.slider("Education Level (1-8)", 1, 8, value=4)
        parent = st.selectbox("Are you a parent?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        married = st.selectbox("Are you married?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        female = st.selectbox("Gender", [0, 1], format_func=lambda x: "Female" if x == 1 else "Male")
        age = st.slider("Age (18-98)", 18, 98, value=30)
        submitted = st.form_submit_button("Predict")

        if submitted:
            input_data = pd.DataFrame([[income, educ2, parent, married, female, age]], columns=features)
            input_data_scaled = scaler.transform(input_data)
            prediction = model.predict(input_data_scaled)
            probability = model.predict_proba(input_data_scaled)

            st.write(f"Prediction: {'LinkedIn User' if prediction[0] == 1 else 'Not a LinkedIn User'}")
            st.write(f"Probability of using LinkedIn: {probability[0][1]:.2f}")

