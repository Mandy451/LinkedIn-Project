import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# --- Helper Functions ---
def clean_sm(x):
    """Convert non-1 values to 0."""
    return np.where(x == 1, 1, 0)

@st.cache_data
def load_and_preprocess_data():
    """Load and preprocess the dataset from GitHub."""
    url = "https://raw.githubusercontent.com/Mandy451/LinkedIn-Project/main/social_media_usage.csv"
    data = pd.read_csv(url)

    # Preprocess data
    data['sm_li'] = clean_sm(data['web1h'])
    features = ['income', 'educ2', 'par', 'marital', 'gender', 'age']
    data['income'] = data['income'].where(data['income'] <= 9)
    data['educ2'] = data['educ2'].where(data['educ2'] <= 8)
    data['age'] = data['age'].where(data['age'] <= 98)
    data['female'] = data['gender'].apply(lambda x: 1 if x == 2 else 0)
    data['parent'] = data['par'].apply(lambda x: 1 if x == 1 else 0)
    data['married'] = data['marital'].apply(lambda x: 1 if x == 1 else 0)
    data.dropna(inplace=True)
    return data, features

# Labels for income and education levels
income_labels = {
    1: "Less than $10,000",
    2: "$10,000 to under $20,000",
    3: "$20,000 to under $30,000",
    4: "$30,000 to under $40,000",
    5: "$40,000 to under $50,000",
    6: "$50,000 to under $75,000",
    7: "$75,000 to under $100,000",
    8: "$100,000 to under $150,000",
    9: "$150,000 or more"
}

educ2_labels = {
    1: "Less than high school",
    2: "High school incomplete",
    3: "High school graduate",
    4: "Some college, no degree",
    5: "Two-year associate degree",
    6: "Four-year college degree",
    7: "Some postgraduate, no degree",
    8: "Postgraduate or professional degree"
}

# --- App Layout ---
st.title("LinkedIn Usage Prediction App")
st.markdown("Upload your dataset to predict LinkedIn usage based on demographic data.")

# Load and preprocess the data
data, features = load_and_preprocess_data()
st.write("### Dataset Preview")
st.dataframe(data.head())

# Train-Test Split
X = data[features]
y = data['sm_li']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression Model
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(class_weight='balanced', random_state=42, max_iter=500)
model.fit(X_train_scaled, y_train)

# Model Evaluation
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
st.write(f"### Model Accuracy: {accuracy:.2f}")
cm = confusion_matrix(y_test, y_pred)
st.write("### Confusion Matrix")
st.dataframe(pd.DataFrame(cm, index=["Non-User", "User"], columns=["Predicted Non-User", "Predicted User"]))
st.write("### Classification Report")
st.text(classification_report(y_test, y_pred))

# Visualizations
st.write("### Feature Distributions by LinkedIn Usage")
for feature in features:
    fig, ax = plt.subplots()
    sns.boxplot(x='sm_li', y=feature, data=data, ax=ax)
    ax.set_title(f'{feature} vs LinkedIn Usage')
    st.pyplot(fig)

# User Input for Prediction
st.write("### Make a Prediction")
with st.form("prediction_form"):
    income = st.selectbox("Household Income", options=list(income_labels.keys()), format_func=lambda x: income_labels[x])
    educ2 = st.selectbox("Education Level", options=list(educ2_labels.keys()), format_func=lambda x: educ2_labels[x])
    parent = st.selectbox("Are you a parent?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    married = st.selectbox("Are you married?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    female = st.selectbox("Gender", [0, 1], format_func=lambda x: "Female" if x == 1 else "Male")
    age = st.slider("Age", 18, 98, value=30)
    submitted = st.form_submit_button("Predict")

    if submitted:
        input_data = pd.DataFrame([[income, educ2, parent, married, female, age]], columns=features)
        input_data_scaled = scaler.transform(input_data)
        prediction = model.predict(input_data_scaled)
        probability = model.predict_proba(input_data_scaled)

        st.write(f"Prediction: {'LinkedIn User' if prediction[0] == 1 else 'Not a LinkedIn User'}")
        st.write(f"Probability of LinkedIn User: {probability[0][1]:.2f}")
