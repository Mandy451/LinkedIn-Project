#!/usr/bin/env python
# coding: utf-8

# # Final Project
# ## Mandy Snell
# ### 13 December 2024
# ---

# ### Load Libraries
# ---

# In[3]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


# ### 1. Read in data & check shape
# ---

# In[4]:


s = pd.read_csv('/Users/mandy/Downloads/social_media_usage.csv')
print(s.shape)
s.head()


# ### 2. Define & test clean_sm
# ---

# In[12]:


def clean_sm(x):
    return np.where(x == 1, 1, 0)  

toy_df = pd.DataFrame({'A': [1, 0, 2], 'B': [1, 1, 0]})
toy_df['A_clean'] = clean_sm(toy_df['A'])
print(toy_df)


# ### 3. Create dataframe "ss" (working data)
# ---

# In[23]:


# Make a copy of the original dataset
ss = s.copy()

# Clean the target column using `clean_sm`
ss['sm_li'] = clean_sm(ss['web1h'])

# Select relevant features
features = ['income', 'educ2', 'par', 'marital', 'gender', 'age']

# Keep only the selected features and the target column
ss = ss[features + ['sm_li']]

# Clean feature data:
# - For income, values above 9 should be set as missing.
ss['income'] = ss['income'].where(ss['income'] <= 9)

# - For education, values above 8 should be set as missing.
ss['educ2'] = ss['educ2'].where(ss['educ2'] <= 8)

# - For age, set values above 98 as missing.
ss['age'] = ss['age'].where(ss['age'] <= 98)

# - For 'gender', set '1' as male, '2' as female, and other values as missing.
ss['female'] = ss['gender'].apply(lambda x: 1 if x == 2 else 0)

# - Parent column: set '1' as parent, '2' as non-parent.
ss['parent'] = ss['par'].apply(lambda x: 1 if x == 1 else 0)

# - Marital column: 1 is married, others will be treated as non-married for simplicity.
ss['married'] = ss['marital'].apply(lambda x: 1 if x == 1 else 0)

# Drop rows with missing values
ss.dropna(inplace=True)

# Display the cleaned dataset
print(ss.head())
print(ss.shape)  # Check the dimensions after cleaning


# ### 3a. Exploratory Data Analysis & check out distribution

# In[31]:


# Create boxplots for each feature
for feature in features:
    plt.figure(figsize=(8, 6))  # Adjust size for readability
    sns.boxplot(x='sm_li', y=feature, data=ss)
    plt.title(f'{feature} vs LinkedIn Usage')
    plt.xlabel('LinkedIn Usage (0 = No, 1 = Yes)')
    plt.ylabel(feature)
    plt.show()


# ### 4. Create target vector (y) and feature set (X)
# ---

# In[33]:


# Separate target variable (y) and feature set (X)
y = ss['sm_li']  # The target column indicating LinkedIn usage
X = ss.drop(columns=['sm_li'])  # Drop the target column from the feature set

# Display the first few rows of X and y to verify
print(X.head())
print(y.head())


# ### 5. Split data into training and test sets
# ---

# In[46]:


from sklearn.model_selection import train_test_split

# Drop the target column from the feature set
X = ss.drop(columns=['sm_li'])

# Split the data into training and test sets (80% training, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display the size of each split
print(f"Training set size: {X_train.shape[0]} rows")
print(f"Test set size: {X_test.shape[0]} rows")


# ### 6. Train the model
# ---

# In[82]:


from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Scale the features
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

# Instantiate the logistic regression model with balanced class weights
model = LogisticRegression(class_weight='balanced', random_state=42, max_iter=500)

# Fit the model with the scaled training data
model.fit(X_train_scaled, y_train)


# ### 7. Evaluate model for accuracy using test data
# ---

# In[84]:


y_pred = model.predict(X_test_scaled)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)


# ### 8. Confusion matrix dataframe
# ---

# In[86]:


# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Create a DataFrame for the confusion matrix
cm_df = pd.DataFrame(cm, 
                     index=['Actual: Non-LinkedIn User', 'Actual: LinkedIn User'], 
                     columns=['Predicted: Non-LinkedIn User', 'Predicted: LinkedIn User'])

# Display the confusion matrix as a DataFrame
print(cm_df)


# ### 9. Create a classification_report using sklearn 
# ---

# In[88]:


from sklearn.metrics import classification_report

# Generate classification report
report = classification_report(y_test, y_pred)

# Display the report
print(report)


# ### 10. Predictions
# ---

# In[92]:


# Define new data for prediction
person_1 = pd.DataFrame([[8, 7, 0, 1, 1, 42, 0, 1, 2]], columns=X_train.columns)
person_2 = pd.DataFrame([[8, 7, 0, 1, 1, 82, 0, 1, 2]], columns=X_train.columns)

# Scale the data using the same scaler as during training
person_1_scaled = pd.DataFrame(scaler.transform(person_1), columns=X_train.columns)
person_2_scaled = pd.DataFrame(scaler.transform(person_2), columns=X_train.columns)

# Make predictions
prediction_1 = model.predict(person_1_scaled)
prediction_2 = model.predict(person_2_scaled)

# Probability predictions
proba_1 = model.predict_proba(person_1_scaled)
proba_2 = model.predict_proba(person_2_scaled)

print(f"Prediction for Person 1: {prediction_1}, Probability: {proba_1}")
print(f"Prediction for Person 2: {prediction_2}, Probability: {proba_2}")


# In[ ]:




