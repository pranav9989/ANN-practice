import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle
import streamlit as st
import tensorflow as tf
# Load the trained model
model = tf.keras.models.load_model('model.h5')

#Load the encoders and scalers
with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)


##Streamlit app
st.title("Customer Churn Prediction")

#User Input
credit_score = st.number_input("Credit Score", min_value=300, max_value=900, )
geography = st.selectbox("Geography", ['France', 'Germany', 'Spain'])
gender = st.selectbox("Gender", ['Male', 'Female'])
age = st.number_input("Age", min_value=18, max_value=100, )
tenure = st.number_input("Tenure (years with bank)", min_value=0, max_value=10,)
balance = st.number_input("Account Balance", min_value=0.0, step=100.0, )
num_of_products = st.number_input("Number of Products", min_value=1, max_value=4, )
has_cr_card = st.selectbox("Has Credit Card", [1, 0], )
is_active_member = st.selectbox("Is Active Member", [1, 0], )
estimated_salary = st.number_input("Estimated Salary", min_value=0.0, step=100.0, )

input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Geography': [geography],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns = onehot_encoder_geo.get_feature_names_out(['Geography']))

input_df = pd.concat([input_data.drop(columns=["Geography"], axis=1), geo_encoded_df], axis=1)

input_df_scaled = scaler.transform(input_df)

prediction = model.predict(input_df_scaled)
prediction_proba = prediction[0][0]

st.write("Predicted Probability: ", prediction_proba)

if prediction_proba > 0.5:
    st.write("The customer is likely to churn")
else:
    st.write("The customer is not likely to churn")