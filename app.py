import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor

# Load the saved models and encoders
with open('crop_nutrient_model.pkl', 'rb') as model_file:
    crop_nutrient_model = pickle.load(model_file)

with open('crop_encoder.pkl', 'rb') as encoder_file:
    crop_encoder = pickle.load(encoder_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

with open('best_rfc.pkl', 'rb') as best_rfc_file:
    best_rfc = pickle.load(best_rfc_file)

# Crop dictionary for decoding
crop_dict = { 0: 'Rice', 1: 'Maize', 2: 'Jute', 3: 'Cotton', 4: 'Coconut', 5: 'Papaya', 6: 'Orange',
             7: 'Apple', 8: 'Muskmelon', 9: 'Watermelon', 10: 'Grapes', 11: 'Mango', 12: 'Banana', 13: 'Pomegranate',
             14: 'Lentil', 15: 'Blackgram', 16: 'MungBean', 17: 'MothBeans', 18: 'PigeonPeas', 19: 'KidneyBeans',
             20: 'ChickPea', 21: 'Coffee' }

# Function to predict crop based on soil nutrients
def predict_crop(nitrogen, phosphorus, potassium, temperature, humidity, pH_value, rainfall):
    # Prepare input features for prediction
    input_features = pd.DataFrame([[nitrogen, phosphorus, potassium, temperature, humidity, pH_value, rainfall]],
                                  columns=['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'pH_Value', 'Rainfall'])

    # Scale the input features
    input_scaled = scaler.transform(input_features)

    # Predict using the first model
    crop_prediction = best_rfc.predict(input_scaled)

    # Convert the prediction to the crop name (ensure correct class label)
    predicted_crop_index = crop_prediction[0]
    predicted_crop = crop_dict.get(predicted_crop_index, "Unknown Crop")
    return predicted_crop

# Function to predict soil nutrients based on crop
def predict_nutrients(crop_name):
    # Ensure the crop name is capitalized for consistency
    crop_name = crop_name.capitalize()

    # Validate if the crop is in the crop dictionary
    if crop_name not in crop_dict.values():
        return "Invalid crop name. Please enter a valid crop."

    # One-hot encode the crop name
    crop_input = pd.DataFrame([[crop_name]], columns=['Crop'])
    crop_encoded = crop_encoder.transform(crop_input)

    # Predict using the second model
    nutrient_predictions = crop_nutrient_model.predict(crop_encoded)

    # Output the predicted nutrients
    return nutrient_predictions[0]

# Streamlit App Layout
st.title("Crop Prediction and  t Nutrient Recommendation")

# Sidebar for navigation
option = st.sidebar.selectbox("Select an Option", ["Predict Crop", "Recommend Nutrients"])

if option == "Predict Crop":
    st.subheader("Enter Soil Nutrients and Conditions to Predict Crop")
    
    # User inputs for soil nutrients and conditions
    nitrogen = st.number_input("Enter Nitrogen value", min_value=0.0, step=0.1)
    phosphorus = st.number_input("Enter Phosphorus value", min_value=0.0, step=0.1)
    potassium = st.number_input("Enter Potassium value", min_value=0.0, step=0.1)
    temperature = st.number_input("Enter Temperature value", min_value=-10.0, step=0.1)
    humidity = st.number_input("Enter Humidity value", min_value=0.0, step=0.1)
    pH_value = st.number_input("Enter pH Value", min_value=0.0, step=0.1)
    rainfall = st.number_input("Enter Rainfall value", min_value=0.0, step=0.1)

    if st.button("Predict Crop"):
        predicted_crop = predict_crop(nitrogen, phosphorus, potassium, temperature, humidity, pH_value, rainfall)
        st.write(f"The predicted crop is: {predicted_crop}")

elif option == "Recommend Nutrients":
    st.subheader("Enter Crop Name to Recommend Ideal Soil Nutrients")
    
    # User input for crop name
    crop_name = st.text_input("Enter Crop Name (e.g., Rice, Maize, Cotton)")

    if st.button("Recommend Nutrients"):
        nutrients = predict_nutrients(crop_name)
        if isinstance(nutrients, str):  # Check if the output is an error message
            st.write(nutrients)
        else:
            st.write(f"Ideal soil nutrients for {crop_name}:")
            st.write(f"Nitrogen: {nutrients[0]}")
            st.write(f"Phosphorus: {nutrients[1]}")
            st.write(f"Potassium: {nutrients[2]}")
            st.write(f"Temperature: {nutrients[3]}")
            st.write(f"Humidity: {nutrients[4]}")
            st.write(f"pH Value: {nutrients[5]}")
            st.write(f"Rainfall: {nutrients[6]}")
