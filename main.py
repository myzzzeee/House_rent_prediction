import joblib
import streamlit as st
import numpy as np

st.title("House Rent Prediction — Linear Regression Model")


label_encoders = joblib.load("label_encoders.pkl")
lr = joblib.load("linear regression.pkl")


def options_for(col):
    return sorted(label_encoders[col].classes_.tolist())

Bhk = st.number_input('BHK', min_value=1, step=1)
Size = st.number_input('Size (sqft)', min_value=1.0)

Floor = st.selectbox('Floor', options_for('Floor'))
Area_Type = st.selectbox('Area Type', options_for('Area Type'))
Area_Locality = st.selectbox('Area Locality', options_for('Area Locality'))
City = st.selectbox('City', options_for('City'))
Furnishing_Status = st.selectbox('Furnishing Status', options_for('Furnishing Status'))
Tenant_Preferred = st.selectbox('Tenant Preferred', options_for('Tenant Preferred'))
Point_of_contact = st.selectbox('Point of Contact', options_for('Point of Contact'))

Bathroom = st.number_input('Bathroom', min_value=1, step=1)

if st.button("Predict Rent"):
    categorical_inputs = {
        'Floor': Floor,
        'Area Type': Area_Type,
        'Area Locality': Area_Locality,
        'City': City,
        'Furnishing Status': Furnishing_Status,
        'Tenant Preferred': Tenant_Preferred,
        'Point of Contact': Point_of_contact
    }

    try:
        encoded = {}
        for col, value in categorical_inputs.items():
            le = label_encoders[col]
            encoded[col] = le.transform([value])[0]

        features = np.array([[
            Bhk,
            Size,
            encoded['Floor'],
            encoded['Area Type'],
            encoded['Area Locality'],
            encoded['City'],
            encoded['Furnishing Status'],
            encoded['Tenant Preferred'],
            Bathroom,
            encoded['Point of Contact']
        ]])

        #predicted_rent = lr.predict(features)[0]
        predicted_rent = max(0, lr.predict(features)[0])
        st.success(f"Predicted Rent: {predicted_rent:,.0f}")

    except ValueError as e:
        st.error(f"One of the entered values wasn't seen during training: {e}")
