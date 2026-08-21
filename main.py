import joblib
import streamlit as st
import numpy as np

st.set_page_config(page_title="House Rent Predictor", page_icon="🏠", layout="centered")

# ---------- Simple styling ----------
st.markdown("""
    <style>
        .main {
            padding-top: 2rem;
        }
        h1 {
            font-size: 2rem !important;
            font-weight: 700 !important;
            margin-bottom: 0.2rem !important;
        }
        .subtitle {
            color: #9CA3AF;
            font-size: 0.95rem;
            margin-bottom: 1.8rem;
        }
        div.stButton > button {
            width: 100%;
            min-height: 3rem;
            background: linear-gradient(135deg, #3B82F6, #1D4ED8);
            color: white;
            font-weight: 600;
            font-size: 1.1rem;
            padding: 0.9rem 1.5rem;
            border-radius: 10px;
            border: none;
            margin-top: 1.2rem;
            box-shadow: 0 4px 14px rgba(37, 99, 235, 0.35);
            transition: all 0.2s ease-in-out;
            letter-spacing: 0.3px;
            white-space: nowrap;
        }
        div.stButton > button:hover {
            background: linear-gradient(135deg, #2563EB, #1E40AF);
            color: white;
            box-shadow: 0 6px 18px rgba(37, 99, 235, 0.5);
            transform: translateY(-1px);
        }
        div.stButton > button:active {
            transform: translateY(0px);
            box-shadow: 0 2px 8px rgba(37, 99, 235, 0.4);
        }
        .result-box {
            background-color: #064E3B;
            border: 1px solid #10B981;
            padding: 1.2rem;
            border-radius: 10px;
            text-align: center;
            margin-top: 1.2rem;
        }
        .result-box h2 {
            color: #34D399;
            margin: 0;
            font-size: 1.6rem;
        }
        .result-box p {
            color: #A7F3D0;
            margin: 0.2rem 0 0 0;
            font-size: 0.85rem;
        }
    </style>
""", unsafe_allow_html=True)

# ---------- Header ----------
st.markdown("# 🏠 House Rent Predictor")
st.markdown('<p class="subtitle">Fill in the property details to estimate monthly rent</p>', unsafe_allow_html=True)

# ---------- Load model + encoders ----------
label_encoders = joblib.load("label_encoders.pkl")
lr = joblib.load("linear regression.pkl")

def options_for(col):
    return sorted(label_encoders[col].classes_.tolist())

# ---------- Inputs ----------
st.markdown("##### Property Details")
col1, col2 = st.columns(2)
with col1:
    Bhk = st.number_input('BHK', min_value=1, step=1)
    Size = st.number_input('Size (sqft)', min_value=1.0)
    Bathroom = st.number_input('Bathroom', min_value=1, step=1)
with col2:
    Floor = st.selectbox('Floor', options_for('Floor'))
    Area_Type = st.selectbox('Area Type', options_for('Area Type'))
    Area_Locality = st.selectbox('Area Locality', options_for('Area Locality'))

st.markdown("##### Location & Preferences")
col3, col4 = st.columns(2)
with col3:
    City = st.selectbox('City', options_for('City'))
    Furnishing_Status = st.selectbox('Furnishing Status', options_for('Furnishing Status'))
with col4:
    Tenant_Preferred = st.selectbox('Tenant Preferred', options_for('Tenant Preferred'))
    Point_of_contact = st.selectbox('Point of Contact', options_for('Point of Contact'))

# ---------- Predict ----------
if st.button("🔍  Predict Rent"):
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

        predicted_rent = max(0, lr.predict(features)[0])

        st.markdown(f"""
            <div class="result-box">
                <p>ESTIMATED MONTHLY RENT</p>
                <h2>₹{predicted_rent:,.0f}</h2>
            </div>
        """, unsafe_allow_html=True)

    except ValueError as e:
        st.error(f"One of the entered values wasn't seen during training: {e}")
