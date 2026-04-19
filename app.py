import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# --- 1. SETUP ---
st.set_page_config(page_title="F1 2026 Predictor Pro", layout="wide")

# Load model
base_path = os.path.dirname(__file__)
model_path = os.path.join(base_path, 'f1_model.pkl')

try:
    model = joblib.load(model_path)
    expected_features = list(model.feature_names_in_)
except:
    st.error("Error: 'f1_model.pkl' not found. Ensure the 6-feature model is in the folder.")
    expected_features = ['grid', 'temperature', 'is_rainy', 'constructorId', 'circuitId', 'driverId']

# --- 2. THE DATABASES (Modern Era 2020-2026) ---
# Filtered for modern circuits only
track_mapping = {
    "Albert Park (Australia)": 1, 
    "Bahrain (Sakhir)": 3, 
    "Barcelona (Spain)": 4, 
    "Monaco (Monte Carlo)": 6, 
    "Montreal (Canada)": 7,
    "Silverstone (UK)": 9, 
    "Budapest (Hungary)": 11,
    "Spa (Belgium)": 13,
    "Monza (Italy)": 14, 
    "Interlagos (Brazil)": 18, 
    "Suzuka (Japan)": 22, 
    "Abu Dhabi (Yas Marina)": 24,
    "Zandvoort (Netherlands)": 39,
    "Mexico City (Mexico)": 32,
    "Austin (USA)": 69,
    "Singapore (Marina Bay)": 70, 
    "Red Bull Ring (Austria)": 70, # Uses same ID in many datasets
    "Baku (Azerbaijan)": 73,
    "Jeddah (Saudi Arabia)": 77,
    "Losail (Qatar)": 78,
    "Miami (USA)": 79,
    "Las Vegas (USA)": 80,
    "Madrid (Spain)": 215, # New for 2026
    "Imola (Italy)": 21,
    "Portimão (Portugal)": 75,
    "Mugello (Italy)": 76
}

# Constructors
constructor_mapping = {
    "McLaren": 1, "Ferrari": 6, "Red Bull": 9, "Mercedes": 131, 
    "Aston Martin": 117, "Williams": 3, "Alpine": 210, "Haas": 210, 
    "RB": 213, "Audi": 215, "Cadillac": 216
}

# Drivers (Ergast IDs)
driver_mapping = {
    "Max Verstappen": 830, "Lewis Hamilton": 1, "Lando Norris": 846,
    "Charles Leclerc": 844, "George Russell": 847, "Oscar Piastri": 857,
    "Fernando Alonso": 4, "Carlos Sainz": 832, "Kimi Antonelli": 859,
    "Oliver Bearman": 860, "Gabriel Bortoleto": 861, "Isack Hadjar": 862,
    "Liam Lawson": 858, "Sergio Perez": 815
}

# --- 3. SIDEBAR REFERENCE ---
with st.sidebar:
    st.header("ID Reference Guide")
    st.markdown("**Modern Circuits (2020-2026)**")
    # Clean DataFrame for the sidebar
    circ_df = pd.DataFrame(list(track_mapping.items()), columns=["Circuit", "ID"]).sort_values("Circuit")
    st.dataframe(circ_df, hide_index=True)
    
    st.subheader("Teams & Drivers")
    st.table(pd.DataFrame(list(constructor_mapping.items()), columns=["Team", "ID"]))

# --- 4. USER INTERFACE ---
st.title("🏎️ Formula 1 Winner Predictor")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        selected_racer = st.selectbox("Select Driver", list(driver_mapping.keys()))
        grid_pos = st.number_input("Starting Grid Position", 1, 20, 1)
        selected_team = st.selectbox("Select Team", list(constructor_mapping.keys()))
        
    with col2:
        selected_track = st.selectbox("Select Circuit", list(track_mapping.keys()))
        temp = st.slider("Track Temperature (°C)", 10, 50, 28)
        rain_option = st.radio("Is it Raining?", ["No", "Yes"], horizontal=True)

    submit = st.form_submit_button("Analyze Win Probability")

# --- 5. PREDICTION LOGIC ---
if submit:
    is_rainy = 1 if rain_option == "Yes" else 0
    
    input_data = {
        'grid': grid_pos,
        'temperature': temp,
        'is_rainy': is_rainy,
        'constructorId': constructor_mapping[selected_team],
        'circuitId': track_mapping[selected_track],
        'driverId': driver_mapping[selected_racer]
    }
    
    # Align columns with model training order
    input_df = pd.DataFrame([input_data])[expected_features]
    
    try:
        prob = model.predict_proba(input_df)[0][1]
        
        st.write("---")
        st.subheader(f"AI Analysis: {selected_racer} @ {selected_track}")
        
        if prob > 0.5:
            st.balloons()
            st.success(f"🏆 High Victory Probability: **{prob*100:.1f}%**")
        else:
            st.warning(f"📉 Predicted Win Probability: **{prob*100:.1f}%**")
            
        st.info(f"Using historical stats for Driver {driver_mapping[selected_racer]} and Team {constructor_mapping[selected_team]}.")
            
    except Exception as e:
        st.error(f"Prediction Error: {e}")