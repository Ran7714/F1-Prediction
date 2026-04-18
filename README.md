# F1-Prediction
Developed an AI-driven F1 Predictor that calculates race win probabilities based on racer skill and 2026 regulations. Streamlines complex sports data into insights.
Project Overview
This is a Predictive Data Science project designed to forecast the probability of a driver winning a Formula 1 Grand Prix. While most basic models focus solely on the car's speed, this project implements a 6-feature AI model that accounts for individual driver skill, team engineering dominance, and real-time environmental conditions.

The project is specifically configured for the 2026 Season, featuring the new regulatory shift, updated team lineups (Audi/Cadillac), and the introduction of the Madrid Street Circuit.

Technical Architecture
The core of this application is a Random Forest Classifier—an ensemble learning method that handles complex, non-linear relationships between racing variables.

Key Features (Inputs):
Grid Position: The starting slot of the driver (1-20).

Driver ID: Historical skill weighting for the specific racer.

Constructor ID: Technical performance weighting for the team.

Circuit ID: Unique characteristics of the track.

Track Temperature: Impact of heat on tire degradation and engine cooling.

Weather (is_rainy): Binary input for wet/dry track conditions.

Technology Stack
Language: Python 3.9+

Data Manipulation: Pandas, NumPy

Machine Learning: Scikit-Learn (Random Forest)

Model Persistence: Joblib

Deployment/UI: Streamlit

Project Structure
app.py: The Streamlit dashboard and frontend logic.

f1_model.pkl: The trained 6-feature Random Forest model.

requirements.txt: Necessary dependencies to run the app.

circuits.csv: Reference data for the circuit mapping.

Installation & Usage
Clone the Repository:
git clone https://github.com/yourusername/f1-predictor-2026.git
cd f1-predictor-2026

Install Dependencies:
pip install -r requirements.txt

Launch the App:
python -m streamlit run app.py

Data Insights & Methodology
The model was trained on the Ergast F1 Dataset, spanning historical data from 1950 to the current 2026 season. By utilizing predict_proba(), the application doesn't just give a "Yes/No" answer but provides a calculated percentage probability, reflecting the inherent uncertainty in high-speed motorsport.
