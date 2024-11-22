import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load data and train model
@st.cache
def load_and_train_model():
    # Update with actual file path if necessary
    file_path = 'dashboard1_modified - Sheet1.csv'
    data = pd.read_csv(file_path)
    
    # Selecting relevant columns and dropping missing values
    data = data[['year', 'district', 'population', 'week', 'rainsum', 'meantemperature', 'weekly_hospitalised_cases']].dropna()
    
    # Define features (X) and target (y)
    X = data[['year', 'district', 'population', 'week', 'rainsum', 'meantemperature']]
    y = data['weekly_hospitalised_cases']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    return model

# Load and cache the model
model = load_and_train_model()

# Streamlit app layout and inputs
st.title("Weekly Hospitalized Cases Prediction")

# Input fields for user to manually enter values
year = st.number_input("Enter the year:", min_value=2000, max_value=2100, value=2024)
district = st.number_input("Enter the district number:", min_value=1, value=1)
population = st.number_input("Enter the population:", min_value=0, value=100000)
week = st.number_input("Enter the week (1 to 52):", min_value=1, max_value=52, value=1)
rainsum = st.number_input("Enter the total rainfall for the week (in mm):", value=0.0)
meantemperature = st.number_input("Enter the mean temperature for the week (in °C):", value=20.0)

# Prepare input data for prediction
input_data = pd.DataFrame([[year, district, population, week, rainsum, meantemperature]], 
                          columns=['year', 'district', 'population', 'week', 'rainsum', 'meantemperature'])

# Prediction button
if st.button("Predict"):
    # Make prediction
    predicted_cases = model.predict(input_data)
    st.write(f"Predicted weekly hospitalized cases: {predicted_cases[0]:.2f}")
