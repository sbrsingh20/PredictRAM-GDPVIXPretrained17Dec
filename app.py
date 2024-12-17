import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# Load the overall results to get the models
results_folder = 'results'

# Load overall results for model evaluation
overall_results_filename = os.path.join(results_folder, 'overall_gdp_vix_xgb_model_evaluation.pkl')
overall_results = joblib.load(overall_results_filename)

# Create a dictionary to store model files for each stock
stock_models = {}

# Populate stock_models with the paths to the pre-trained models
for result in overall_results:
    stock_models[result['stock']] = result['model']

# Streamlit User Interface
st.title('Stock Return Prediction using GDP, Inflation, Interest Rate, and VIX')
st.write('This tool allows you to predict stock returns based on GDP, Inflation, Interest Rate, and VIX data using pre-trained models.')

# Dropdown to select the stock
stock_selection = st.selectbox('Select a Stock:', list(stock_models.keys()))

# Get the pre-trained model for the selected stock
selected_model = stock_models[stock_selection]

# Input fields for GDP, Inflation, Interest Rate, and VIX
gdp_value = st.number_input('Enter the expected GDP value:', min_value=0.0, value=3.0, step=0.1)
inflation_value = st.number_input('Enter the expected Inflation rate (%):', min_value=0.0, value=2.0, step=0.1)
interest_rate_value = st.number_input('Enter the expected Interest Rate (%):', min_value=0.0, value=5.0, step=0.1)
vix_value = st.number_input('Enter the expected VIX value:', min_value=0.0, value=20.0, step=0.1)

# Prepare the input data as a dataframe
input_data = pd.DataFrame({
    'GDP': [gdp_value],
    'Inflation': [inflation_value],
    'Interest Rate': [interest_rate_value],
    'VIX': [vix_value]
})

# Scaling the input data using the same scaler used during training
scaler = selected_model.named_steps['scaler']
scaled_input = scaler.transform(input_data)

# Prediction
if st.button('Predict Stock Return'):
    # Make prediction using the pre-trained model
    stock_return_prediction = selected_model.predict(scaled_input)
    
    # Display the predicted stock return
    st.write(f'Predicted Stock Return for {stock_selection}: {stock_return_prediction[0]:.4f}')
