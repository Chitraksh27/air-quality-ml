import streamlit as st
import joblib
import numpy as np
import pandas as pd
import math
from datetime import datetime
from openaq import OpenAQ
import os
from dotenv import load_dotenv

load_dotenv()

# 1. Load Artifacts efficiently using Streamlit's cache
@st.cache_resource
def load_artifacts():
    with open('saved_model_artifacts/features.pkl', 'rb') as f:
        expected_features = joblib.load(f)
    with open('saved_model_artifacts/medians.pkl', 'rb') as f:
        medians = joblib.load(f)
    with open('saved_model_artifacts/scaler.pkl', 'rb') as f:
        scaler = joblib.load(f)
    with open('saved_model_artifacts/all_models.pkl', 'rb') as f:
        models = joblib.load(f)
    return expected_features, medians, scaler, models

expected_features, medians, scaler, models = load_artifacts()

st.title("Real-Time CO(GT) Predictive Dashboard")
st.write("Compare Machine Learning model predictions against live OpenAQ sensor data.")

# 2. The Dynamic Trigger Button
if st.button("Fetch Live Data & Compare Models"):
    with st.spinner("Fetching data from OpenAQ API..."):
        
        # --- API FETCHING PHASE ---
        client = OpenAQ(api_key=os.getenv("OPENAQ_API")) 
        station_id = 8118 # New Delhi Station Example
        
        try:
            # 1. Fetch location metadata to map Sensor IDs to Parameter Names
            location_data = client.locations.get(station_id)
            
            sensor_map = {}
            if location_data.results and len(location_data.results) > 0:
                station = location_data.results[0]
                # Check for sensors safely
                if getattr(station, 'sensors', None):
                    for sensor in station.sensors:
                        # Map the unique sensor ID to its parameter name (e.g., 'no2', 'co')
                        sensor_map[sensor.id] = sensor.parameter.name.lower()
            
            # 2. Fetch the actual latest measurements for this location
            latest_data = client.locations.latest(station_id)
            
            # -> client.close() REMOVED here so you can test freely <-
            
            # Set fallback defaults from historical medians in case the live station 
            # is temporarily missing a specific sensor during the API call.
            live_no2 = medians.get('NO2(GT)', 45.0)
            live_temp = medians.get('T', 30.0)
            live_rh = medians.get('RH', 50.0)
            live_o3 = medians.get('PT08.S5(O3)', 25.0) 
            live_co_ppm = 1.5 # Fallback CO
            
            # 3. Parse the latest measurements using our map
            if getattr(latest_data, 'results', None):
                for reading in latest_data.results:
                    # Python SDK maps API's 'sensorsId' to 'sensors_id'. 
                    # Using getattr chain as a safety net to ensure it grabs the ID.
                    sensor_id = getattr(reading, 'sensors_id', getattr(reading, 'sensorsId', None))
                    val = getattr(reading, 'value', None)
                    
                    if sensor_id in sensor_map and val is not None:
                        param_name = sensor_map[sensor_id]
                        
                        if param_name == 'no2':
                            live_no2 = val
                        elif param_name == 'o3':
                            live_o3 = val
                        elif param_name == 'temperature':
                            live_temp = val
                        elif param_name in ['relativehumidity', 'rh']: 
                            live_rh = val
                        elif param_name == 'co':
                            live_co_ppm = val

            # --- PREPROCESSING & ENGINEERING PHASE ---
            # Convert API CO (ppm) to Ground Truth CO (mg/m³) for the final comparison
            actual_co_mgm3 = live_co_ppm * 1.15 
            
            # Thermodynamic Absolute Humidity calculation
            exponent = math.exp((17.67 * live_temp) / (live_temp + 243.5))
            live_ah = (6.112 * exponent * live_rh * 18.02) / ((273.15 + live_temp) * 100 * 0.08314)
            
            # Temporal engineering
            now = datetime.now()
            hour_sin = np.sin(2 * np.pi * now.hour / 23.0)
            hour_cos = np.cos(2 * np.pi * now.hour / 23.0)
            is_weekend = 1 if now.weekday() >= 5 else 0
            
            # Construct the 14-column feature vector using API data and imputed medians
            current_data = {
                'PT08.S1(CO)': medians.get('PT08.S1(CO)', 1000),
                'C6H6(GT)': medians.get('C6H6(GT)', 10.0),
                'PT08.S2(NMHC)': medians.get('PT08.S2(NMHC)', 900),
                'NOx(GT)': medians.get('NOx(GT)', 100), 
                'PT08.S3(NOx)': medians.get('PT08.S3(NOx)', 800),
                'NO2(GT)': live_no2,
                'PT08.S4(NO2)': medians.get('PT08.S4(NO2)', 1500),
                'PT08.S5(O3)': live_o3,
                'T': live_temp,
                'RH': live_rh,
                'AH': live_ah,
                'Month': now.month,
                'DayOfWeek': now.weekday(),
                'Is_Weekend': is_weekend,
                'Hour_sin': hour_sin,
                'Hour_cos': hour_cos
            }
            
            df_input = pd.DataFrame([current_data], columns=expected_features)
            scaled_input = scaler.transform(df_input)

           # --- MODEL COMPARISON PHASE ---
            st.success("Data fetched and processed successfully!")
            
            # Iterate through all models and collect predictions
            comparison_results = []  
            
            for model_name, model in models.items():
                pred = model.predict(scaled_input)[0] 
                error = abs(actual_co_mgm3 - pred)
                
                comparison_results.append({
                    "Model Algorithm": model_name,
                    "Predicted CO (mg/m³)": round(pred, 2),
                    "Absolute Error": round(error, 2)
                })
                
            df_results = pd.DataFrame(comparison_results)
            
            # Sort by error to highlight the best models first
            df_results = df_results.sort_values(by="Absolute Error").reset_index(drop=True)
            
            st.markdown("### 📊 Model Head-to-Head Performance")

            # Define number of columns per row (3 is usually perfect for 6 models)
            num_cols = 3
            
            # Loop through the dataframe in chunks of 'num_cols' to create rows
            for i in range(0, len(df_results), num_cols):
                cols = st.columns(num_cols)
                chunk = df_results.iloc[i:i+num_cols]
                
                for j, (_, row) in enumerate(chunk.iterrows()):
                    with cols[j]:
                        # Wrap each model's stats in a clean HTML card with a subtle border
                        st.markdown(
                            f"""
                            <div style="border: 1px solid #444; border-radius: 8px; padding: 15px; margin-bottom: 15px; background-color: rgba(255,255,255,0.02);">
                                <h4 style='text-align: center; color: #4DA8DA; margin-top: 0;'>{row['Model Algorithm']}</h4>
                                <div style='text-align: center; font-size: 16px;'><b>Predicted:</b> {row['Predicted CO (mg/m³)']:.2f}</div>
                                <div style='text-align: center; font-size: 16px;'><b>Actual:</b> {actual_co_mgm3:.2f}</div>
                                <div style='text-align: center; font-size: 16px; color: #ff6b6b;'><b>Error:</b> {row['Absolute Error']:.2f}</div>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
            
            st.markdown("---")
            
            # --- LINE CHART PHASE ---
            st.markdown("### 📈 Prediction Variance Across All Models")
            
            # Add the constant Actual Ground Truth value so it draws a flat baseline
            df_results["Actual CO (mg/m³)"] = actual_co_mgm3
            
            # Set the index to the Model Names so they appear on the X-Axis
            chart_data = df_results.set_index("Model Algorithm")[["Predicted CO (mg/m³)", "Actual CO (mg/m³)"]]
            
            # Plot the line chart
            st.line_chart(chart_data)

        except Exception as e:
            st.error(f"An error occurred while fetching or processing data: {e}")