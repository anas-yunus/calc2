import streamlit as st
import pandas as pd
import joblib
import folium
from streamlit_folium import st_folium
import geopandas as gpd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load model and data
model = joblib.load('rainfall_model.pkl')
rainfall_data = pd.read_csv('data/rainfall_data.csv')
geo_data = gpd.read_file('district_map.geojson')  # Load GeoJSON file for district map

# Normalize names for consistency
def normalize_name(name):
    return name.strip().lower().replace(" ", "")

# Apply normalization to rainfall data for easier matching
rainfall_data['STATE_UT_NAME'] = rainfall_data['STATE_UT_NAME'].apply(normalize_name)
rainfall_data['DISTRICT'] = rainfall_data['DISTRICT'].apply(normalize_name)
geo_data['st_nm'] = geo_data['st_nm'].apply(normalize_name)
geo_data['district'] = geo_data['district'].apply(normalize_name)

# Predict rainfall for selected district
def predict_rainfall(district_data):
    # Extract the required columns from the Series, keeping them in a DataFrame format with correct feature names
    input_features = pd.DataFrame([district_data[['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']].values], 
                                  columns=['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'])

    predicted_rainfall = model.predict(input_features)
    return predicted_rainfall[0]


# Seasonal Forecast calculation
def calculate_seasonal_forecast(district_data):
    # Assuming seasonal averages are provided directly, you can also calculate these from the monthly data
    monsoon_avg = district_data[['JUN', 'JUL', 'AUG', 'SEP']].mean()  # mean for these months
    winter_avg = district_data[['DEC', 'JAN', 'FEB']].mean()  # mean for these months
    pre_monsoon_avg = district_data[['MAR', 'APR', 'MAY']].mean()  # mean for these months
    post_monsoon_avg = district_data[['OCT']].mean()  # only one month

    return monsoon_avg, winter_avg, pre_monsoon_avg, post_monsoon_avg

# Anomaly Detection (simple threshold-based anomaly)
def detect_anomalies(district_data):
    anomalies = {}
    # Compare each month to the overall mean and std of that month
    for month in ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']:
        # Directly get the value for that district for the specific month
        monthly_data = district_data[month]  # This is the value for the district, no need for .iloc[0]
        month_mean = rainfall_data[month].mean()  # Calculate mean for the entire dataset
        month_std = rainfall_data[month].std()  # Calculate std for the entire dataset
        
        # Simple anomaly detection: if the value deviates more than 2 standard deviations, it's an anomaly
        if monthly_data > month_mean + 2 * month_std:
            anomalies[month] = 'Anomaly Detected'
    
    return anomalies



# Comparison with neighboring districts (Assume neighboring districts are identified in the geo_data)
def compare_with_neigbors(district_data, state_name, district_name):
    # Get neighbors from geo_data or predefined list (assuming geo_data contains this info)
    neighbors = geo_data[geo_data['st_nm'] == state_name]  # Get districts from the same state
    neighbor_data = rainfall_data[rainfall_data['DISTRICT'].isin(neighbors['district'])]  # Get data for neighboring districts
    
    # Get comparison (mean rainfall for neighbors)
    neighbor_mean_rainfall = neighbor_data[['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']].mean(axis=1)
    return neighbor_mean_rainfall.mean()  # Return the average rainfall of neighboring districts

# Streamlit app
st.title("Rainfall Prediction for Indian Districts")

# Display map with clickable districts
m = folium.Map(location=[20.5937, 78.9629], zoom_start=5)

# Create a color map for rainfall prediction (for color-coding)
rainfall_data['predicted_rainfall'] = rainfall_data.apply(lambda row: predict_rainfall(row), axis=1)

for _, row in geo_data.iterrows():
    district_name = row['district']
    state_name = row['st_nm']

    # Define popup function
    def popup_html(district, state):
        # Filter the rainfall data for the clicked district and state
        district_data = rainfall_data[(rainfall_data['STATE_UT_NAME'] == state) & (rainfall_data['DISTRICT'] == district)]
        
        # Check if district data is found
        if not district_data.empty:
            latest_data = district_data.iloc[-1]  # Get latest record
            predicted_rainfall = predict_rainfall(latest_data)
            # Seasonal forecast
            monsoon_avg, winter_avg, pre_monsoon_avg, post_monsoon_avg = calculate_seasonal_forecast(latest_data)
            # Anomalies
            anomalies = detect_anomalies(latest_data)
            # Comparison with neighbors
            neighbor_rainfall = compare_with_neigbors(latest_data, state, district)

            popup_content = f"<strong>{district}, {state}</strong><br>"
            popup_content += f"Predicted Annual Rainfall: {predicted_rainfall:.2f} mm<br>"
            popup_content += f"Monsoon Avg: {monsoon_avg:.2f} mm<br>"
            popup_content += f"Winter Avg: {winter_avg:.2f} mm<br>"
            popup_content += f"Pre-Monsoon Avg: {pre_monsoon_avg:.2f} mm<br>"
            popup_content += f"Post-Monsoon Avg: {post_monsoon_avg:.2f} mm<br>"
            
            if anomalies:
                popup_content += "<strong>Anomalies detected in:</strong><br>"
                for month, status in anomalies.items():
                    popup_content += f"{month}: {status}<br>"

            popup_content += f"Comparison with Neighbors: {neighbor_rainfall:.2f} mm (avg)<br>"

            return popup_content
        else:
            # Display message with normalized names for troubleshooting
            return f"No data available for {district}, {state}"

    # Add polygon for each district with a popup
    district_boundary = folium.GeoJson(
        row['geometry'],
        name=district_name,
        tooltip=district_name,
        popup=folium.Popup(popup_html(district_name, state_name), max_width=300),
        style_function=lambda x: {
            'fillColor': 'green' if not rainfall_data.loc[rainfall_data['DISTRICT'] == district_name, 'predicted_rainfall'].empty and rainfall_data.loc[rainfall_data['DISTRICT'] == district_name, 'predicted_rainfall'].values[0] > 1000 else 'red', 
            'color': 'black', 
            'weight': 1
        }
    )
    district_boundary.add_to(m)


# Display map in Streamlit
st_data = st_folium(m, width=700, height=500)
