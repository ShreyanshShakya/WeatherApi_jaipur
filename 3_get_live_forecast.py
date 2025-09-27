import requests
import pandas as pd

def get_live_weather_data():
    """Fetches the last 8 days of hourly weather data for Jaipur."""
    print("Fetching live weather data for Jaipur, Rajasthan...")
    
    # Coordinates for Jaipur, Rajasthan, India
    latitude = 26.92
    longitude = 75.79
    
    url = (
        f"https://archive-api.open-meteo.com/v1/archive?latitude={latitude}&longitude={longitude}"
        "&past_days=8"
        "&hourly=temperature_2m,relative_humidity_2m,dew_point_2m,apparent_temperature,precipitation,rain,snowfall,snow_depth,pressure_msl,surface_pressure,cloud_cover,cloud_cover_low,cloud_cover_mid,cloud_cover_high,wind_speed_10m,wind_speed_100m,wind_gusts_10m"
    )
    
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        print("Successfully fetched data from Open-Meteo.")
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None

def process_data_for_api(live_data):
    """Processes raw hourly data into the daily JSON format for our API."""
    print("Processing hourly data into daily summaries...")
    
    hourly_df = pd.DataFrame(live_data['hourly'])
    hourly_df = hourly_df.rename(columns={'time': 'date'})
    hourly_df['date'] = pd.to_datetime(hourly_df['date'])
    hourly_df = hourly_df.set_index('date')
    
    aggregation_logic = {
        'temperature_2m': 'mean', 'relative_humidity_2m': 'mean', 'dew_point_2m': 'mean',
        'apparent_temperature': 'mean', 'pressure_msl': 'mean', 'surface_pressure': 'mean',
        'cloud_cover': 'mean', 'cloud_cover_low': 'mean', 'cloud_cover_mid': 'mean',
        'cloud_cover_high': 'mean', 'wind_speed_10m': 'mean', 'wind_speed_100m': 'mean',
        'snow_depth': 'mean', 'precipitation': 'sum', 'rain': 'sum', 'snowfall': 'sum',
        'wind_gusts_10m': 'max'
    }

    daily_df = hourly_df.resample('D').agg(aggregation_logic).round(2)
    daily_df = daily_df.dropna()
    
    api_payload_df = daily_df.reset_index()
    api_payload_df['date'] = api_payload_df['date'].dt.strftime('%Y-%m-%d')
    
    final_json = {"historical_data": api_payload_df.to_dict(orient='records')}
    
    print("Data formatted for API.")
    return final_json
    
def get_prediction_from_api(api_payload):
    """Sends formatted data to the LIVE Flask API to get a prediction."""
    
    # --- Calling your live Render URL ---
    api_url = "https://weatherapi-jaipur.onrender.com/predict"
    print(f"Sending request to LIVE API at {api_url}...")
    
    try:
        response = requests.post(api_url, json=api_payload, timeout=20)
        response.raise_for_status()
        prediction = response.json()
        print("\n" + "="*30)
        print("   LIVE WEATHER FORECAST RECEIVED")
        print("="*30)
        print(f"  Date: {prediction['prediction_date']}")
        print(f"  Predicted Avg Temp: {prediction['predicted_avg_temp_celsius']}°C")
        print("="*30)
    except requests.exceptions.RequestException as e:
        print(f"\n--- API CALL FAILED ---")
        print(f"Could not get prediction from the live API: {e}")
        print("Check the logs on your Render dashboard for more details.")

if __name__ == "__main__":
    live_weather_data = get_live_weather_data()
    
    if live_weather_data:
        payload = process_data_for_api(live_weather_data)
        get_prediction_from_api(payload)

