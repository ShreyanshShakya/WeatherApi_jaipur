#final deployment
from flask import Flask, request, jsonify
import pandas as pd
import xgboost as xgb
import json
import os

app = Flask(__name__)

MODEL_FILE = 'weather_prediction_model.json'
COLUMNS_FILE = 'weather_model_columns.json'

model = None
model_columns = None

print("Loading model and columns")
try:
    model = xgb.XGBRegressor()
    model.load_model(MODEL_FILE)
    
    with open(COLUMNS_FILE, 'r') as f:
        model_columns = json.load(f)
        
    print("Model and columns loaded successfully.")
except Exception as e:
    print(f"ERROR: Could not load model files. Ensure '{MODEL_FILE}' and '{COLUMNS_FILE}' exist.")
    print(e)
    

@app.route('/')
def home():
    return "Weather Prediction API is running. Use the /predict endpoint."

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or model_columns is None:
        return jsonify({"error": "Model is not loaded. Check server logs."}), 500

    json_data = request.get_json()
    if not json_data or 'historical_data' not in json_data:
        return jsonify({"error": "Invalid input: 'historical_data' key missing"}), 400

    hist_df = pd.DataFrame(json_data['historical_data'])

   
    hist_df['date'] = pd.to_datetime(hist_df['date'])
    hist_df = hist_df.set_index('date')
    
    last_date = hist_df.index.max()
    prediction_date = last_date + pd.Timedelta(days=1)
    
    predict_df = pd.DataFrame(index=[prediction_date])


    predict_df['month'] = predict_df.index.month
    predict_df['day_of_year'] = predict_df.index.dayofyear
    predict_df['day_of_week'] = predict_df.index.dayofweek
    predict_df['year'] = predict_df.index.year


    predict_df['temp_lag_1'] = hist_df['temperature_2m'].iloc[-1]
    predict_df['temp_lag_2'] = hist_df['temperature_2m'].iloc[-2]
    predict_df['precip_lag_1'] = hist_df['precipitation'].iloc[-1]
    

    rolling_mean_7 = hist_df['temperature_2m'].rolling(window=7, min_periods=1).mean().iloc[-1]
    predict_df['temp_rolling_mean_7'] = rolling_mean_7

   
    for col in model_columns:
        if col not in predict_df.columns and col in hist_df.columns:
             predict_df[col] = hist_df[col].iloc[-1]
        elif col not in predict_df.columns:
            predict_df[col] = 0 
  
 
    try:
        predict_df = predict_df[model_columns]
        
        prediction = model.predict(predict_df)
        native_python_float = float(prediction[0])
        
        return jsonify({
            'prediction_date': prediction_date.strftime('%Y-%m-%d'),
            'predicted_avg_temp_celsius': round(native_python_float, 2)
        })
    except Exception as e:
        return jsonify({"error": f"Error during prediction: {e}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)


