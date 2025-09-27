import pandas as pd
import xgboost as xgb
import json
import os

print("--- Starting Model Training Script ---")

TRAINING_DATA_FILE = 'Jaipur_2.csv'

if not os.path.exists(TRAINING_DATA_FILE):
    print(f"ERROR: Data file '{TRAINING_DATA_FILE}' not found.")
    print("Please make sure the training data CSV is in the same directory.")
    exit(1)

print(f"Loading training data from '{TRAINING_DATA_FILE}'...")
df = pd.read_csv(TRAINING_DATA_FILE) 
print("Successfully loaded data.")



df = df.drop(columns=['Unnamed: 0'], errors='ignore')
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')


daily_df = df.rename(columns={'tavg': 'temperature_2m', 'prcp': 'precipitation'})
daily_df = daily_df.fillna(method='ffill')
print("Data prepared for feature engineering.")



daily_df['month'] = daily_df.index.month
daily_df['day_of_year'] = daily_df.index.dayofyear
daily_df['day_of_week'] = daily_df.index.dayofweek
daily_df['year'] = daily_df.index.year

daily_df['temp_lag_1'] = daily_df['temperature_2m'].shift(1)
daily_df['temp_lag_2'] = daily_df['temperature_2m'].shift(2)
daily_df['precip_lag_1'] = daily_df['precipitation'].shift(1)

daily_df['temp_rolling_mean_7'] = daily_df['temperature_2m'].shift(1).rolling(window=7).mean()

daily_df_final = daily_df.dropna()
print("Feature engineering complete.")


y = daily_df_final['temperature_2m']
X = daily_df_final.drop('temperature_2m', axis=1)


X_numeric = X.select_dtypes(include=['number'])

model = xgb.XGBRegressor(n_estimators=500, early_stopping_rounds=50, learning_rate=0.05)
model.fit(X_numeric, y, verbose=False)
print("Model training complete.")


model.save_model('weather_prediction_model.json')
model_columns = list(X_numeric.columns)
with open('weather_model_columns.json', 'w') as f:
    json.dump(model_columns, f)

print("Model and columns artifacts have been saved successfully.")

