
# End-to-End Weather Prediction API

This repository contains the complete code for an end-to-end machine learning project that predicts the next day's average temperature. The project trains an XGBoost model on historical weather data and deploys it as a robust, containerized FastAPI application.

The entire workflow is automated within a Docker container—from data preparation and model training to serving the final API—ensuring perfect reproducibility and eliminating environment-related errors.

# 🎯 Model Performance

The final model achieves a Mean Absolute Error (MAE) of approximately 0.26°C, indicating a very high level of accuracy. The plot below shows the model's predictions (orange) closely tracking the actual temperatures (blue) on the unseen test dataset.

# ✨ Key Features

- Accurate Forecasting: Utilizes an XGBoost Regressor, a powerful gradient-boosting model.

- Intelligent Feature Engineering: Creates predictive features from historical data, including time-based attributes, lag features, and rolling window averages.

- High-Performance API: Built with FastAPI, providing high performance, automatic data validation, and self-generating interactive documentation.

- Containerized and Reproducible: The entire application is containerized with Docker, guaranteeing a consistent environment for both development and deployment.

- Automated Training on Deploy: The Docker container automatically runs the training script upon deployment, ensuring the model artifacts are always perfectly compatible with the server environment.

# 🛠️ Technology Stack

- Language: Python 3.11

- Machine Learning: XGBoost, Scikit-learn, Pandas

- API Framework: FastAPI

- Web Server: Uvicorn

- Containerization: Docker

# 🔄 Project Workflow

1. The project follows a complete MLOps workflow automated within the Docker container:

2. Data Preparation: The train_model.py script loads historical daily weather data for a specific city from a local CSV file.

3. Feature Engineering: The script creates new, predictive features from the time-series data to improve model performance.

4. Model Training: An XGBoost model is trained on the engineered features.

5. Artifact Generation: The script saves the trained model (.json) and the list of feature columns (.json) to disk.

6. API Serving: The app.py script loads these freshly generated artifacts into a FastAPI application, which exposes a /predict endpoint.

7. Containerization: The Dockerfile defines the environment, runs the training script, and starts the FastAPI server, packaging the entire process into a single, portable image.

# ⚡ API Usage
The deployed application provides a single endpoint for predictions: /predict. You can interact with it using any HTTP client or through the automatic documentation provided by FastAPI at the /docs endpoint.




```bash
curl -X 'POST' \
  'https://your-hugging-face-space-url/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "historical_data": [
        { "date": "2024-01-20", "temperature_2m": 15.5, "precipitation": 0.1, "month": 1, "day_of_year": 20, "day_of_week": 5, "year": 2024 },
        { "date": "2024-01-21", "temperature_2m": 16.0, "precipitation": 0.0, "month": 1, "day_of_year": 21, "day_of_week": 6, "year": 2024 },
        { "date": "2024-01-22", "temperature_2m": 15.8, "precipitation": 0.0, "month": 1, "day_of_year": 22, "day_of_week": 0, "year": 2024 },
        { "date": "2024-01-23", "temperature_2m": 16.2, "precipitation": 0.0, "month": 1, "day_of_year": 23, "day_of_week": 1, "year": 2024 },
        { "date": "2024-01-24", "temperature_2m": 16.5, "precipitation": 0.0, "month": 1, "day_of_year": 24, "day_of_week": 2, "year": 2024 },
        { "date": "2024-01-25", "temperature_2m": 16.8, "precipitation": 0.0, "month": 1, "day_of_year": 25, "day_of_week": 3, "year": 2024 },
        { "date": "2024-01-26", "temperature_2m": 17.0, "precipitation": 0.0, "month": 1, "day_of_year": 26, "day_of_week": 4, "year": 2024 }
    ]
}'

```
**Expected Response**
``` json
{
  "prediction_date": "2024-01-27",
  "predicted_avg_temp_celsius": 17.25
}
```
# 💻 Local Development and Testing

You can build and run this application on your local machine if you have Docker installed.

- Clone the repository.
- Add Training Data: Place your training data file (e.g., Jaipur_2.csv) in the root of the project directory.
- Build the Docker image.
- Run the Docker container:



**Example curl Request**

To get a prediction, send a POST request with at least 7 days of historical weather data.
