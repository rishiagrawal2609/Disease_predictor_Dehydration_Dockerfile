import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

# Load the model from .h5 file
model = load_model("./model.h5")

# Initialize FastAPI
app = FastAPI()

# Define input schema
class ModelInput(BaseModel):
    feature1: float  # First biometric/physiological measurement
    feature2: float  # Second measurement (e.g., heart rate, temp)
    feature3: float  # Third measurement


@app.get("/")
def home():
    return {"message": "Welcome to the model API!"}

@app.post("/predict")
def predict(data: ModelInput):
    # Convert input data to a numpy array
    input_data = np.array([
        data.feature1, data.feature2, data.feature3
    ]).reshape(1, -1)

    # Make prediction
    prediction = model.predict(input_data)

    # Convert output to list
    prediction = prediction.tolist()

    return {"prediction": prediction}

# Run the FastAPI Server (Uncomment the following lines to run directly)
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)