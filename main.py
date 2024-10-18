from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import uvicorn


# Initialize FastAPI app
app = FastAPI()

# Load the trained model
try:
    model = load_model('./model.h5')
except:
    raise HTTPException(status_code=500, detail="Failed to load the model")

# Load the scaler
try:
    scaler = joblib.load('./scaler.pkl')
except:
    raise HTTPException(status_code=500, detail="Failed to load the scaler")

# Define the input data format
class InputData(BaseModel):
    features: list[float]

# Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'Hello Everyobe '}

# Prediction endpoint
@app.post("/predict")
async def predict(data: InputData):
    try:
        # Convert input data to numpy array and reshape
        features = np.array(data.features).reshape(1, -1)
        
        # Check if the number of features matches the model's expectation
        if features.shape[1] != 20:
            raise ValueError(f"Expected 20 features, but got {features.shape[1]}")
        
        # Scale the features
        features = scaler.transform(features)
        
        # Make the prediction
        prediction = model.predict(features)
        predicted_class = 1 if prediction > 0.5 else 0
        
        return {"prediction": int(predicted_class)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Health check endpoint
@app.get("/health")
async def health():
    return {"status": "OK"}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
