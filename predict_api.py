from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

model = joblib.load("rf_wifi_threat_model.pkl")
encoder = joblib.load("rf_label_encoder.pkl")

app = FastAPI()

class InputFeatures(BaseModel):
    features: list

@app.post("/predict")
def predict_threat(data: InputFeatures):
    features = np.array(data.features).reshape(1, -1)
    pred = model.predict(features)
    label = encoder.inverse_transform(pred)
    return {"threat": label[0]}
