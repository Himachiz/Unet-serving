from fastapi import FastAPI
import numpy as np
from app.model import Model

api = FastAPI()
model = Model("models/weights.npy")

@api.get("/predict")
def predict():
    return {"score": model.predict(np.ones(16))}
