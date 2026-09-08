import os
from fastapi import FastAPI
from app.model import Model

api = FastAPI()
model = Model(os.environ["WEIGHTS"])

@api.get("/predict")
def predict(image: str):
    return model.predict(image)
