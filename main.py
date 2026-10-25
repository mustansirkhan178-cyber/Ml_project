from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()

reg_model = joblib.load("regression_model.joblib")
clf_model = joblib.load("classification_model.joblib")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class HouseInput(BaseModel):
    area: float
    bedrooms: int
    bathrooms: int


class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.post("/predict/regression")
def predict_regression(data: HouseInput):
    features = pd.DataFrame(
        [[data.area, data.bedrooms, data.bathrooms]],
        columns=["area", "bedrooms", "bathrooms"],
    )
    prediction = reg_model.predict(features)
    return {"predicted_price": float(prediction[0])}

@app.post("/predict/classification")
def predict_classification(data: IrisInput):
    features = pd.DataFrame(
        [[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]],
        columns=["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"],
    )
    prediction = clf_model.predict(features)
    return {"predicted_species": str(prediction[0])}

