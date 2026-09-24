from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import pickle
from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = BASE_DIR / "train.csv"
MODEL_FILE = BASE_DIR / "model.pkl"


# --------------------------------------------------
# FASTAPI APP
# --------------------------------------------------

app = FastAPI(
    title="Linear Regression Prediction API",
    description="Linear Regression model using FastAPI",
    version="1.0"
)


# --------------------------------------------------
# CORS
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# LOAD TRAINING DATA
# --------------------------------------------------

data = pd.read_csv(DATA_FILE)

X = data[["hours"]]
y = data["marks"]


# --------------------------------------------------
# TRAIN MODEL
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()

model.fit(X_train, y_train)


# --------------------------------------------------
# MODEL EVALUATION
# --------------------------------------------------

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)

r2 = r2_score(y_test, y_pred)


# --------------------------------------------------
# SAVE MODEL
# --------------------------------------------------

with open(MODEL_FILE, "wb") as file:
    pickle.dump(model, file)


# --------------------------------------------------
# REQUEST FORMAT
# --------------------------------------------------

class PredictionRequest(BaseModel):
    hours: float


# --------------------------------------------------
# HOME
# --------------------------------------------------

@app.get("/")
def home():

    return {
        "message": "Linear Regression API is running",
        "algorithm": "Linear Regression"
    }


# --------------------------------------------------
# MODEL INFORMATION
# --------------------------------------------------

@app.get("/model-info")
def model_info():

    return {
        "algorithm": "Linear Regression",

        "training_samples": len(data),

        "slope": round(
            float(model.coef_[0]),
            4
        ),

        "intercept": round(
            float(model.intercept_),
            4
        ),

        "mean_squared_error": round(
            float(mse),
            2
        ),

        "r2_score": round(
            float(r2),
            4
        )
    }


# --------------------------------------------------
# GRAPH DATA
# --------------------------------------------------

@app.get("/graph-data")
def graph_data():

    # Sort data by hours

    sorted_data = data.sort_values(
        by="hours"
    )

    hours = sorted_data["hours"].tolist()

    marks = sorted_data["marks"].tolist()


    # Generate regression line values

    predicted_marks = model.predict(
        sorted_data[["hours"]]
    )


    return {

        "hours": hours,

        "actual_marks": marks,

        "predicted_marks": [
            round(float(value), 2)
            for value in predicted_marks
        ]

    }


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

@app.post("/predict")
def predict(request: PredictionRequest):

    input_data = pd.DataFrame(
        [[request.hours]],
        columns=["hours"]
    )

    prediction = model.predict(
        input_data
    )

    return {

        "input_hours": request.hours,

        "predicted_marks": round(
            float(prediction[0]),
            2
        )

    }