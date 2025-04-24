import mlflow
import mlflow.pyfunc
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from dotenv import load_dotenv
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

os.environ['AWS_ACCESS_KEY_ID'] = os.getenv('AWS_ACCESS_KEY_ID', 'ROOTUSER')
os.environ['AWS_SECRET_ACCESS_KEY'] = os.getenv('AWS_SECRET_ACCESS_KEY', 'CHANGEME123')
os.environ['MLFLOW_S3_ENDPOINT_URL'] = os.getenv('MLFLOW_S3_ENDPOINT_URL', 'http://localhost:9000')

mlflow.set_tracking_uri("http://localhost:5000")

client = mlflow.tracking.MlflowClient()

model_name = "IrisRFModel"

latest_version = client.get_latest_versions(model_name, stages=["None"])[0].version

model = mlflow.pyfunc.load_model(f"models:/{model_name}/{latest_version}")

app = FastAPI()

class PredictionRequest(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class PredictionResponse(BaseModel):
    prediction: int

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        data = pd.DataFrame([request.dict()])

        prediction = model.predict(data)[0]
        logger.info(f"Prediction made: {prediction} for input {request.dict()}")

        return PredictionResponse(prediction=prediction)

    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    logger.info("Root endpoint hit")
    return {"message": "ML Model is up and running!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
