import mlflow
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from mlflow.models.signature import infer_signature
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()

os.environ['AWS_ACCESS_KEY_ID'] = os.getenv('AWS_ACCESS_KEY_ID', 'ROOTUSER')
os.environ['AWS_SECRET_ACCESS_KEY'] = os.getenv('AWS_SECRET_ACCESS_KEY', 'CHANGEME123')
os.environ['MLFLOW_S3_ENDPOINT_URL'] = os.getenv('MLFLOW_S3_ENDPOINT_URL', 'http://localhost:9000')

mlflow.set_tracking_uri("http://localhost:5000") 
mlflow.set_experiment("iris_experiment")  

print("Tracking URI:", mlflow.get_tracking_uri())
print("Artifact URI:", mlflow.get_artifact_uri())

def train_and_log_model():
    
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    input_example = pd.DataFrame(X_train[:1], columns=iris.feature_names)

    signature = infer_signature(X_train, model.predict(X_train))

    if mlflow.active_run():
        mlflow.end_run()

        with mlflow.start_run():
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="rf_model",
                registered_model_name="IrisRFModel",
                input_example=input_example,
                signature=signature 
            )
            mlflow.log_param("n_estimators", 100)
            mlflow.log_metric("accuracy", model.score(X_test, y_test))

        print("The model is trained and logged in MLflow (artifacts in MinIO).")

if __name__ == "__main__":
    train_and_log_model()

