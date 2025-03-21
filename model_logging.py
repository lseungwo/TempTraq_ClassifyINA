import mlflow
from mlflow.models import infer_signature
from datetime import datetime

def set_experiment(experiment):
    mlflow.set_tracking_uri(uri='http://127.0.0.1:5000')

    # Create a new MLflow Experiment
    current_time =  datetime.now().replace(microsecond=0)

    mlflow.set_experiment(f'{experiment}_{current_time}')

def log_model(model, params, aucpr, dtrain, model_number):
    # Start an MLflow run
    with mlflow.start_run():
        # Log the hyperparameters
        mlflow.log_params(params)

        # ✅ Ensure input example is serializable (convert to list)
        input_example= dtrain.get_data().toarray()  # Convert to a list (if applicable)

        # Log the loss metric
        mlflow.log_metric("aucpr", aucpr)

        # # Infer the model signature --> keeps failing.
        # Error msg: 2025/03/18 19:23:35 WARNING mlflow.models.signature: Failed to infer schema for inputs. Setting schema to `Schema([ColSpec(type=AnyType())]` as default. Note that MLflow doesn't validate data types during inference for AnyType. To see the full traceback, set logging level to DEBUG.
        # 2025/03/18 19:23:39 WARNING mlflow.models.model: Failed to validate serving input example {
        # signature = infer_signature(dtrain, model.predict(dtrain))

        # Log the model
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path = "model",
            # signature=signature,
            input_example=input_example,
            registered_model_name=f"Model_{model_number}",
        )