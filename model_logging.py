import pandas as pd
import json

import mlflow
from mlflow.models import infer_signature
from datetime import datetime

import xgboost as xgb
import sqlite3

def set_experiment(experiment):
    # Clean environment

    mlflow.set_tracking_uri("file:///app/lseungwo/Fever_AbxClassify/mlruns")

    # Create a new MLflow Experiment
    current_time =  datetime.now().replace(microsecond=0, second  = 0, minute = 0)
    formatted_time = current_time.strftime("%m-%d_%H")

    mlflow.set_experiment(f'{experiment}_{formatted_time}')

def log_model(model, params, aucpr, dtrain, model_number, label_name):
    with mlflow.start_run():
        # Log hyperparameters and metric
        mlflow.log_params(params)
        mlflow.log_metric("aucpr", aucpr)
        
        # try:
        #     # Get the raw features from DMatrix
        #     X_input = dtrain.get_data()
            
        #     # Convert to numpy array if sparse
        #     if hasattr(X_input, "toarray"):
        #         X_input = X_input.toarray()
            
        #     # For signature creation, we need to predict on raw features
        #     # Create a small sample of the data for the signature
        #     X_sample = X_input[:5]
            
        #     # For XGBoost, we need to create a new DMatrix for prediction
        #     sample_dmatrix = xgb.DMatrix(X_sample)
            
        #     # Get predictions on the sample
        #     output_example = model.predict(sample_dmatrix)
            
        #     # Create signature using the sample input and output
        #     signature = infer_signature(X_sample, output_example)
            
        #     # Input example should be the raw features, not the DMatrix
        #     input_example = X_sample
            
        # except Exception as e:
        #     print(f"⚠️ Signature inference failed: {e}")
        #     signature = None
        #     input_example = None

        # ✅ Log the model

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            # input_example=input_example,
            # signature=signature
        )

# def log_model(model, params, aucpr, dtrain, model_number, label_name):
#     # Start an MLflow run
#     with mlflow.start_run():
#         # Log the hyperparameters
#         mlflow.log_params(params)

#         # ✅ Ensure input example is serializable (convert to list)
#         input_example= dtrain.get_data().toarray()  # Convert to a list (if applicable)

#         # Log the loss metric
#         mlflow.log_metric("aucpr", aucpr)

#         # # Infer the model signature --> keeps failing.
#         # Error msg: 2025/03/18 19:23:35 WARNING mlflow.models.signature: Failed to infer schema for inputs. Setting schema to `Schema([ColSpec(type=AnyType())]` as default. Note that MLflow doesn't validate data types during inference for AnyType. To see the full traceback, set logging level to DEBUG.
#         # 2025/03/18 19:23:39 WARNING mlflow.models.model: Failed to validate serving input example {
#         signature = infer_signature(dtrain, model.predict(dtrain))
        
#         # Log the model
#         mlflow.sklearn.log_model(
#             sk_model=model,
#             artifact_path = "model",
#             signature=signature,
#             input_example=input_example,
#             registered_model_name=f"{label_name}_Model_{model_number}",
#         )


def store_all_cv_results(cv_results_all, db_path="cv_results.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    if isinstance(cv_results_all, dict):
        cv_results_all = [cv_results_all]
    
    # Create summary table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS cv_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            params TEXT,
            best_score REAL,
            best_iteration INTEGER
        )
    ''')

    # Create detailed metrics table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS cv_metrics (
            run_id INTEGER,
            iteration INTEGER,
            train_aucpr_mean REAL,
            train_aucpr_std REAL,
            test_aucpr_mean REAL,
            test_aucpr_std REAL,
            FOREIGN KEY(run_id) REFERENCES cv_runs(id)
        )
    ''')

    for result in cv_results_all:
        params = result["params"]
        best_score = float(result["best_score"])
        best_iteration = int(result["best_iteration"])
        cv_df = result["cv_results"].copy()
        cv_df.reset_index(drop=True, inplace=True)

        # Insert summary
        cursor.execute('''
            INSERT INTO cv_runs (params, best_score, best_iteration)
            VALUES (?, ?, ?)
        ''', (json.dumps(params), best_score, best_iteration))
        run_id = cursor.lastrowid

        # Insert CV metrics row by row
        for i, row in cv_df.iterrows():
            cursor.execute('''
                INSERT INTO cv_metrics (
                    run_id, iteration,
                    train_aucpr_mean, train_aucpr_std,
                    test_aucpr_mean, test_aucpr_std
                ) VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                run_id, i,
                float(row["train-aucpr-mean"]),
                float(row["train-aucpr-std"]),
                float(row["test-aucpr-mean"]),
                float(row["test-aucpr-std"])
            ))

    conn.commit()
    conn.close()
    # print(f"✅ Stored {len(cv_results_all)} runs to {db_path}")