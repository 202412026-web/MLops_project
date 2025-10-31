import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import mlflow
import mlflow.sklearn
import os
from sklearn.impute import SimpleImputer

def train_model():
    # Cargar datos procesados
    data = pd.read_csv("mlops_project/data/processed/train.csv")

    # Asegurar que la columna target exista
    target_col = "target"  # cambia este nombre según tu dataset real

    # Filtrar columnas numéricas
    X = data.select_dtypes(include=["number"]).copy()
    if target_col in data.columns:
        y = data[target_col]
    else:
        raise ValueError(f"❌ La columna '{target_col}' no se encuentra en el dataset")

# Imputar valores faltantes
    imputer = SimpleImputer(strategy="mean")
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)


    # Entrenar modelo
    model = LogisticRegression(max_iter=200)
    model.fit(X, y)

    # Guardar modelo
    os.makedirs("models", exist_ok=True)
    import joblib
    joblib.dump(model, "models/model.pkl")

    # Registrar en MLflow
    mlflow.set_experiment("mlops_demo")
    with mlflow.start_run(run_name="logistic_regression"):
        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_metric("train_rows", len(X))
        mlflow.sklearn.log_model(model, "model")

    print("✅ Modelo entrenado y registrado con MLflow.")

if __name__ == "__main__":
    train_model()
