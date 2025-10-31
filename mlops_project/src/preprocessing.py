import pandas as pd
from sklearn.model_selection import train_test_split
import os

def load_and_split_data(path: str):
    df = pd.read_csv(path)
    train, test = train_test_split(df, test_size=0.2, random_state=42)

    # Crear la carpeta si no existe
    os.makedirs("mlops_project/data/processed", exist_ok=True)

    # Guardar los archivos donde DVC los espera
    train.to_csv("mlops_project/data/processed/train.csv", index=False)
    test.to_csv("mlops_project/data/processed/test.csv", index=False)
    print("✅ Data preprocessed and saved.")
    

if __name__ == "__main__":
    load_and_split_data("mlops_project/data/raw/Data_CU_venta.csv")
