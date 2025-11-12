#!/usr/bin/env python3
"""
Script de predicción de temperatura usando modelo LSTM entrenado.

Uso:
    python predict.py
"""

import numpy as np
import pandas as pd
import pickle
import json
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    from tensorflow import keras
except ImportError:
    print("Error: TensorFlow no está instalado.")
    print("Instala: pip install -r requirements.txt")
    exit(1)

# Rutas
MODEL_PATH = 'models/temperature_lstm_model_final.keras'
SCALER_PATH = 'models/scaler.pkl'
METADATA_PATH = 'models/metadata.json'

def load_production_model(model_path=MODEL_PATH, scaler_path=SCALER_PATH, metadata_path=METADATA_PATH):
    """Carga modelo, scaler y metadata."""
    # Verificar archivos
    for path, name in [(model_path, "Modelo"), (scaler_path, "Scaler"), (metadata_path, "Metadata")]:
        if not Path(path).exists():
            raise FileNotFoundError(
                f"{name} no encontrado en: {path}\n"
                "Primero ejecuta el notebook para entrenar el modelo."
            )
    
    # Cargar
    model = keras.models.load_model(model_path)
    print(f"✓ Modelo cargado: {model_path}")
    
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    print(f"✓ Scaler cargado: {scaler_path}")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    print(f"✓ Metadata cargada: {metadata_path}")
    
    # Info
    print(f"\nMODELO:")
    print(f"  Versión: {metadata.get('model_version', 'N/A')}")
    print(f"  MAE (test): {metadata.get('test_mae', 0):.3f}°C")
    print(f"  R² (test): {metadata.get('test_r2', 0):.4f}")
    
    return model, scaler, metadata

def main():
    print("\nSISTEMA DE PREDICCIÓN DE TEMPERATURA - LSTM\n")
    
    if not Path(MODEL_PATH).exists():
        print("❌ Modelo no encontrado.")
        print("\nPasos:")
        print("1. Ejecuta: jupyter notebook weather_temperature_prediction.ipynb")
        print("2. Entrena el modelo (ejecutar todas las celdas)")
        print("3. Vuelve a ejecutar: python predict.py")
        return
    
    try:
        model, scaler, metadata = load_production_model()
        print("\n✓ Sistema listo para predicciones")
        print("\nPara usar con tus datos:")
        print("  from predict import load_production_model")
        print("  # Consulta el notebook para ejemplos completos")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()
