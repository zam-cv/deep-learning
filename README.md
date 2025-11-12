# Predicción de Temperatura con LSTM - Weather Dataset

Proyecto de Deep Learning para predicción de temperatura usando redes neuronales recurrentes (LSTM) con TensorFlow/Keras.

## Descripción del Proyecto

Este proyecto implementa un modelo de aprendizaje profundo basado en arquitectura LSTM (Long Short-Term Memory) para predecir la temperatura 24 horas adelante, utilizando datos históricos de variables meteorológicas.

### Características Principales

- **Dataset:** Weather Dataset de Kaggle (~96,453 registros horarios)
- **Modelo:** LSTM multicapa con regularización
- **Objetivo:** Predicción de temperatura 24 horas adelante
- **Framework:** TensorFlow 2.x + Keras
- **Métricas:** MAE, RMSE, R², MAPE
- **Gestión de dependencias:** UV (moderno y rápido)

## Estructura del Proyecto

```
deep-learning/
├── weather_temperature_prediction.ipynb  # Notebook principal
├── download_dataset.py                   # Script para descargar dataset
├── predict.py                            # Script para predicciones
├── pyproject.toml                        # Dependencias (UV)
├── requirements.txt                      # Dependencias (pip legacy)
├── README.md                            # Este archivo
├── .gitignore                           
│
├── .venv/                               # Entorno virtual (generado por UV)
├── datasets/                            # Datos (se genera al descargar)
│   └── weatherHistory.csv              
├── models/                              # Modelos entrenados (se genera)
│   ├── temperature_lstm_model_final.keras
│   ├── scaler.pkl
│   └── metadata.json
└── results/                             # Visualizaciones (se genera)
```

## Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/zam-cv/deep-learning
cd deep-learning
```

### 2. Instalar dependencias con UV

Este proyecto usa [UV](https://github.com/astral-sh/uv) para gestión de dependencias (más rápido que pip).

```bash
# UV instalará automáticamente las dependencias y creará .venv/
uv sync
```

**Alternativa con pip tradicional:**
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configurar Kaggle API

El dataset se descarga desde Kaggle. Configura tus credenciales:

```bash
# 1. Ve a https://www.kaggle.com/settings/account
# 2. Sección "API" → "Create New Token"
# 3. Se descargará kaggle.json

# En Mac/Linux:
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# En Windows:
# Crea: C:\Users\<TuUsuario>\.kaggle\
# Mueve kaggle.json ahí
```

### 4. Descargar el Dataset

**Opción A: Script automático (RECOMENDADO)**

```bash
uv run python download_dataset.py
```

**Opción B: Desde el notebook**

El notebook descargará el dataset automáticamente si tienes Kaggle configurado.

**Opción C: Descarga manual**

1. Ve a: https://www.kaggle.com/datasets/muthuj7/weather-dataset
2. Click "Download" (requiere cuenta Kaggle gratis)
3. Descomprime y coloca `weatherHistory.csv` en `datasets/`

## Uso

### Ejecutar el Notebook (Entrenamiento)

```bash
# Con UV
uv run jupyter notebook weather_temperature_prediction.ipynb

# O activa el entorno manualmente
source .venv/bin/activate
jupyter notebook weather_temperature_prediction.ipynb
```

El notebook está organizado en:

1. **Introducción** - Planteamiento del problema
2. **Datos** - Carga, análisis y preprocesamiento
3. **Desarrollo** - Modelo baseline y LSTM V1
4. **Ajuste** - LSTM V2 mejorado con regularización
5. **Resultados** - Evaluación en conjunto de prueba
6. **Conclusiones** - Análisis y mejoras futuras
7. **Aplicación** - Función de predicción lista para usar

### Usar el Modelo Entrenado

Después de entrenar el modelo en el notebook:

```bash
uv run python predict.py
```

## Arquitectura del Modelo Final (LSTM V2)

```
Input: (168 horas, 13 características)
  ↓
LSTM(128) → Dropout(0.2) → BatchNormalization
  ↓
LSTM(64) → Dropout(0.2) → BatchNormalization
  ↓
LSTM(32) → Dropout(0.2)
  ↓
Dense(16, relu) → Dropout(0.1)
  ↓
Dense(1) → Temperatura predicha 24h adelante
```

**Mejoras implementadas:**
- 3 capas LSTM (vs 2 en V1)
- Dropout (20%) para regularización
- Batch Normalization para estabilidad
- 128→64→32 unidades (arquitectura descendente)
- Capa densa adicional antes de salida

## Resultados

| Métrica | Valor |
|---------|-------|
| MAE | X.XX°C |
| RMSE | X.XX°C |
| R² | 0.XXX |
| MAPE | X.XX% |

*Los valores exactos se obtienen al ejecutar el notebook*

## Dependencias Principales

- TensorFlow 2.20.0
- Pandas 2.3.3
- NumPy 2.3.4
- Scikit-learn 1.7.2
- Matplotlib 3.10.7
- Jupyter Notebook 7.4.7
- Kaggle 1.7.4 (para descarga de datos)

Ver `pyproject.toml` para la lista completa.

## Solución de Problemas

### Error 404 al descargar dataset

Si ves "HTTP Error 404: Not Found":

1. **Verifica credenciales de Kaggle:**
   ```bash
   cat ~/.kaggle/kaggle.json  # Debe existir
   ```

2. **Usa el script de descarga:**
   ```bash
   uv run python download_dataset.py
   ```

3. **Descarga manual:**
   - Ve a Kaggle (enlace arriba)
   - Descarga manualmente
   - Coloca en `datasets/`

### Problemas con UV

Si UV no está instalado:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Referencias

### Dataset
- [Weather Dataset - Kaggle](https://www.kaggle.com/datasets/muthuj7/weather-dataset)

### Papers
- Hochreiter & Schmidhuber (1997). Long Short-Term Memory
- Cho et al. (2014). Learning Phrase Representations using RNN Encoder-Decoder

### Documentación
- [TensorFlow Time Series](https://www.tensorflow.org/tutorials/structured_data/time_series)
- [Keras LSTM](https://keras.io/api/layers/recurrent_layers/lstm/)
- [UV Package Manager](https://docs.astral.sh/uv/)

## Autor

Alberto Velázquez

## Licencia

Proyecto académico - Uso educativo

---

**Última actualización:** Noviembre 2025
