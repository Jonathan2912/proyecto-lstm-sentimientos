# Clasificador de Sentimientos en Español con LSTM

Proyecto de Deep Learning desarrollado utilizando una red neuronal recurrente de tipo LSTM para clasificar reseñas en español como positivas o negativas

## Descripción

El proyecto utiliza el dataset `SetFit/amazon_reviews_multi_es`, compuesto por reseñas de productos escritas en español

El objetivo es desarrollar un modelo capaz de identificar automáticamente el sentimiento expresado en una reseña

El proyecto incluye

- Exploración del dataset
- Limpieza y preprocesamiento del texto
- Conversión de etiquetas de 5 clases a clasificación binaria
- Tokenización del texto
- Padding de las secuencias
- Construcción de una red neuronal LSTM
- Entrenamiento y validación
- Evaluación mediante métricas de clasificación
- Matriz de confusión
- Gráficas de entrenamiento
- Predicciones sobre nuevas reseñas
- Modelo entrenado guardado
- Aplicación web desarrollada con Streamlit

## Dataset

Se utilizo el dataset

`SetFit/amazon_reviews_multi_es`

El dataset contiene

- 200.000 registros de entrenamiento
- 5.000 registros de validación
- 5.000 registros de prueba
- 5 clases originales de sentimiento

Para este proyecto se realizó una clasificación binaria

| Etiqueta original | Sentimiento |
|---|---|
| 0 | Negativo |
| 1 | Negativo |
| 2 | Negativo |
| 3 | Positivo |
| 4 | Positivo |

Para el entrenamiento se utilizaron 100.000 reseñas balanceadas

- 50.000 negativas
- 50.000 positivas

Los conjuntos de validación y prueba se mantuvieron separados para evaluar correctamente el modelo

## Preprocesamiento

El texto fue procesado mediante

1. Conversión a minúsculas
2. Eliminación de URLs
3. Eliminación de caracteres especiales
4. Conservación de caracteres propios del español como `á`, `é`, `í`, `ó`, `ú`, `ü` y `ñ`
5. Eliminación de espacios adicionales
6. Tokenización
7. Padding y truncamiento a una longitud máxima de 200 tokens

Se utilizó un vocabulario máximo de 20.000 palabras y un token especial `<OOV>` para palabras desconocidas

## Arquitectura del modelo

La red neuronal utiliza la siguiente arquitectura:


Input
   ↓
Embedding
   ↓
SpatialDropout1D
   ↓
LSTM
   ↓
Dropout
   ↓
Dense + Sigmoid


Configuración principal:

- Vocabulary size: 20.000
- Embedding dimension: 64
- LSTM units: 64
- Dropout: 0.3
- Learning rate: 0.001
- Optimizer: Adam
- Loss: Binary Crossentropy
- Función de activación final: Sigmoid

El modelo utiliza `mask_zero=True` en la capa Embedding para ignorar los valores utilizados durante el padding de las secuencias

## Resultados

El modelo obtuvo una precisión de

**85% de accuracy en el conjunto de prueba.**

### Métricas de clasificación

| Clase | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Negativo | 0.8874 | 0.8590 | 0.8730 |
| Positivo | 0.7982 | 0.8365 | 0.8169 |
| Accuracy | | | **0.8500** |

### Matriz de confusión


[[2577  423]
 [ 327 1673]]


Interpretación:

- 2.577 reseñas negativas fueron clasificadas correctamente
- 423 reseñas negativas fueron clasificadas como positivas
- 327 reseñas positivas fueron clasificadas como negativas
- 1.673 reseñas positivas fueron clasificadas correctamente

## Predicciones de prueba

Se realizaron pruebas con 10 reseñas del conjunto de prueba

El modelo clasificó correctamente 9 de las 10 muestras seleccionadas, obteniendo un resultado de

**90% de aciertos en esta muestra de 10 predicciones.**

Esta prueba es únicamente una muestra adicional y no reemplaza la evaluación realizada sobre las 5.000 muestras del conjunto de prueba

## Aplicación Streamlit

El proyecto incluye una aplicación web desarrollada con Streamlit.

La aplicación permite

1. Introducir una reseña en español
2. Procesar automáticamente el texto
3. Clasificar el sentimiento
4. Mostrar la confianza de la predicción

Ejemplo:

Entrada:
El producto es excelente, me encantó y funciona perfectamente

Resultado:
Positivo
Confianza: 99.27%


Otro ejemplo es:


Entrada:
El producto es horrible, no funciona y estoy muy decepcionado

Resultado:
Negativo
Confianza: 99.78%


## Estructura del proyecto

proyecto-lstm-sentimientos/
│
├── Dataset/
│
├── models/
│   ├── modelo_sentimientos_lstm_es.keras
│   └── tokenizer_es.json
│
├── notebook/
│   └── sentimientos_lstm.ipynb
│
├── src/
│   ├── __init__.py
│   ├── predict.py
│   └── train.py
│
├── app/
│   └── app.py
│
├── results/
│
├── .gitignore
├── README.md
└── requirements.txt


## Instalación

Crear y activar un entorno virtual:


python -m venv .venv


Windows:


.venv\Scripts\activate


Instalar las dependencias:


pip install -r requirements.txt


## Ejecutar la aplicación

Desde la carpeta principal del proyecto:


streamlit run app/app.py


Después abrir en el navegador:


http://localhost:8501


## Tecnologías utilizadas

- Python
- TensorFlow
- Keras
- LSTM
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Hugging Face Datasets
- Google Colab
