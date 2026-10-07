import os
import json
import re
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json


# Ruta principal del proyecto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Rutas del modelo y tokenizer
MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "modelo_sentimientos_lstm_es.keras"
)

TOKENIZER_PATH = os.path.join(
    BASE_DIR,
    "models",
    "tokenizer_es.json"
)

# Configuración
MAX_LEN = 200


# Cargar modelo
model = tf.keras.models.load_model(MODEL_PATH)


# Cargar tokenizer
with open(TOKENIZER_PATH, "r", encoding="utf-8") as archivo:
    tokenizer = tokenizer_from_json(archivo.read())


def limpiar_texto(texto):
    """
    Limpia el texto antes de enviarlo al modelo.
    """

    texto = str(texto).lower()

    texto = re.sub(
        r"http\S+|www\S+",
        " ",
        texto
    )

    texto = re.sub(
        r"[^a-záéíóúüñ0-9\s]",
        " ",
        texto
    )

    texto = re.sub(
        r"\s+",
        " ",
        texto
    ).strip()

    return texto


def predecir_sentimiento(texto):
    """
    Predice si un texto es positivo o negativo.
    """

    texto_limpio = limpiar_texto(texto)

    secuencia = tokenizer.texts_to_sequences(
        [texto_limpio]
    )

    secuencia_pad = pad_sequences(
        secuencia,
        maxlen=MAX_LEN,
        padding="post",
        truncating="post"
    )

    probabilidad = float(
        model.predict(
            secuencia_pad,
            verbose=0
        )[0][0]
    )

    if probabilidad >= 0.5:
        sentimiento = "Positivo"
        confianza = probabilidad
    else:
        sentimiento = "Negativo"
        confianza = 1 - probabilidad

    return sentimiento, confianza