import os
import re

import numpy as np
import tensorflow as tf
from datasets import load_dataset
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


# =========================
# CONFIGURACIÓN
# =========================

VOCAB_SIZE = 20000
MAX_LEN = 200
EMBEDDING_DIM = 64
LSTM_UNITS = 64

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(
    MODEL_DIR,
    "modelo_sentimientos_lstm_es.keras"
)

TOKENIZER_PATH = os.path.join(
    MODEL_DIR,
    "tokenizer_es.json"
)

os.makedirs(MODEL_DIR, exist_ok=True)


# =========================
# LIMPIEZA DEL TEXTO
# =========================

def limpiar_texto(texto):
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


# =========================
# CARGAR DATASET
# =========================

print("Cargando dataset...")

dataset = load_dataset(
    "SetFit/amazon_reviews_multi_es"
)


# Convertir los datos a NumPy/Pandas
train_data = dataset["train"]

texts = np.array(train_data["text"])
labels_originales = np.array(train_data["label"])


# =========================
# CONVERSIÓN A BINARIO
# =========================

# 0, 1 y 2 = Negativo
# 3 y 4 = Positivo

labels = np.where(
    labels_originales >= 3,
    1,
    0
)


# =========================
# BALANCEAR DATASET
# =========================

indices_negativos = np.where(labels == 0)[0]
indices_positivos = np.where(labels == 1)[0]

cantidad = min(
    len(indices_negativos),
    len(indices_positivos),
    50000
)

rng = np.random.default_rng(42)

indices_negativos = rng.choice(
    indices_negativos,
    size=cantidad,
    replace=False
)

indices_positivos = rng.choice(
    indices_positivos,
    size=cantidad,
    replace=False
)

indices = np.concatenate(
    [
        indices_negativos,
        indices_positivos
    ]
)

rng.shuffle(indices)

texts = texts[indices]
labels = labels[indices]


print(
    f"Cantidad de muestras de entrenamiento: {len(texts)}"
)


# =========================
# LIMPIAR TEXTOS
# =========================

print("Limpiando textos...")

texts_clean = [
    limpiar_texto(texto)
    for texto in texts
]


# =========================
# TOKENIZACIÓN
# =========================

print("Creando tokenizer...")

tokenizer = Tokenizer(
    num_words=VOCAB_SIZE,
    oov_token="<OOV>"
)

tokenizer.fit_on_texts(texts_clean)

sequences = tokenizer.texts_to_sequences(
    texts_clean
)

X_train = pad_sequences(
    sequences,
    maxlen=MAX_LEN,
    padding="post",
    truncating="post"
)

y_train = labels


# =========================
# MODELO LSTM
# =========================

print("Construyendo modelo LSTM...")

model = tf.keras.Sequential([
    layers.Input(
        shape=(MAX_LEN,)
    ),

    layers.Embedding(
        input_dim=VOCAB_SIZE,
        output_dim=EMBEDDING_DIM,
        mask_zero=True
    ),

    layers.SpatialDropout1D(0.2),

    layers.LSTM(
        LSTM_UNITS,
        dropout=0.2,
        recurrent_dropout=0.2
    ),

    layers.Dropout(0.3),

    layers.Dense(
        1,
        activation="sigmoid"
    )
])


# =========================
# COMPILACIÓN
# =========================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


model.summary()


# =========================
# ENTRENAMIENTO
# =========================

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="loss",
    patience=2,
    restore_best_weights=True
)

print("Iniciando entrenamiento...")

model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=128,
    callbacks=[early_stopping],
    verbose=1
)


# =========================
# GUARDAR MODELO
# =========================

model.save(
    MODEL_PATH
)

print(
    f"Modelo guardado en: {MODEL_PATH}"
)


# =========================
# GUARDAR TOKENIZER
# =========================

with open(
    TOKENIZER_PATH,
    "w",
    encoding="utf-8"
) as archivo:

    archivo.write(
        tokenizer.to_json()
    )


print(
    f"Tokenizer guardado en: {TOKENIZER_PATH}"
)

print("Entrenamiento finalizado correctamente.")