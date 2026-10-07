import sys
import os

# Agregar la carpeta principal del proyecto al PATH
sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from src.predict import predecir_sentimiento

import streamlit as st


# Configuración de la página
st.set_page_config(
    page_title="Clasificador de Sentimientos",
    page_icon="💬",
    layout="centered"
)


# Título
st.title("💬 Clasificador de Sentimientos")

st.write(
    "Escribe una reseña o comentario en español y "
    "el modelo LSTM determinará si el sentimiento "
    "es positivo o negativo."
)


# Entrada de texto
reseña = st.text_area(
    "Escribe aquí tu reseña:",
    placeholder=(
        "Ejemplo: El producto me encantó, "
        "es excelente y funciona muy bien."
    ),
    height=150
)


# Botón
if st.button("🔍 Analizar sentimiento"):

    if reseña.strip() == "":
        st.warning(
            "⚠️ Por favor, escribe un texto antes de analizar."
        )

    else:

        sentimiento, confianza = predecir_sentimiento(
            reseña
        )

        st.subheader("📊 Resultado")

        st.write(
            f"**Sentimiento:** {sentimiento}"
        )

        st.write(
            f"**Confianza:** {confianza * 100:.2f}%"
        )

        if sentimiento == "Positivo":

            st.success(
                "😊 La reseña tiene un sentimiento POSITIVO."
            )

        else:

            st.error(
                "😞 La reseña tiene un sentimiento NEGATIVO."
            )