import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

# 1) Carga del modelo .keras
model_path = 'trainedModel.keras'
model = load_model(model_path)

# 2) Lectura dinámica de clases (para que coincide con el entrenamiento)
classes = ['metal','glass','plastic','cardboard','paper','trash']

def preprocess_image(image):
    img_resized = cv2.resize(image, (224, 224))
    img_array  = np.array(img_resized) / 255.0
    return np.expand_dims(img_array, axis=0)

st.set_page_config(page_title="Detector3000", layout="wide")
st.title("🌿 Detector3000: Clasificador de Residuos")

uploaded = st.file_uploader("Escoge una imagen", type=["jpg","jpeg","png"])
if uploaded:
    # Leer bytes y decodificar imagen
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    frame      = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    rgb_frame  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Mostrar input
    st.image(rgb_frame, caption="Imagen cargada", use_container_width=True)

    # Inferencia
    input_array = preprocess_image(frame)
    preds = model.predict(input_array)
    idx = np.argmax(preds)
    label = classes[idx]
    confidence = float(preds[0][idx])

    # Resultado
    st.success(f"Residuo detectado: **{label}**")
    st.write(f"Nivel de confianza: {confidence:.2f}")
