import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

# 1) Carga del modelo .h5
model_path = 'model.h5'
model = load_model(model_path)

# 2) Leer la lista real de clases desde clases.npy
#    (esta matriz fue guardada en train.py con np.save("classes.npy", le.classes_))
classes = np.load("classes.npy")  # e.g. array(['cardboard','glass','metal','paper','plastic'], dtype='<U8')

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

    # Convertimos BGR→RGB para mostrar con st.image
    rgb_frame  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    st.image(rgb_frame, caption="Imagen cargada", use_container_width=True)

    # Inferencia: usamos frame (en BGR) directo para preprocess (no importa si es BGR o RGB
    # mientras que el canal esté ordenado consistentemente en train.py y aquí)
    input_array = preprocess_image(frame)
    preds = model.predict(input_array)
    idx = np.argmax(preds)                # índice de la clase con mayor probabilidad
    label = classes[idx]                  # etiqueta “real” desde classes.npy
    confidence = float(preds[0][idx])

    # Mostrar resultado
    st.success(f"Residuo detectado: **{label}**")
    st.write(f"Nivel de confianza: {confidence:.2f}")
