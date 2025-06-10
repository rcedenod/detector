# Detector3000

🌿 **Detector3000** es una aplicación de clasificación de residuos basada en visión por computador y aprendizaje profundo. Utiliza **Transfer Learning** con **MobileNetV2** para entrenar un modelo capaz de reconocer distintos tipos de basura (cartón, vidrio, metal, papel, plástico) y una interfaz web interactiva construida con **Streamlit**.

---

## 📂 Estructura del proyecto

```
├── app.py              # Interfaz web en Streamlit
├── train.py            # Script de entrenamiento del modelo
├── evaluate.py         # Script de evaluación y generación de métricas
├── model.h5            # Modelo Keras preentrenado (se genera tras train.py)
├── classes.npy         # Array de etiquetas (se genera tras train.py)
├── trashnet/           # Carpeta con subcarpetas por clase y sus imágenes
│   ├── cardboard/
│   ├── glass/
│   ├── metal/
│   ├── paper/
│   └── plastic/
├── report.txt          # Informe de clasificación (se genera tras evaluate.py)
└── matrix.csv          # Matriz de confusión en CSV (se genera tras evaluate.py)
```

---

## 🚀 Requisitos

* Python 3.7 o superior
* GPU opcional (acelera el entrenamiento y la inferencia)

Dependencias principales (crear un `requirements.txt` con estas líneas):

```
streamlit
opencv-python
numpy
tensorflow>=2.4
scikit-learn
pandas
```

Instálalas con:

```bash
pip install -r requirements.txt
```

---

## 💾 Preparar los datos

Coloca tu dataset en la carpeta `trashnet/`, organizado así:

```
trashnet/
├── cardboard/
│   ├── img1.jpg
│   ├── img2.png
│   └── …
├── glass/
│   └── …
├── metal/
│   └── …
├── paper/
│   └── …
└── plastic/
    └── …
```

Cada subcarpeta debe contener las imágenes de esa clase.

---

## 🏋️‍♂️ Entrenamiento (`train.py`)

1. **Carga de imágenes**
   Recorre `trashnet/`, redimensiona a 224×224 y almacena en arrays de NumPy.

2. **Preprocesamiento**
   Normaliza los píxeles (`images = images / 255.0`) y codifica etiquetas con `LabelEncoder`.

3. **Modelo**

   * Usa `MobileNetV2` preentrenado en ImageNet como **base** (capas congeladas).
   * Añade una capa `Flatten`, una capa densa de 128 unidades (ReLU) y salida softmax para `n` clases.

4. **Compilación y ajuste**

   ```python
   model.compile(
       optimizer='adam',
       loss='sparse_categorical_crossentropy',
       metrics=['accuracy']
   )
   model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
   ```

5. **Guardado**

   ```python
   model.save("model.h5")
   np.save("classes.npy", le.classes_)
   ```

**Ejecuta**:

```bash
python train.py
```

---

## 🧪 Evaluación (`evaluate.py`)

1. Carga `model.h5`.
2. Usa `ImageDataGenerator(rescale=1./255)` para leer todo `trashnet/` sin mezclar (`shuffle=False`).
3. Calcula predicciones y genera:

   * **Classification report** (precision, recall, F1) → `report.txt`
   * **Matriz de confusión** → `matrix.csv`

**Ejecuta**:

```bash
python evaluate.py
```

---

## 🌐 Interfaz web (`app.py`)

La app de Streamlit permite al usuario subir una imagen y ver:

1. **Preprocesamiento**:

   * Redimensiona a 224×224
   * Normaliza a rango \[0,1]

2. **Inferencia**:

   * Predice con el modelo cargado (`model.h5`)
   * Busca la etiqueta correspondiente en `classes.npy`

3. **Salida**:

   * Muestra la imagen
   * Indica la clase detectada y el nivel de confianza

**Ejecuta**:

```bash
streamlit run app.py
```

---

## ⚙️ Personalización

* Para ajustar **épocas**, **tamaño de batch** o **arquitectura**, edita `train.py`.
* Si tu dataset tiene más o menos clases, el modelado se adapta automáticamente al número de carpetas en `trashnet/`.
* Puedes mejorar la UI de Streamlit (títulos, estilos) modificando `app.py`.
