import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

MODEL_PATH = 'model.h5'
DATA_DIR = 'trashnet' # Ruta del dataset
BATCH_SIZE = 32
IMG_SIZE = (224, 224)

model = load_model(MODEL_PATH)

datagen = ImageDataGenerator(rescale=1./255)
val_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size = IMG_SIZE,
    batch_size = BATCH_SIZE,
    class_mode = 'categorical',
    shuffle = False
)

y_true = val_gen.classes
labels = list(val_gen.class_indices.keys())
y_pred_probs = model.predict(val_gen, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)
report = classification_report(y_true, y_pred, target_names=labels)
cm = confusion_matrix(y_true, y_pred)

print("Classification report \n", report)
print("Matrix \n", cm)

with open('report.txt', 'w') as f:
    f.write(report)

df_cm = pd.DataFrame(cm, index=labels, columns=labels)
df_cm.to_csv('matrix.csv')
