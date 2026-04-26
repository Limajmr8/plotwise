import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models

print('TF ok')

# Rebuild exact architecture from train_disease_model.py
base = EfficientNetB0(weights=None, include_top=False, input_shape=(224, 224, 3))
model = models.Sequential([
    base,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(6, activation='softmax')
], name='PlotWise_DiseaseDetector')

model.build((None, 224, 224, 3))
model.load_weights('ml/saved_models/disease_model.h5', by_name=True)
print('Model ok')
