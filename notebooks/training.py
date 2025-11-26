
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

csv_path = "/kaggle/input/fer2013/fer2013.csv"
data = pd.read_csv(csv_path)

print("Data shape:", data.shape)
print(data['emotion'].value_counts())

pixels = data['pixels'].tolist()
width, height = 48, 48
faces = []

for pixel_seq in pixels:
    face = np.array([int(p) for p in pixel_seq.split(' ')]).reshape(width, height)
    faces.append(face.astype('uint8'))

faces = np.array(faces)
emotions = pd.get_dummies(data['emotion']).values

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
faces_clahe = np.array([clahe.apply(face) for face in faces])

faces_clahe = faces_clahe.astype('float32') / 255.0
faces_clahe = np.expand_dims(faces_clahe, -1)

X_train, X_val, y_train, y_val = train_test_split(
    faces_clahe, emotions, test_size=0.1, stratify=emotions, random_state=42
)

print("Train:", X_train.shape, "Validation:", X_val.shape)

y_train_labels = np.argmax(y_train, axis=1)
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_labels),
    y=y_train_labels
)
class_weights = dict(enumerate(class_weights))
print("Class weights:", class_weights)

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest'
)
datagen.fit(X_train)

num_classes = 7
img_size = 48

model = Sequential([
    Conv2D(64, (3,3), activation='relu', padding='same', input_shape=(img_size, img_size, 1)),
    BatchNormalization(),
    Conv2D(64, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(0.25),

    Conv2D(128, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(128, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(0.3),

    Conv2D(256, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(256, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(0.4),

    Flatten(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

checkpoint = ModelCheckpoint(
    'best_scratch_cnn_fer2013.h5',
    monitor='val_accuracy',
    verbose=1,
    save_best_only=True,
    mode='max'
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    verbose=1,
    min_lr=1e-6
)

early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=8,
    verbose=1,
    restore_best_weights=True
)

epochs = 40
batch_size = 64

history = model.fit(
    datagen.flow(X_train, y_train, batch_size=batch_size),
    validation_data=(X_val, y_val),
    epochs=epochs,
    class_weight=class_weights,
    callbacks=[checkpoint, reduce_lr, early_stop]
)

val_loss, val_acc = model.evaluate(X_val, y_val)
print(f"\n✅ Validation Accuracy: {val_acc*100:.2f}%")

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy Curve')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss Curve')
plt.legend()
plt.show()

best_val_acc = int(val_acc * 100)
model_filename = f"scratch_cnn_fer2013_acc_{best_val_acc}.h5"
model.save(model_filename)
print(f"Model saved as {model_filename}")