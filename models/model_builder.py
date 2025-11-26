
# models/model_builder.py

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization

def build_emotion_cnn(input_shape=(48, 48, 3), num_classes=7):
    """
    Builds a deep, Keras-compatible CNN model architecture that matches
    the saved weights from the user's trained model.
    """
    model = Sequential()

    # --- Block 1 ---
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # --- Block 2 ---
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # --- Block 3 ---
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # --- Flattening and Dense Layers ---
    model.add(Flatten())
    
    # 🔧 THE FIX: Changed 256 to 512 to match the saved weights
    model.add(Dense(512, activation='relu'))
    
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    
    # --- Output Layer ---
    model.add(Dense(num_classes, activation='softmax'))

    return model