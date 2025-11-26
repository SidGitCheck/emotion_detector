
# models/load_model_h5.py

import os
# Force TensorFlow to use only the CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Import our new model-building function
from models.model_builder import build_emotion_cnn

def load_emotion_model(weights_path='models/best_cnn_fer2013.h5'):
    """
    Loads the emotion detection model by rebuilding its architecture
    and then loading the pre-trained weights. This avoids version
    incompatibility issues with legacy .h5 files.

    Args:
        weights_path (str): The path to the .h5 weights file.

    Returns:
        tuple: A tuple containing the loaded Keras model and class labels.
    """
    try:
        # 1. Build a fresh, compatible model architecture
        model = build_emotion_cnn()
        
        # 2. Load only the learned weights into this new model
        model.load_weights(weights_path)
        
        print("✅ Model architecture built and weights loaded successfully.")
    except Exception as e:
        print(f"❌ Error building model or loading weights from {weights_path}: {e}")
        return None, None

    # Define the class labels
    class_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    
    return model, class_labels