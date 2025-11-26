
import cv2
import numpy as np

class Preprocessor:
    def __init__(self, target_size=(48, 48)):
        self.target_size = target_size

    def preprocess_face(self, frame, bbox):
        """
        Crop, grayscale, resize, normalize a face from the frame.
        
        Parameters:
            frame : np.array
                Original BGR frame
            bbox : tuple
                (x_min, y_min, x_max, y_max) bounding box coordinates
        
        Returns:
            face_img : np.array
                Preprocessed face ready for CNN
        """
        x_min, y_min, x_max, y_max = bbox

        # Crop face
        face = frame[y_min:y_max, x_min:x_max]

        # If bbox is invalid (height/width 0), return None
        if face.size == 0:
            return None

        # Convert to grayscale
        face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # Resize to target size
        face_resized = cv2.resize(face_gray, self.target_size)

        # Normalize to [0,1]
        face_normalized = face_resized.astype('float32') / 255.0

        # Expand dims to match CNN input shape (48,48,1)
        face_input = np.expand_dims(face_normalized, axis=-1)

        return face_input
