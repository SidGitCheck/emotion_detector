
# main.py

import cv2
import numpy as np
import mediapipe as mp
from models.load_model_h5 import load_emotion_model # Using our safe loader

# --- 1. Load Model and Initialize ---
MODEL_PATH = 'models/best_cnn_fer2013.h5'
model, class_labels = load_emotion_model(MODEL_PATH)

if model is None:
    print("❌ Critical Error: Model could not be loaded. Exiting.")
    exit()

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Cannot open webcam.")
    exit()

print("✅ Starting real-time emotion detection... Press 'q' to quit.")

# --- 2. Main Application Loop ---
while True:
    success, frame = cap.read()
    if not success:
        print("⚠️ Failed to capture frame. Exiting...")
        break

    frame_height, frame_width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)

    # --- 3. Process Detections ---
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            bbox = int(bboxC.xmin * frame_width), int(bboxC.ymin * frame_height), \
                   int(bboxC.width * frame_width), int(bboxC.height * frame_height)
            
            x, y, w, h = bbox
            
            if w <= 0 or h <= 0 or x < 0 or y < 0:
                continue

            # Crop the face region from the original BGR frame
            face_roi = frame[y:y+h, x:x+w]
            
            if face_roi.size == 0:
                continue

            # --- 4. Preprocess the Face for the Model (THE FIX) ---
            # 1. Resize the COLOR face to the model's expected input size (48x48)
            #    We are NO LONGER converting to grayscale.
            resized_face = cv2.resize(face_roi, (48, 48), interpolation=cv2.INTER_AREA)
            
            # 2. Normalize pixel values to the range [0, 1]
            normalized_face = resized_face / 255.0
            
            # 3. Expand dimensions to match model input shape: (1, 48, 48, 3)
            input_face = np.expand_dims(normalized_face, axis=0)
            
            # --- 5. Predict Emotion ---
            try:
                # The model expects a batch of images, so we pass the single face
                prediction = model.predict(input_face)
                predicted_emotion = class_labels[np.argmax(prediction)]
            except Exception as e:
                print(f"Prediction error: {e}")
                predicted_emotion = "Error"
            
            # --- 6. Display Output on Frame ---
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, predicted_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the final frame
    cv2.imshow("Real-Time Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 7. Cleanup ---
print("Releasing resources...")
cap.release()
cv2.destroyAllWindows()