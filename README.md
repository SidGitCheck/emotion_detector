# Emotion Detector

A real-time facial emotion recognition system using a Convolutional Neural Network (CNN) trained on the FER-2013 dataset.  
The model detects faces from a webcam feed and classifies them into seven basic emotions:

- Angry  
- Disgust  
- Fear  
- Happy  
- Neutral  
- Sad  
- Surprise  

This project is built using TensorFlow, OpenCV, and MediaPipe.

---

## Features

- Real-time webcam emotion detection  
- Face detection using MediaPipe  
- Works efficiently on CPU  
- Easy to set up and run locally  

---

## Project Structure

emotion_detector/
│
├─ main.py # Real-time detection application
├─ requirements.txt # Dependency list
│
├─ models/ # Model architecture and trained weights
│ ├─ best_cnn_fer2013.h5
│ ├─ load_model_h5.py
│ └─ model_builder.py
│
└─ assets/
└─ demo.mp4 # Demonstration video of the working project

yaml
Copy code

---

## Installation & Usage

### 1️⃣ Clone the repository
```bash
git clone https://github.com/SidGitCheck/emotion_detector.git
cd emotion_detector
2️⃣ Install the required libraries
bash
Copy code
pip install -r requirements.txt
3️⃣ Run the application
bash
Copy code
python main.py
Press Q to stop the webcam stream.

Model Details
Dataset: FER-2013 (Kaggle)

Input size: 48 × 48 × 3

Classes: 7

Framework: TensorFlow / Keras

The model was trained separately using a custom CNN architecture, and the trained weights are included in this repository.

Demo
A working example of the system is available here:

bash
Copy code
assets/demo.mp4
Open the video to see real-time emotion detection results.

Author
Siddhant Srivastava
Email: siddhant110806@gmail.com
GitHub: https://github.com/SidGitCheck

yaml
Copy code

---

Done ✔  
This is clean, fully Markdown, and copy-paste ready.

If you want, I can:

- Add Shields.io **badges** (Python version, TensorFlow, repo stars)
- Create a **banner/header image** for the top
- Add **future work** and **results section**
- Improve UI in the code (colored bounding boxes, probability display)

Would you like badges added next to make it look more polished?








