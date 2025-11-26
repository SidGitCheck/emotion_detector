# 🎭 Emotion Detector — Real-Time Facial Emotion Recognition

A deep-learning-based application that predicts human emotions from a **live webcam feed** using a Convolutional Neural Network (CNN) trained on the **FER-2013 dataset**.

The model recognizes **7 basic human emotions**:
<br>

😡 Angry &nbsp;&nbsp; 🤢 Disgust &nbsp;&nbsp; 😨 Fear &nbsp;&nbsp; 😀 Happy  
😢 Sad &nbsp;&nbsp; 😮 Surprise &nbsp;&nbsp; 😐 Neutral

Built using **TensorFlow**, **OpenCV**, and **MediaPipe** — optimized for smooth CPU performance.

---

## 🧠 What the App Does

✔ Detects faces in real-time  
✔ Classifies facial expressions into emotions  
✔ Displays the predicted emotion above each face  
✔ Works on standard webcams  
✔ CPU-friendly & fast 🚀

---

## 📂 Project Structure

emotion_detector/
│
├─ main.py # Real-time webcam detection app
├─ requirements.txt # Dependencies
├─ README.md # Project documentation
│
├─ models/ # Model architecture & weights
│ ├─ best_cnn_fer2013.h5
│ ├─ load_model_h5.py
│ └─ model_builder.py
│
└─ assets/
└─ demo.mp4 # Demo video of the working system

yaml
Copy code

---

## ⚙️ Setup & Run

### 1️⃣ Clone the repository
```bash
git clone https://github.com/SidGitCheck/emotion_detector.git
cd emotion_detector
2️⃣ Install required libraries
bash
Copy code
pip install -r requirements.txt
3️⃣ Start the emotion detector
bash
Copy code
python main.py
Press Q to exit the webcam stream.

🧩 Model Information
Dataset: FER-2013 (Kaggle)

Model: Custom CNN (48×48 input resolution)

Classes: 7 emotions

Training Framework: TensorFlow/Keras

Designed specifically for real-time edge performance ⚡

🎥 Demo Video
Check out how the model performs 🡻

👉 assets/demo.mp4
(Plays inside GitHub on supported devices)

🚀 Future Enhancements
📌 High confidence emotion overlay
📌 Multi-face emotion support at once
📌 Improve model accuracy with transfer learning
📌 Deploy as a web/desktop app
📌 Add dataset & model training logs

👤 Author
Siddhant Srivastava
📧 Email: siddhant110806@gmail.com
🔗 GitHub: SidGitCheck

