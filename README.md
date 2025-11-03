# 🎭 Emotion Detector
A deep learning–based **facial emotion recognition** project that identifies human emotions from facial expressions using **Convolutional Neural Networks (CNN)**. Trained on the **FER-2013 dataset**, this model can recognize seven basic emotions from images or real-time webcam feeds.

---

## 🧠 Overview
This project detects emotions such as: 😡 **Angry**, 🤢 **Disgust**, 😨 **Fear**, 😀 **Happy**, 😢 **Sad**, 😮 **Surprise**, and 😐 **Neutral**.
It uses CNN-based deep learning techniques to analyze facial features and classify them into emotion categories.

---

## 📁 Project Structure
```
emotion_detector/
│
├── models/          # Trained model (.h5) and training history (.pkl)
├── modules/         # Custom preprocessing or helper scripts
├── notebooks/       # Jupyter notebooks for training/testing
├── assets/          # Sample images or visualizations
│
├── app.py           # Application file (Streamlit or Flask)
├── requirements.txt # List of dependencies
├── README.md        # Project documentation
└── .gitignore
```

---

## ⚙️ Installation & Usage

### 1️⃣ Clone this repository
```
git clone https://github.com/SidGitCheck/emotion_detector.git
cd emotion_detector
```

### 2️⃣ Install dependencies
```
pip install -r requirements.txt
```

### 3️⃣ Run the application
If using Streamlit:
```
streamlit run app.py
```
Or if using Flask:
```
python app.py
```

---

## 🧩 Model Info
- **Dataset:** FER-2013 (from Kaggle)
- **Model:** Custom CNN architecture
- **Optimizer:** Adam
- **Loss:** Categorical Crossentropy
- **Validation Accuracy:** ~63% (example)

The trained model and training history are available in the `models/` folder:
```
models/
├── model.h5
└── history.pkl
```

---

## 🚀 Future Enhancements
- Add live webcam emotion detection
- Improve model accuracy using transfer learning
- Deploy as a web app (Heroku / Hugging Face)

---

## 👤 Author
**Siddhant Srivastava**
- GitHub: [SidGitCheck](https://github.com/SidGitCheck)
- Email: [siddhant110806@gmail.com](mailto:siddhant110806@gmail.com)

---

## 📄 License
This project is licensed under the **MIT License**.  
If you found this project helpful, consider ⭐ giving it a star on GitHub!

