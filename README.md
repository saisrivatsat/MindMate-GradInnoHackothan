# 🧠 MindMate: AI-Powered Mental Health Detection and Support System

---

## 📑 Table of Contents
- [🧠 About the Project](#-about-the-project)
- [❓ Problem Statement](#-problem-statement)
- [📊 Dataset Overview](#-dataset-overview)
- [⚙️ Methodology](#️-methodology)
- [🚀 App Features](#-app-features)
- [🌟 Unique Innovations](#-unique-innovations)
- [📈 Results & Performance](#-results--performance)
- [⚠️ Limitations & Challenges](#️-limitations--challenges)
- [✅ Conclusion](#-conclusion)
- [🛠️ Tech Stack](#️-tech-stack)
- [📁 File Structure](#-file-structure)
- [📌 Setup Instructions](#-setup-instructions)

---

## 🧠 About the Project
**MindMate** is an AI-powered web application built to detect and respond to mental health conditions in real-time using **text and audio inputs**. It leverages machine learning (Logistic Regression) and natural language processing (NLP) techniques to predict mental health statuses like *Depression*, *Anxiety*, *Stress*, *Bipolar Disorder*, *Personality Disorders*, and *Suicidal Ideation* with an accuracy of **76%**.

---

## ❓ Problem Statement
Mental health care is often reactive, inaccessible, or stigmatized. Millions suffer in silence due to societal judgment or financial limitations. **MindMate** addresses these barriers by:
- Providing early intervention through real-time predictions
- Enabling proactive care via chatbot suggestions
- Sending alerts to emergency contacts for high-risk cases

---

## 📊 Dataset Overview
**Source:** [Kaggle - Sentiment Analysis for Mental Health](https://www.kaggle.com/datasets/engsaiedali/sentiment-analysis-for-mental-health)  
**Size:** 53,042 text records  
**Classes:**
- Normal (31%)
- Depression (29%)
- Suicidal (20%)
- Anxiety (7%)
- Stress (5%)
- Bipolar (5%)
- Personality Disorder (2%)

**Preprocessing includes:**
- Handling nulls  
- Tokenization, stopword removal, lemmatization  
- TF-IDF vectorization

---

## ⚙️ Methodology

### 3.1 Data Preprocessing
- Handled missing values (1%)
- Applied NLP techniques via **NLTK**
- Vectorized text using **TF-IDF**

### 3.2 Model Training
Tested 8 models; **Logistic Regression** was selected due to:
- Accuracy: **76%**
- Macro F1-score: **0.73**
- Best interpretability + consistent results across classes

### 3.3 App Development
Built using **Streamlit**, with integrations:
- `scikit-learn`: ML model
- `nltk`: NLP preprocessing & sentiment
- `speech_recognition`: Audio transcription
- `Altair`: Probability visualizations
- `Twilio`: WhatsApp alerts
- `YouTube API`: Video suggestions

---

## 🚀 App Features

| Feature | Description |
|--------|-------------|
| 🎤 Dual Input | Accepts **Text** & **Audio** input |
| 📊 Probability Visuals | Displays **class prediction confidence** |
| 🔔 Crisis Alerts | Sends **WhatsApp alert** if suicidal content is detected |
| 🤖 AI Chatbot | Personalized, empathetic chatbot responses |
| 🧘‍♀️ YouTube Integration | Recommends **guided meditations** |
| 💡 Confidence Labels | Risk level: Low / Moderate / High |
| 🧩 Sentiment Analysis | Adds tone/context to chatbot replies |
| 🖥️ Responsive UI | Clean layout with minimal scrolling |

---
![image](https://github.com/user-attachments/assets/412851ae-3c30-43de-bf31-6b9544347132)
![image](https://github.com/user-attachments/assets/bc55bf1f-1f07-4733-a9d1-45f2b7d32aec)
![image](https://github.com/user-attachments/assets/e60d936c-66ba-44a1-b292-c845300aef66)
![image](https://github.com/user-attachments/assets/f14ab0e4-6082-4323-8608-0879a50d2844)

## 🌟 Unique Innovations
- ✅ Dual Input Modes (Text + Audio)
- 📲 WhatsApp-based Crisis Notification System
- 🔍 Transparent Confidence Levels for User Trust
- 🎯 Tailored Suggestions Based on Condition
- 📉 Visual Probabilities for Transparency
- 🗣️ Sentiment-Aware Chatbot Responses

---

## 📈 Results & Performance
| Metric | Value |
|--------|-------|
| Accuracy | **76%** |
| F1-Score (Normal) | **0.89** |
| F1-Score (Anxiety) | **0.81** |
| F1-Score (Stress) | **0.59** |
| Crisis Alert | Working via **Twilio API** |
| Audio Input | 80% accuracy with clean input; fails with noise |

---

## ⚠️ Limitations & Challenges
- 📡 **Audio Dependency**: Google Speech API requires internet and is noise-sensitive  
- 📉 **Class Imbalance**: Poor F1-score for minority classes like *Stress*  
- 🚨 **False Positives in Alerts**: May trigger alerts at low thresholds  
- ♿ **Limited Accessibility**: No TTS or multi-language support (yet)

**Planned Fixes:**
- Switch to **Whisper** for offline audio transcription  
- Add **class weighting / oversampling**  
- Increase alert threshold to ≥ 0.85  
- Integrate **text-to-speech & multilingual support**

---

## ✅ Conclusion
MindMate is a scalable, AI-driven solution to one of the most pressing issues of our time: mental health. It delivers **real-time predictions**, **empathetic support**, and **life-saving alerts** with a professional UI and high accuracy.

Future plans include:
- Better minority class detection
- Offline capabilities
- Full accessibility integration

MindMate aims to **normalize mental health conversations**, reduce stigma, and empower individuals to take control of their emotional well-being.

---

## 🛠️ Tech Stack
- Python 3.10+
- Streamlit
- Scikit-learn
- NLTK
- SpeechRecognition
- Twilio API
- YouTube Data API
- Altair

---

## 📁 File Structure
```bash
MindMate/
│
├── app.py                  # Main Streamlit app
├── train_model.py          # ML training script
├── LogisticRegression_model.pkl  # Trained ML model
├── utils.py                # Helper functions (text/audio processing)
├── requirements.txt        # Python dependencies
├── .env                    # API keys (Twilio, YouTube)
├── assets/                 # Icons, images
└── README.md               # Project documentation
