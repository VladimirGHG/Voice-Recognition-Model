# Voice-Recognition-Model

#  Voice Gender Classification Model

A machine learning project that classifies **male vs. female voices** using audio signal processing and feature extraction techniques.  
This project demonstrates an **end-to-end ML pipeline**: from raw audio recording → feature engineering → model training → real-time prediction.

---

## Features
- Records live audio via microphone (sounddevice, wavio).
- Extracts distinguishing acoustic features using **Librosa** and **SciPy**:
  - Pitch statistics (mean, median, standard deviation, range).
  - Spectral features (entropy, centroid, flatness).
  - Distribution-based features (skewness, kurtosis).
- Trains a **Decision Tree Classifier** (entropy criterion, max depth = 10) with scikit-learn.
- Saves and loads models using **Joblib** for reproducibility.
- Provides **real-time gender prediction** on recorded audio samples.

---

## Tech Stack
- **Python**  
- **Libraries**:  
  - scikit-learn → Decision Tree Classifier  
  - librosa → audio processing & feature extraction  
  - sounddevice & wavio → audio recording  
  - numpy, scipy → numerical features  
  - pandas → dataset handling  
  - joblib → model persistence  

---

## Project Structure
VOICEREC/

│── .venv/ # Virtual environment

│── decisionTree/

│ ├── pycache/

│ ├── modelUse.py # Code for running the model

│ ├── recording.py # Record audio & predict gender

│ ├── voice.csv # Dataset (features + labels)

│ ├── voiceExtractor.py # Feature extraction functions

│ ├── voiceRec.py # Model training and creating script

│── requirements.txt # Python dependencies

│── voice_gender_model.pkl # Model itself
