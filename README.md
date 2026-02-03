# speech-emotion-recognition
Speech Emotion Recognition | CNN, LSTM, SVM | RAVDESS Dataset
Speech Emotion Recognition (SER)
📌 Project Overview

Hi Thats my final year project, this project implements a Speech Emotion Recognition system that classifies human emotions from speech audio using machine learning and deep learning techniques.
The system is trained on the RAVDESS dataset and uses CNN-LSTM and SVM models for emotion classification.

🎯 Features

Emotion classification from speech audio

Audio preprocessing and augmentation

MFCC feature extraction

CNN-LSTM deep learning model

Confusion matrix and accuracy evaluation

🧠 Emotions Recognized

Neutral

Calm

Happy

Sad

Angry

Fearful

Disgust

Surprised

📂 Project Structure
SER_CODE/
│
├── DATASET/               # RAVDESS speech emotion dataset
├── emotion_model.h5       # Trained CNN-LSTM model
├── src.ipynb              # Training, evaluation & visualization code
├── README.txt             # Project documentation
📊 Dataset

RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)

Speech-only audio samples

Multiple actors and emotional classes

Used for training and testing the SER model

⚙️ Methodology
1. Audio Preprocessing

Fixed audio duration and offset

Normalization and trimming

2. Data Augmentation

Noise addition

Pitch shifting

Time shifting

Time stretching

3. Feature Extraction

MFCC (Mel-Frequency Cepstral Coefficients)

Extracted using Librosa

🏗️ Model Architecture
CNN-LSTM Model

TimeDistributed Conv1D layers

Batch Normalization

LSTM layer for temporal learning

Dense layers with Dropout & L2 regularization

Softmax output layer

SVM

Used as a traditional machine learning baseline

🛠️ Technologies Used

Python

TensorFlow / Keras

Librosa

NumPy, Pandas

Scikit-learn

Matplotlib, Seaborn

📈 Training & Evaluation

Optimizer: Adam

Loss Function: Categorical Cross-Entropy

Metrics: Accuracy

Evaluation using confusion matrix and learning curves

🚀 How to Run the Project

Install required libraries:

pip install librosa tensorflow numpy pandas scikit-learn matplotlib seaborn

Place the RAVDESS dataset inside the DATASET folder

Open and run src.ipynb

Load the trained model from emotion_model.h5 for testing

📌 Output

Emotion prediction results

Accuracy score

Confusion matrix visualization

🔮 Future Scope

Real-time emotion detection using microphone input

Web or mobile deployment

Transformer-based models

Cross-dataset evaluation
