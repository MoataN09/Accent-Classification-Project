
import streamlit as st
import requests
import os
import numpy as np
import librosa
from moviepy.editor import VideoFileClip
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import tempfile
import glob

# ---------------------------
# Helper Functions
# ---------------------------
def download_video(url, filename='video.mp4'):
    response = requests.get(url, stream=True)
    with open(filename, 'wb') as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
    return filename

def extract_audio(video_path, audio_path='audio.wav'):
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path)
    return audio_path

def extract_mfcc_features(audio_path):
    y_audio, sr = librosa.load(audio_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    return mfcc_mean

def load_model():
    return joblib.load('accent_model.pkl')

def predict_accent(audio_path, model):
    features = extract_mfcc_features(audio_path)
    probs = model.predict_proba([features])[0]
    class_index = np.argmax(probs)
    class_label = model.classes_[class_index]
    confidence = probs[class_index] * 100
    explanation = f"Detected accent patterns similar to {class_label} English speakers."
    return class_label, confidence, explanation

# ---------------------------
# Streamlit App
# ---------------------------
st.set_page_config(page_title="English Accent Classifier", layout="centered")
st.title("🎙️ English Accent Classifier")

url = st.text_input("Enter a public video URL (MP4 or direct link):")

if st.button("Analyze Accent") and url:
    with st.spinner("Downloading and processing video..."):
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = os.path.join(tmpdir, 'video.mp4')
            audio_path = os.path.join(tmpdir, 'audio.wav')
            download_video(url, video_path)
            extract_audio(video_path, audio_path)
            model = load_model()
            accent, confidence, explanation = predict_accent(audio_path, model)

    st.success(f"Accent: {accent}")
    st.info(f"Confidence: {confidence:.2f}%")
    st.write(explanation)

st.markdown("---")
st.caption("Built for REM Waste AI Engineer Challenge – Proof of Concept")
