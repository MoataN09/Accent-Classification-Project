
import os
import numpy as np
import librosa
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import glob

def extract_features_from_folder(base_path):
    X, y = [], []
    for label in os.listdir(base_path):
        label_path = os.path.join(base_path, label)
        if os.path.isdir(label_path):
            for file in glob.glob(os.path.join(label_path, '*.wav')):
                try:
                    y_audio, sr = librosa.load(file, sr=16000)
                    mfcc = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=13)
                    mfcc_mean = np.mean(mfcc.T, axis=0)
                    X.append(mfcc_mean)
                    y.append(label)
                except Exception as e:
                    print(f"Failed to process {file}: {e}")
    return np.array(X), np.array(y)

def main():
    dataset_path = 'accents_dataset'  # ensure this exists with subfolders like 'british', 'american', etc.
    X, y = extract_features_from_folder(dataset_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    joblib.dump(model, 'accent_model.pkl')
    print("Model saved as 'accent_model.pkl'")

if __name__ == '__main__':
    main()
