import os
import pandas as pd
import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Пътища
metadata_path = "UrbanSound8K/metadata/UrbanSound8K.csv"
audio_dir = "UrbanSound8K/audio"

# Четене на CSV
df = pd.read_csv(metadata_path)

# Извличане на характеристики
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, mono=True, duration=4.0)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"Грешка при {file_path}: {e}")
        return None

# Зареждане на всички данни
features = []
labels = []

for i, row in df.iterrows():
    fold = f"fold{row['fold']}"
    file_name = row['slice_file_name']
    file_path = os.path.join(audio_dir, fold, file_name)
    data = extract_features(file_path)
    if data is not None:
        features.append(data)
        labels.append(row['classID'])

# Преобразуване в NumPy
X = np.array(features)
y = np.array(labels)

# Разделяне
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Оценка
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# Записване на модела
joblib.dump(clf, "urban_rf_model.pkl")
print("Моделът е запазен като urban_rf_model.pkl")
