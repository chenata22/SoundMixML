import os
import librosa
import numpy as np
import pickle
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ==== Конфигурация ====
AUDIO_DIR = "audio_segments"
min_speech = 400
min_non_speech = 400

X = []
y = []
speech_count = 0
non_speech_count = 0

print("🔄 Зареждане и извличане на MFCC от локалните файлове...")
all_files = [f for f in os.listdir(AUDIO_DIR) if f.endswith(".wav")]
pbar = tqdm(all_files, desc="Файлове")

for filename in pbar:
    filepath = os.path.join(AUDIO_DIR, filename)

    # Определи етикета по името на файла
    if filename.startswith("speech_"):
        label = 1
        if speech_count >= min_speech:
            continue
        speech_count += 1
    elif filename.startswith("nonspeech_"):
        label = 0
        if non_speech_count >= min_non_speech:
            continue
        non_speech_count += 1
    else:
        continue  # Непознато име

    try:
        y_val, sr = librosa.load(filepath, sr=16000)
        mfcc = librosa.feature.mfcc(y=y_val, sr=sr, n_mfcc=13)
        features = np.mean(mfcc, axis=1)
    except Exception:
        continue

    X.append(features)
    y.append(label)

    if speech_count >= min_speech and non_speech_count >= min_non_speech:
        break

pbar.close()

print(f"\n✅ Заредени: Говор={speech_count}, Не говор={non_speech_count}")

# ==== Обучение ====
if len(X) == 0:
    print("[X] Няма достатъчно данни за обучение!")
    exit(1)

X = np.array(X)
y = np.array(y)

print("🧠 Обучение на модел...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# ==== Оценка ====
y_pred = clf.predict(X_test)
report = classification_report(y_test, y_pred)
print("\n📊 Резултати от класификация:\n")
print(report)

# ==== Запис ====
with open("random_forest_speech_model.pkl", "wb") as f:
    pickle.dump(clf, f)

with open("report.txt", "w") as f:
    f.write(report)

with open("Xy_data.pkl", "wb") as f:
    pickle.dump((X, y), f)

print("\n💾 Моделът е записан като: random_forest_speech_model.pkl")
