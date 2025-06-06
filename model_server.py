import socket
import joblib
import numpy as np
import librosa
import struct
import threading

# Зареждане на модели
speech_model = joblib.load("random_forest_speech_model.pkl")
urban_model = joblib.load("urban_rf_model.pkl")

urban_label_map = {
    0: "air_conditioner",
    1: "car_horn",
    2: "children_playing",
    3: "dog_bark",
    4: "drilling",
    5: "engine_idling",
    6: "gun_shot",
    7: "jackhammer",
    8: "siren",
    9: "street_music"
}

BUFFER_SIZE = 1600  # Примерен размер в 16kHz семпли

def extract_features(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=512, hop_length=256, fmax=8000)
    features = np.mean(mfcc.T, axis=0).reshape(1, -1)
    print(f"[🧩 Features extracted] Shape: {features.shape}", flush=True)
    return features

def handle_client(conn, addr):
    print(f"[🔌 Connection from {addr}]", flush=True)
    audio_buffer = []

    try:
        while True:
            # Чакаме 4 байта дължина
            length_bytes = conn.recv(4)
            if not length_bytes:
                print("[❌ Connection lost]", flush=True)
                break

            if len(length_bytes) < 4:
                print("[⚠️ Incomplete header received]", flush=True)
                break

            length = struct.unpack('<I', length_bytes)[0]
            print(f"[📏 Expecting {length} bytes of audio data]", flush=True)

            audio_bytes = b''
            while len(audio_bytes) < length:
                data = conn.recv(length - len(audio_bytes))
                if not data:
                    print("[⚠️ Incomplete audio received]", flush=True)
                    break
                audio_bytes += data
                print(f"[⬇️ Receiving data] {len(audio_bytes)}/{length} bytes received", flush=True)

            if len(audio_bytes) != length:
                print("[⚠️ Wrong amount of audio data received]", flush=True)
                break

            audio_np = np.frombuffer(audio_bytes, dtype=np.float32)
            print(f"[📥 Received] {len(audio_np)} samples at 48kHz", flush=True)

            audio_np_16k = librosa.resample(audio_np, orig_sr=48000, target_sr=16000)
            print(f"[🔄 Resampled] {len(audio_np_16k)} samples at 16kHz", flush=True)

            audio_buffer.extend(audio_np_16k.tolist())
            if len(audio_buffer) > BUFFER_SIZE:
                audio_buffer = audio_buffer[-BUFFER_SIZE:]

            print(f"[📊 Buffer] Current size: {len(audio_buffer)}/{BUFFER_SIZE}", flush=True)

            if len(audio_buffer) >= BUFFER_SIZE:
                buffer_chunk = np.array(audio_buffer[:BUFFER_SIZE])
                audio_buffer = audio_buffer[BUFFER_SIZE:]

                print(f"[🔬 Classifying chunk of {BUFFER_SIZE} samples]", flush=True)
                feat = extract_features(buffer_chunk, sr=16000)

                pred1 = speech_model.predict(feat)[0]
                pred2 = urban_model.predict(feat)[0]
                urban_label = urban_label_map.get(pred2, "unknown")

                print(f"[DEBUG] pred1={pred1} ({type(pred1)}), pred2={pred2} ({urban_label})", flush=True)

                # 🔍 Проверка за говор
                if pred1 == 1:
                    print("🗣️ Speech detected!", flush=True)
                else:
                    print("🤫 No speech.", flush=True)

                if urban_label in ["siren", "car_horn"]:
                    decision = "pass"
                else:
                    decision = "mute"

                print(f"[Decision] {decision}", flush=True)
                conn.sendall(decision.encode())

    except Exception as e:
        print(f"[⚠️ Error]: {e}", flush=True)
    finally:
        conn.close()
        print(f"[🔒 Connection with {addr} closed]", flush=True)

def main():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('localhost', 5050))
    server_socket.listen(5)
    print("📡 Model server ready on port 5050", flush=True)

    while True:
        print("[🔗 Waiting for connection...]", flush=True)
        conn, addr = server_socket.accept()
        thread = threading.Thread(target=handle_client, args=(conn, addr))
        thread.start()

if __name__ == "__main__":
    main()
