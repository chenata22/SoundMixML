import csv
import os
import subprocess
from tqdm import tqdm

SPEECH_LABELS = {
    "/m/09x0r",       # Speech
    "/m/0ytgt",       # Conversation
    "/m/03fwl",       # Narration
    "/m/04rlf",       # Babbling
    "/m/012xff",      # Whispering
    "/t/dd00003",     # Male speech, man speaking
    "/t/dd00005",     # Female speech, woman speaking
    "/t/dd00006",     # Child speech, kid speaking
}

CSV_FILE = "unbalanced_train_segments.csv"  # Замени с името на твоя CSV файл
OUTPUT_DIR = "audio_segments"
os.makedirs(OUTPUT_DIR, exist_ok=True)

min_speech = 400
min_non_speech = 400
max_attempts = 100000  # Максимален брой опити

speech_count = 0
non_speech_count = 0
attempts = 0

def is_speech(labels_str):
    labels = [label.strip().strip('"').strip("'") for label in labels_str.split(",")]
    return any(label in SPEECH_LABELS for label in labels)

print("Тагове на първите 10 реда:")
with open(CSV_FILE, "r", encoding="utf-8") as f:
    reader = csv.reader(f, delimiter=",")
    header = next(reader)
    for i, row in enumerate(reader):
        if i >= 10:
            break
        print(row[3])

if speech_count == 0:
    print("[!] В момента нямаме речеви примери - увери се, че CSV-то съдържа записи с тагове от SPEECH_LABELS.")

with open(CSV_FILE, "r", encoding="utf-8") as f:
    reader = csv.reader(f, delimiter=",")
    next(reader)  # Пропускаме хедъра

    pbar = tqdm(total=min_speech + min_non_speech, desc="Извлечени примери")
    while attempts < max_attempts and (speech_count < min_speech or non_speech_count < min_non_speech):
        try:
            row = next(reader)
        except StopIteration:
            print("[!] Достигнат е край на CSV файла.")
            break

        attempts += 1

        if len(row) < 4:
            continue  # Пропускаме редове с малко колони

        yt_id = row[0]
        start = row[1]
        end = row[2]
        labels = row[3]

        speech = is_speech(labels)

        # Пропускаме, ако вече имаме достатъчно примери от съответната категория
        if speech and speech_count >= min_speech:
            continue
        if not speech and non_speech_count >= min_non_speech:
            continue

        label_prefix = "speech" if speech else "nonspeech"
        filename = f"{label_prefix}_{yt_id}_{start}_{end}.wav"
        filepath = os.path.join(OUTPUT_DIR, filename)

        if os.path.exists(filepath):
            # Ако файлът вече съществува, добавяме към брояча и продължаваме
            if speech:
                speech_count += 1
            else:
                non_speech_count += 1
            pbar.update(1)
            continue

        cmd = [
            "yt-dlp",
            "-x",
            "--audio-format", "wav",
            "--download-sections", f"*{start}-{end}",
            "-o", filepath,
            f"https://www.youtube.com/watch?v={yt_id}"
        ]

        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            print(f"[!] Грешка при сваляне на {yt_id} ({start}-{end}): {e}")
            continue

        if speech:
            speech_count += 1
        else:
            non_speech_count += 1

        pbar.update(1)

    pbar.close()

print(f"Говор: {speech_count}, Не говор: {non_speech_count}, Опити: {attempts}")

if speech_count == 0:
    print("[!] ВНИМАНИЕ: Няма свалени речеви примери! Проверете SPEECH_LABELS и CSV файла.")
elif non_speech_count == 0:
    print("[!] ВНИМАНИЕ: Няма свалени не-речеви примери!")
else:
    print("[+] Свободно продължаваме към обучението или следващата стъпка.")
