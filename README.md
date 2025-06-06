# SoundMixML

PortAudio

Библиотека за аудио вход/изход в реално време

Пакетите, които трябва да инсталираш на Ubuntu/Debian:

bash
Copy
Edit
sudo apt-get install portaudio19-dev libportaudiocpp0
libsndfile

За зареждане и запис на аудио файлове (wav)

Ubuntu пакет:

bash
Copy
Edit
sudo apt-get install libsndfile1-dev
rnnoise

Библиотека за шумопотискане, от Xiph.org

Обикновено трябва да клонираш и компилираш от източника:

bash
Copy
Edit
git clone https://github.com/xiph/rnnoise.git
cd rnnoise
./autogen.sh
./configure
make
sudo make install
Може да се наложи да имаш и autoconf, automake и други инструменти:

bash
Copy
Edit
sudo apt-get install autoconf automake libtool
pthread (POSIX threads)

Обикновено вече е инсталиран и част от стандартната libc, но трябва да го линкнеш при компилация с -lpthread.

Python зависимости (ако имаш отделен Python скрипт)
numpy – за числени операции

soundfile – за четене/запис на аудио файлове (Python wrapper за libsndfile)

portaudio (чрез pyaudio или друга обвивка) – за аудио вход/изход (ако използваш Python за аудио стрийминг)

Примерен requirements.txt за Python скрипта:

nginx
Copy
Edit
numpy
soundfile
pyaudio
Как да инсталираш
За системните библиотеки:
bash
Copy
Edit
sudo apt-get update
sudo apt-get install -y portaudio19-dev libportaudiocpp0 libsndfile1-dev autoconf automake libtool
После клонирай и компилирай rnnoise:

bash
Copy
Edit
git clone https://github.com/xiph/rnnoise.git
cd rnnoise
./autogen.sh
./configure
make
sudo make install
За Python зависимостите:
bash
Copy
Edit
pip install -r requirements.txt
(ако нямаш requirements.txt, директно pip install numpy soundfile pyaudio)

Как да компилираш C++ кода
bash
Copy
Edit
g++ audio.cpp -o audio_mix -lportaudio -lsndfile -lrnnoise -lpthread
