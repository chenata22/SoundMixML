// Същите include-и
#include <iostream>
#include <vector>
#include <cmath>
#include <csignal>
#include <portaudio.h>
#include <sndfile.h>
#include "rnnoise.h"
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <thread>
#include <mutex>
#include <chrono>

#define SAMPLE_RATE 48000
#define FRAMES_PER_BUFFER 480
#define ENERGY_THRESHOLD 0.002f
#define SILENCE_TIMEOUT 1.0f

volatile bool keepRunning = true;

template<typename T>
T clamp(T v, T lo, T hi) {
    return (v < lo) ? lo : (v > hi) ? hi : v;
}

void signalHandler(int signum) {
    keepRunning = false;
}

class BackgroundMusic {
public:
    std::vector<float> samples;
    size_t currentIndex = 0;

    bool load(const std::string& path) {
        SF_INFO sfinfo;
        SNDFILE* file = sf_open(path.c_str(), SFM_READ, &sfinfo);
        if (!file) return false;

        std::vector<float> buffer(sfinfo.frames * sfinfo.channels);
        sf_readf_float(file, buffer.data(), sfinfo.frames);
        sf_close(file);

        if (sfinfo.channels == 2) {
            samples.resize(sfinfo.frames);
            for (int i = 0; i < sfinfo.frames; ++i) {
                samples[i] = 0.5f * (buffer[2 * i] + buffer[2 * i + 1]);
            }
        } else {
            samples = buffer;
        }
        return true;
    }

    float getSample() {
        if (samples.empty()) return 0.0f;
        float sample = samples[currentIndex++];
        if (currentIndex >= samples.size()) currentIndex = 0;
        return sample;
    }
};

class AudioEngine {
public:
    std::vector<float> sendBuffer;
    int frameCount = 0;
    int sockfd = -1;
    std::string lastDecision = "pass";
    DenoiseState* rnState;
    BackgroundMusic bgMusic;
    int silenceFrames = 0;

    AudioEngine() {
        rnState = rnnoise_create(NULL);
    }

    ~AudioEngine() {
        rnnoise_destroy(rnState);
        if (sockfd >= 0) close(sockfd);
    }

    bool connectToServer(const char* ip, int port) {
        sockfd = socket(AF_INET, SOCK_STREAM, 0);
        if (sockfd < 0) return false;

        struct sockaddr_in serv_addr = {0};
        serv_addr.sin_family = AF_INET;
        serv_addr.sin_port = htons(port);
        if (inet_pton(AF_INET, ip, &serv_addr.sin_addr) <= 0) return false;

        if (connect(sockfd, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) return false;
        return true;
    }

    void sendAudioBuffer(float* buffer, size_t size) {
        uint32_t len = size * sizeof(float);
        uint32_t len_le = len;
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
        len_le = __builtin_bswap32(len);
#endif
        send(sockfd, &len_le, sizeof(len_le), 0);

        size_t total_sent = 0;
        while (total_sent < len) {
            ssize_t s = send(sockfd, (char*)buffer + total_sent, len - total_sent, 0);
            if (s <= 0) return;
            total_sent += s;
        }

        char resp[16] = {0};
        ssize_t r = recv(sockfd, resp, sizeof(resp) - 1, 0);
        if (r > 0) {
            resp[r] = '\0';
            lastDecision = std::string(resp);
            std::cout << "[Server response]: " << lastDecision << std::endl;
        } else {
            lastDecision = "pass";
        }
    }
};

// 🔒 Глобални за синхронизация
std::mutex sendMutex;
bool stopSender = false;

void sendThreadFunc(AudioEngine* engine) {
    while (!stopSender) {
        std::vector<float> chunk;
        {
            std::lock_guard<std::mutex> lock(sendMutex);
            if (engine->sendBuffer.size() >= 4800) {
                chunk.assign(engine->sendBuffer.begin(), engine->sendBuffer.begin() + 4800);
                engine->sendBuffer.erase(engine->sendBuffer.begin(), engine->sendBuffer.begin() + 4800);
            }
        }

        if (!chunk.empty()) {
            engine->sendAudioBuffer(chunk.data(), chunk.size());
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
}

static int audioCallback(const void* inputBuffer, void* outputBuffer,
                         unsigned long framesPerBuffer,
                         const PaStreamCallbackTimeInfo* timeInfo,
                         PaStreamCallbackFlags statusFlags,
                         void* userData) {
    AudioEngine* engine = (AudioEngine*)userData;
    const float* in = (const float*)inputBuffer;
    float* out = (float*)outputBuffer;

    if (!inputBuffer) {
        std::fill(out, out + framesPerBuffer, 0.0f);
        return paContinue;
    }

    float input_frame[480];
    for (int j = 0; j < 480; ++j) input_frame[j] = in[j];

    rnnoise_process_frame(engine->rnState, input_frame, input_frame);

    float energy = 0.0f;
    for (int j = 0; j < 480; ++j) energy += input_frame[j] * input_frame[j];
    energy /= 480;

    if (energy < ENERGY_THRESHOLD) {
        engine->silenceFrames++;
    } else {
        engine->silenceFrames = 0;
    }

    if (engine->silenceFrames * (480.0 / SAMPLE_RATE) < SILENCE_TIMEOUT) {
        std::lock_guard<std::mutex> lock(sendMutex);
        engine->sendBuffer.insert(engine->sendBuffer.end(), input_frame, input_frame + 480);
    }

    for (int j = 0; j < 480; ++j) {
        float music_sample = engine->bgMusic.getSample();
        if (engine->lastDecision == "pass") {
            out[j] = clamp(input_frame[j] + music_sample * 0.5f, -1.0f, 1.0f);
        } else {
            out[j] = music_sample;
        }
    }

    engine->frameCount += framesPerBuffer;
    return paContinue;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: ./audio_mix <background_music.wav>\n";
        return 1;
    }

    signal(SIGINT, signalHandler);

    AudioEngine engine;
    if (!engine.bgMusic.load(argv[1])) {
        std::cerr << "Failed to load background music.\n";
        return 1;
    }

    if (!engine.connectToServer("127.0.0.1", 5050)) {
        std::cerr << "Failed to connect to server.\n";
        return 1;
    }

    PaError err = Pa_Initialize();
    if (err != paNoError) {
        std::cerr << "PortAudio error: " << Pa_GetErrorText(err) << "\n";
        return 1;
    }

    PaStream* stream;
    err = Pa_OpenDefaultStream(&stream, 1, 1, paFloat32, SAMPLE_RATE, FRAMES_PER_BUFFER, audioCallback, &engine);
    if (err != paNoError) {
        std::cerr << "Stream open error: " << Pa_GetErrorText(err) << "\n";
        return 1;
    }

    err = Pa_StartStream(stream);
    if (err != paNoError) {
        std::cerr << "Stream start error: " << Pa_GetErrorText(err) << "\n";
        return 1;
    }

    std::thread sender(sendThreadFunc, &engine);

    std::cout << "Streaming audio. Press Ctrl+C to stop...\n";
    while (keepRunning) {
        Pa_Sleep(100);
    }

    stopSender = true;
    sender.join();

    Pa_StopStream(stream);
    Pa_CloseStream(stream);
    Pa_Terminate();

    return 0;
}
