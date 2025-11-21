#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <iostream>
#include <vector>
#include <queue>
#include <mutex>
#include <thread>
#include <atomic>
#include <condition_variable>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <cmath>
#include <csignal>
#include <cstring>
#include <memory>
#include <numeric>
#include <iomanip>
#include <cstdlib>
#include <cctype>
#include <sys/stat.h>
#include <map>
#include <sstream>

// ALSA
#include <alsa/asoundlib.h>

// ONNX Runtime (for VAD)
#include "onnxruntime_cxx_api.h"

// Vosk
#include <vosk_api.h>

// Networking
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <getopt.h>

#include "JsonEventServer.h"
#include "wav.h"

// =============================================================================
// Global Variables and Signal Handling
// =============================================================================
std::atomic<bool> keep_running{true};
std::atomic<bool> signal_received{false};
volatile sig_atomic_t vad_config_update_requested = 0;
volatile sig_atomic_t vosk_mode_update_requested = 0;
volatile sig_atomic_t commands_update_requested = 0;

static JsonEventServer *g_vad_event_server = nullptr;
static JsonEventServer *g_volume_server = nullptr;
static JsonEventServer *g_vosk_final_server = nullptr;
static JsonEventServer *g_vosk_partial_server = nullptr;
const int COMMAND_PORT = 6975;
std::condition_variable vad_queue_cv;
std::condition_variable vosk_queue_cv;
std::string g_last_hostname;
std::atomic<bool> vosk_processor_finished{false};
// Slot Definitions and Grammar System
struct CommandSlots {
    std::string action;      // turn_on, turn_off, set
    std::string target;      // led, brightness, audio
    std::string side;        // front, rear
    std::string color;       // color name
    std::string brightness;  // min, max, numeric, dim, bright
    std::string audio_port;  // speaker, headphone
};

enum class Intent {
    UNKNOWN,
    LED_POWER_ON,
    LED_POWER_OFF,
    LED_SET_COLOR,
    BRIGHTNESS_SET,
    AUDIO_PORT_SET
};

std::mutex grammar_mutex;


// NEW: Hardcoded trigger phrase for command mode
const std::string TRIGGER_PHRASE = "hey aurora";
const int TRIGGER_TIMEOUT_MS = 5000; // 5 seconds

void signal_handler(int signal) {
    if (signal == SIGINT) {
        if (signal_received.exchange(true)) {
            std::cout << "\n\nForce exit!" << std::endl;
            _exit(1);
        }
        std::cout << "\n\nCtrl+C detected. Stopping gracefully..." << std::endl;
        keep_running = false;
        
        vad_queue_cv.notify_all();
        vosk_queue_cv.notify_all();
        
        if (g_vad_event_server) {
            g_vad_event_server->stop();
        }
        if (g_volume_server) {
            g_volume_server->stop();
        }
        if (g_vosk_final_server) {
            g_vosk_final_server->stop();
        }
        if (g_vosk_partial_server) {
            g_vosk_partial_server->stop();
        }
    }
}

void unified_config_handler(int sig) {
    vad_config_update_requested = 1;
    vosk_mode_update_requested = 1;
    commands_update_requested = 1;
    signal(SIGUSR2, unified_config_handler);
}

// =============================================================================
// Audio Queues
// =============================================================================
std::queue<std::vector<float>> vad_audio_queue;
std::queue<std::vector<int16_t>> vosk_audio_queue;
std::mutex vad_queue_mutex;
std::mutex vosk_queue_mutex;

// =============================================================================
// Configuration
// =============================================================================
struct Config {
    std::string alsa_device = "hw:0,7";
    std::string vad_model_path="models/silero_vad_16k_op15.onnx";
    std::string vosk_model_path = "models/vosk_model";
    int vad_event_port = 12346;
    int volume_port = 12345;
    std::string vosk_tcp_host = "127.0.0.1";
    int vosk_final_port = 9999;
    int vosk_partial_port = 9998;
    std::string mode_file = "mode.txt";
    std::string grammar_file = "keywords.txt";
    std::string commands_file = "commands.txt";
};

Config parse_args(int argc, char *argv[]) {
    Config cfg;
    
    static struct option long_opts[] = {
        {"device", required_argument, nullptr, 'd'},
        {"vad-model", required_argument, nullptr, 'v'},
        {"vosk-model", required_argument, nullptr, 'k'},
        {"vad-event-port", required_argument, nullptr, 'e'},
        {"volume-port", required_argument, nullptr, 'l'},
        {"vosk-host", required_argument, nullptr, 'h'},
        {"vosk-final-port", required_argument, nullptr, 'f'},
        {"vosk-partial-port", required_argument, nullptr, 'a'},
        {"mode-file", required_argument, nullptr, 'm'},
        {"grammar-file", required_argument, nullptr, 'g'},
        {"commands-file", required_argument, nullptr, 'c'},
        {"command-port", required_argument, nullptr, 'p'},
        {nullptr, 0, nullptr, 0}
    };
    
    int opt;
    while ((opt = getopt_long(argc, argv, "d:v:k:e:l:h:f:a:m:g:c:p:", long_opts, nullptr)) != -1) {
        switch (opt) {
            case 'd': cfg.alsa_device = optarg; break;
            case 'v': cfg.vad_model_path = optarg; break;
            case 'k': cfg.vosk_model_path = optarg; break;
            case 'e': cfg.vad_event_port = std::stoi(optarg); break;
            case 'l': cfg.volume_port = std::stoi(optarg); break;
            case 'h': cfg.vosk_tcp_host = optarg; break;
            case 'f': cfg.vosk_final_port = std::stoi(optarg); break;
            case 'a': cfg.vosk_partial_port = std::stoi(optarg); break;
            case 'm': cfg.mode_file = optarg; break;
            case 'g': cfg.grammar_file = optarg; break;
            case 'c': cfg.commands_file = optarg; break;
            default:
                std::cerr << "Usage: " << argv[0] << "\n"
                          << "  -d, --device <dev>            ALSA device (default: hw:0,7)\n"
                          << "  -v, --vad-model <path>        VAD model path\n"
                          << "  -k, --vosk-model <path>       Vosk model path (default: models/vosk_model)\n"
                          << "  -e, --vad-event-port <port>   VAD event port (default: 12346)\n"
                          << "  -l, --volume-port <port>      Volume port (default: 12345)\n"
                          << "  -h, --vosk-host <ip>          Vosk TCP host (default: 127.0.0.1)\n"
                          << "  -f, --vosk-final-port <port>  Finals port (default: 9999)\n"
                          << "  -a, --vosk-partial-port <port> Partials port (default: 9998)\n"
                          << "  -m, --mode-file <path>        Mode file (default: mode.txt)\n"
                          << "  -g, --grammar-file <path>     Grammar file (default: keywords.txt)\n";

        }
    }
    return cfg;
}

// =============================================================================
// VAD Config
// =============================================================================
static void getEnvVar(const char *field, const char *default_val, char *out, int len)
{
    strcpy(out, default_val);
}

// =============================================================================
// Volume Calculation
// =============================================================================
volumeData_t calculateVolumeStereo(const std::vector<float>& samples) {
    float sumSquaresL = 0.0f, sumSquaresR = 0.0f;
    int count = samples.size() / 2;
    
    if (count == 0) {
        return {-100, -100, 0, 0};
    }
    
    for (size_t i = 0; i < samples.size(); i += 2) {
        sumSquaresL += samples[i] * samples[i];
        sumSquaresR += samples[i + 1] * samples[i + 1];
    }
    
    float rmsL = sqrt(sumSquaresL / count);
    float rmsR = sqrt(sumSquaresR / count);
    float peakDbL = 20 * log10(rmsL);
    float peakDbR = 20 * log10(rmsR);
    
    if (std::isinf(peakDbL)) peakDbL = -100;
    if (std::isinf(peakDbR)) peakDbR = -100;
    
    return {peakDbL, peakDbR, rmsL, rmsR};
}

// =============================================================================
// Circular Audio Buffer (for VAD)
// =============================================================================
class CircularAudioBuffer {
public:
    CircularAudioBuffer(size_t max_samples)
        : buffer(max_samples, 0.0f), capacity(max_samples), head(0), tail(0), size_(0) {}
    
    void push(const std::vector<float>& chunk) {
        for (float sample : chunk) {
            buffer[head] = sample;
            head = (head + 1) % capacity;
            if (size_ < capacity) {
                ++size_;
            } else {
                tail = (tail + 1) % capacity;
            }
        }
    }
    
    std::vector<float> get_latest(size_t num_samples) const {
        num_samples = std::min(num_samples, size_);
        std::vector<float> out(num_samples);
        size_t start = (head + capacity - num_samples) % capacity;
        for (size_t i = 0; i < num_samples; ++i)
            out[i] = buffer[(start + i) % capacity];
        return out;
    }
    
    void discard_latest(size_t num_samples) {
        num_samples = std::min(num_samples, size_);
        size_ -= num_samples;
        head = (head + capacity - num_samples) % capacity;
    }
    
    size_t size() const { return size_; }

private:
    std::vector<float> buffer;
    size_t capacity, head, tail, size_;
};

// =============================================================================
// Timestamp Structure
// =============================================================================
class timestamp_t {
public:
    int start, end;
    
    timestamp_t(int start = -1, int end = -1) : start(start), end(end) {}
    
    bool operator==(const timestamp_t& a) const {
        return (start == a.start && end == a.end);
    }
};

// =============================================================================
// VAD Iterator Class
// =============================================================================
class VadIterator {
private:
    Ort::Env env;
    Ort::SessionOptions session_options;
    std::shared_ptr<Ort::Session> session = nullptr;
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeCPU);
    
    const int context_samples = 64;
    std::vector<float> _context;
    int window_size_samples, effective_window_size, sr_per_ms;
    
    bool word_gate_enable = true;
    int speech_start_timeout = 3;
    int speech_end_timeout = 3;
    
    std::vector<Ort::Value> ort_inputs;
    std::vector<const char*> input_node_names = {"input", "state", "sr"};
    std::vector<float> input;
    unsigned int size_state = 2 * 1 * 128;
    std::vector<float> _state;
    std::vector<int64_t> sr;
    int64_t input_node_dims[2] = {};
    const int64_t state_node_dims[3] = {2, 1, 128};
    const int64_t sr_node_dims[1] = {1};
    std::vector<const char*> output_node_names = {"output", "stateN"};
    
    int sample_rate;
    float threshold;
    int min_silence_samples, min_silence_samples_at_max_speech, min_speech_samples;
    float max_speech_samples;
    int speech_pad_samples;
    size_t samples_since_last_write = 0;
    size_t segment_interval_samples = 2 * 16000;

public:
    int audio_length_samples;
    float speech_prob;
    bool triggered = false;
    int temp_end = 0, current_sample = 0, prev_end = 0, next_start = 0;
    std::vector<timestamp_t> speeches;
    timestamp_t current_speech;
    CircularAudioBuffer audio_buffer;
    
    VadIterator(const std::string& ModelPath, int Sample_rate = 16000,
                int windows_frame_size = 32, float Threshold = 0.5,
                int min_silence_duration_ms = 2, int speech_pad_ms = 30,
                int min_speech_duration_ms = 10,
                float max_speech_duration_s = std::numeric_limits<float>::infinity())
        : sample_rate(Sample_rate), threshold(Threshold),
          speech_pad_samples(speech_pad_ms), prev_end(0),
          audio_buffer(30 * Sample_rate) {
        
        sr_per_ms = sample_rate / 1000;
        window_size_samples = windows_frame_size * sr_per_ms;
        effective_window_size = window_size_samples + context_samples;
        input_node_dims[0] = 1;
        input_node_dims[1] = effective_window_size;
        _state.resize(size_state);
        sr.resize(1);
        sr[0] = sample_rate;
        _context.assign(context_samples, 0.0f);
        min_speech_samples = sr_per_ms * min_speech_duration_ms;
        max_speech_samples = (sample_rate * max_speech_duration_s - 
                             window_size_samples - 2 * speech_pad_samples);
        min_silence_samples = sr_per_ms * min_silence_duration_ms;
        min_silence_samples_at_max_speech = sr_per_ms * 98;
        init_onnx_model(ModelPath);
    }
    
    void init_onnx_model(const std::string& model_path) {
        session_options.SetIntraOpNumThreads(1);
        session_options.SetInterOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session = std::make_shared<Ort::Session>(env, model_path.c_str(), session_options);
    }
    
    void reset_states() {
        std::memset(_state.data(), 0, _state.size() * sizeof(float));
        triggered = false;
        temp_end = current_sample = prev_end = next_start = 0;
        speeches.clear();
        current_speech = timestamp_t();
        std::fill(_context.begin(), _context.end(), 0.0f);
    }
    
    void set_speech_timeout(int start_timeout_ms, int end_timeout_ms) {
        speech_start_timeout = start_timeout_ms;
        speech_end_timeout = end_timeout_ms;
    }
    
    void set_speech_threshold(float new_threshold) {
        threshold = 1 + std::log10(new_threshold) / std::exp(1.0f);
        std::cout << "[VAD] Speech threshold set to: " << threshold << std::endl;
    }
    
    void set_word_gate_enable(bool enable) {
        word_gate_enable = enable;
    }
    
    void predict(const std::vector<float>& data_chunk) {
        audio_buffer.push(data_chunk);
        
        std::vector<float> new_data(effective_window_size, 0.0f);
        std::copy(_context.begin(), _context.end(), new_data.begin());
        std::copy(data_chunk.begin(), data_chunk.end(), new_data.begin() + context_samples);
        input = new_data;
        
        Ort::Value input_ort = Ort::Value::CreateTensor<float>(
            memory_info, input.data(), input.size(), input_node_dims, 2);
        Ort::Value state_ort = Ort::Value::CreateTensor<float>(
            memory_info, _state.data(), _state.size(), state_node_dims, 3);
        Ort::Value sr_ort = Ort::Value::CreateTensor<int64_t>(
            memory_info, sr.data(), sr.size(), sr_node_dims, 1);
        
        ort_inputs.clear();
        ort_inputs.emplace_back(std::move(input_ort));
        ort_inputs.emplace_back(std::move(state_ort));
        ort_inputs.emplace_back(std::move(sr_ort));
        
        auto ort_outputs = session->Run(Ort::RunOptions{nullptr},
                                        input_node_names.data(), ort_inputs.data(), ort_inputs.size(),
                                        output_node_names.data(), output_node_names.size());
        
        speech_prob = ort_outputs[0].GetTensorMutableData<float>()[0];
        float* stateN = ort_outputs[1].GetTensorMutableData<float>();
        std::memcpy(_state.data(), stateN, size_state * sizeof(float));
        current_sample += window_size_samples;
        
        static int event = 0;
        static int speech_start = 0;
        
        if (speech_prob >= threshold) {
            float speech = current_sample - window_size_samples;
            if (temp_end != 0) {
                temp_end = 0;
                if (next_start < prev_end)
                    next_start = current_sample - window_size_samples;
            }
            if (!triggered) {
                triggered = true;
                current_speech.start = current_sample - window_size_samples;
                printf("[VAD] Speech start: %.3f s (prob: %.3f)\n",
                       speech / (float)sample_rate, speech_prob);
            }
            if (!speech_start) {
                speech_start = current_speech.start;
            }
            if (!event && (current_sample - speech_start) >= 
                (speech_start_timeout * sample_rate / 1000)) {
                event = 1;
                if (word_gate_enable && g_vad_event_server) {
                    g_vad_event_server->broadcast_event(event);
                }
                printf("[VAD] Speech segment started at: %.3f s (event: %d)\n",
                       speech_start / (float)sample_rate, event);
            }
        } else if (speech_prob < (threshold - 0.15f)) {
            if (triggered && (current_sample - current_speech.start) > min_speech_samples) {
                temp_end = current_sample;
                next_start = current_sample;
                triggered = false;
            }
            if (event && ((current_sample - temp_end) >= 
                (speech_end_timeout * sample_rate / 1000))) {
                event = 0;
                speech_start = 0;
                if (word_gate_enable && g_vad_event_server) {
                    g_vad_event_server->broadcast_event(event);
                }
                printf("[VAD] Speech segment ended at: %.3f s (event: %d)\n",
                       temp_end / (float)sample_rate, event);
            }
        }
        
        std::copy(new_data.end() - context_samples, new_data.end(), _context.begin());
    }
    
    void reset() {
        reset_states();
    }
};

// =============================================================================
// Grammar-Based Phrase Loading
// =============================================================================
std::vector<std::string> load_phrases_from_file(const std::string& filepath) {
    std::vector<std::string> phrases;
    std::ifstream file(filepath);
    
    if (!file.is_open()) {
        std::cerr << "[GRAMMAR] Failed to open " << filepath << ". Using defaults.\n";
        return {
            " on ",
            " off ",
            "down the ",
            "up the display brightness",
            "set led to color green magenta cyan orange white blue red",
            "turn off the screen",
            "hey aurora",
            "aura",
            "change the from"
        };
    }
    
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        
        std::string phrase = line;
        // Trim trailing whitespace
        while (!phrase.empty() && (phrase.back() == ' ' || phrase.back() == '\t' || 
               phrase.back() == '\n' || phrase.back() == '\r')) {
            phrase.pop_back();
        }
        
        // Convert to lowercase
        std::transform(phrase.begin(), phrase.end(), phrase.begin(), ::tolower);
        
        if (!phrase.empty()) {
            phrases.push_back(phrase);
        }
    }
    
    file.close();
    return phrases;
}

// Build Vosk grammar JSON from phrases
std::string build_grammar_json(const std::vector<std::string>& phrases) {
    std::string grammar = "[";
    for (size_t i = 0; i < phrases.size(); ++i) {
        grammar += "\"" + phrases[i] + "\"";
        if (i + 1 < phrases.size()) {
            grammar += ", ";
        }
    }
    grammar += "]";
    return grammar;
}

// =============================================================================
// Normalization and Synonym Mapping
// =============================================================================

std::string normalize_string(const std::string& input) {
    std::string result = input;
    
    // Convert to lowercase
    std::transform(result.begin(), result.end(), result.begin(), ::tolower);
    
    // Trim leading whitespace
    size_t start = 0;
    while (start < result.length() && std::isspace(result[start])) {
        start++;
    }
    
    // Trim trailing whitespace
    size_t end = result.length();
    while (end > start && std::isspace(result[end - 1])) {
        end--;
    }
    
    result = result.substr(start, end - start);
    
    // Replace multiple spaces with single space
    std::string cleaned;
    bool last_was_space = false;
    for (char c : result) {
        if (std::isspace(c)) {
            if (!last_was_space) {
                cleaned += ' ';
                last_was_space = true;
            }
        } else {
            cleaned += c;
            last_was_space = false;
        }
    }
    
    return cleaned;
}

std::string remove_filler_words(const std::string& input) {
    static const std::vector<std::string> filler_words = {
        "the", "a", "an", "please", "could", "you", "can", 
        "would", "should", "is", "are", "am", "was", "were",
        "to", "of", "and"
    };
    
    std::string result = " " + input + " "; // Add spaces for boundary matching
    
    for (const auto& filler : filler_words) {
        std::string pattern = " " + filler + " ";
        size_t pos = 0;
        while ((pos = result.find(pattern, pos)) != std::string::npos) {
            result.replace(pos, pattern.length(), " ");
        }
    }
    
    return normalize_string(result);
}

std::string apply_synonyms(const std::string& input) {
    std::string result = input;
    
    // LED synonyms
    size_t pos = 0;
    while ((pos = result.find("light", pos)) != std::string::npos) {
        result.replace(pos, 5, "led");
        pos += 3;
    }
    pos = 0;
    while ((pos = result.find("lamp", pos)) != std::string::npos) {
        result.replace(pos, 4, "led");
        pos += 3;
    }
    
    // Display synonyms
    pos = 0;
    while ((pos = result.find("screen", pos)) != std::string::npos) {
        result.replace(pos, 6, "lcd");
        pos += 3;
    }
    pos = 0;
    while ((pos = result.find("display", pos)) != std::string::npos) {
        result.replace(pos, 7, "lcd");
        pos += 3;
    }
    
    // Audio synonyms
    pos = 0;
    while ((pos = result.find("speakers", pos)) != std::string::npos) {
        result.replace(pos, 8, "speaker");
        pos += 7;
    }
    pos = 0;
    while ((pos = result.find("headphones", pos)) != std::string::npos) {
        result.replace(pos, 10, "headphone");
        pos += 9;
    }
    
    return result;
}

// =============================================================================
// Grammar Rules for Intent Detection
// =============================================================================

Intent detect_intent(const std::string& normalized_text) {
    // LED Power On patterns
    if (normalized_text.find("turn on") != std::string::npos && 
        normalized_text.find("led") != std::string::npos) {
        return Intent::LED_POWER_ON;
    }
    if (normalized_text.find("switch on") != std::string::npos && 
        normalized_text.find("led") != std::string::npos) {
        return Intent::LED_POWER_ON;
    }
    
    // LED Power Off patterns
    if (normalized_text.find("turn off") != std::string::npos && 
        normalized_text.find("led") != std::string::npos) {
        return Intent::LED_POWER_OFF;
    }
    if (normalized_text.find("switch off") != std::string::npos && 
        normalized_text.find("led") != std::string::npos) {
        return Intent::LED_POWER_OFF;
    }
    
    // LED Color Setting patterns
    if ((normalized_text.find("set") != std::string::npos || 
         normalized_text.find("change") != std::string::npos) && 
        normalized_text.find("led") != std::string::npos &&
        (normalized_text.find("color") != std::string::npos ||
         normalized_text.find("red") != std::string::npos ||
         normalized_text.find("green") != std::string::npos ||
         normalized_text.find("blue") != std::string::npos ||
         normalized_text.find("cyan") != std::string::npos ||
         normalized_text.find("magenta") != std::string::npos ||
         normalized_text.find("yellow") != std::string::npos ||
         normalized_text.find("white") != std::string::npos ||
         normalized_text.find("orange") != std::string::npos)) {
        return Intent::LED_SET_COLOR;
    }
    
    // Brightness patterns
    if ((normalized_text.find("brightness") != std::string::npos ||
         normalized_text.find("dim") != std::string::npos ||
         normalized_text.find("bright") != std::string::npos) &&
        (normalized_text.find("set") != std::string::npos ||
         normalized_text.find("brightness") != std::string::npos)) {
        return Intent::BRIGHTNESS_SET;
    }
    
    // Audio Port patterns
    if (normalized_text.find("audio") != std::string::npos && 
        normalized_text.find("port") != std::string::npos &&
        (normalized_text.find("speaker") != std::string::npos ||
         normalized_text.find("headphone") != std::string::npos)) {
        return Intent::AUDIO_PORT_SET;
    }
    if (normalized_text.find("default") != std::string::npos &&
        (normalized_text.find("speaker") != std::string::npos ||
         normalized_text.find("headphone") != std::string::npos)) {
        return Intent::AUDIO_PORT_SET;
    }
    
    return Intent::UNKNOWN;
}

// =============================================================================
// Slot Extraction
// =============================================================================

CommandSlots extract_slots(const std::string& normalized_text) {
    CommandSlots slots;
    
    // Extract ACTION
    if (normalized_text.find("turn on") != std::string::npos ||
        normalized_text.find("switch on") != std::string::npos) {
        slots.action = "turn_on";
    } else if (normalized_text.find("turn off") != std::string::npos ||
               normalized_text.find("switch off") != std::string::npos) {
        slots.action = "turn_off";
    } else if (normalized_text.find("set") != std::string::npos ||
               normalized_text.find("change") != std::string::npos) {
        slots.action = "set";
    }
    
    // Extract TARGET
    if (normalized_text.find("led") != std::string::npos) {
        slots.target = "led";
    } else if (normalized_text.find("brightness") != std::string::npos ||
               normalized_text.find("lcd") != std::string::npos) {
        slots.target = "brightness";
    } else if (normalized_text.find("audio") != std::string::npos ||
               normalized_text.find("speaker") != std::string::npos ||
               normalized_text.find("headphone") != std::string::npos) {
        slots.target = "audio";
    }
    
    // Extract SIDE
    if (normalized_text.find("front") != std::string::npos) {
        slots.side = "front";
    } else if (normalized_text.find("rear") != std::string::npos || 
               normalized_text.find("back") != std::string::npos) {
        slots.side = "rear";
    } else {
        slots.side = "front"; // Default to front
    }
    
    // Extract COLOR (extended colors)
    if (normalized_text.find("red") != std::string::npos) {
        slots.color = "red";
    } else if (normalized_text.find("green") != std::string::npos) {
        slots.color = "green";
    } else if (normalized_text.find("blue") != std::string::npos) {
        slots.color = "blue";
    } else if (normalized_text.find("cyan") != std::string::npos) {
        slots.color = "cyan";
    } else if (normalized_text.find("magenta") != std::string::npos) {
        slots.color = "magenta";
    } else if (normalized_text.find("yellow") != std::string::npos) {
        slots.color = "yellow";
    } else if (normalized_text.find("white") != std::string::npos) {
        slots.color = "white";
    } else if (normalized_text.find("orange") != std::string::npos) {
        slots.color = "orange";
    }
    
    // Extract BRIGHTNESS
    if (normalized_text.find("min") != std::string::npos || 
        normalized_text.find("minimum") != std::string::npos) {
        slots.brightness = "min";
    } else if (normalized_text.find("max") != std::string::npos || 
               normalized_text.find("maximum") != std::string::npos) {
        slots.brightness = "max";
    } else if (normalized_text.find("dim") != std::string::npos) {
        slots.brightness = "dim";
    } else if (normalized_text.find("bright") != std::string::npos) {
        slots.brightness = "bright";
    } else if (normalized_text.find("half") != std::string::npos || 
               normalized_text.find("50") != std::string::npos) {
        slots.brightness = "half";
    } else {
        // Try to extract numeric value
        std::istringstream iss(normalized_text);
        std::string word;
        while (iss >> word) {
            if (std::all_of(word.begin(), word.end(), ::isdigit)) {
                slots.brightness = word;
                break;
            }
        }
    }
    
    // Extract AUDIO PORT
    if (normalized_text.find("speaker") != std::string::npos) {
        slots.audio_port = "speaker";
    } else if (normalized_text.find("headphone") != std::string::npos) {
        slots.audio_port = "headphone";
    }
    
    return slots;
}

// =============================================================================
// Color Mapping
// =============================================================================

std::string color_to_hex(const std::string& color_name) {
    static const std::map<std::string, std::string> color_map = {
        {"red", "FF0000"},
        {"green", "00FF00"},
        {"blue", "0000FF"},
        {"cyan", "00FFFF"},
        {"magenta", "FF00FF"},
        {"yellow", "FFFF00"},
        {"white", "FFFFFF"},
        {"orange", "FF8000"}
    };
    
    auto it = color_map.find(color_name);
    if (it != color_map.end()) {
        return it->second;
    }
    return "00FF00"; // Default to green
}

// =============================================================================
// Brightness Mapping
// =============================================================================

int brightness_to_value(const std::string& brightness_level, bool is_lcd) {
    if (brightness_level == "min") {
        return is_lcd ? 1 : 10;
    } else if (brightness_level == "max") {
        return is_lcd ? 10 : 100;
    } else if (brightness_level == "dim") {
        return is_lcd ? 3 : 30;
    } else if (brightness_level == "bright") {
        return is_lcd ? 8 : 80;
    } else if (brightness_level == "half") {
        return is_lcd ? 5 : 50;
    } else if (!brightness_level.empty() && 
               std::all_of(brightness_level.begin(), brightness_level.end(), ::isdigit)) {
        int val = std::stoi(brightness_level);
        if (is_lcd) {
            return std::min(10, std::max(1, val));
        } else {
            return std::min(100, std::max(0, val));
        }
    }
    return is_lcd ? 5 : 50; // Default to middle
}

// =============================================================================
// Command Mapping (Intent + Slots → Command String)
// =============================================================================

std::string map_to_command(Intent intent, const CommandSlots& slots) {
    std::lock_guard<std::mutex> lock(grammar_mutex);
    
    switch (intent) {
        case Intent::LED_POWER_ON: {
            std::string color = slots.color.empty() ? "green" : slots.color;
            return "config_led " + slots.side + " " + color_to_hex(color) + " 100";
        }
        
        case Intent::LED_POWER_OFF: {
            return "config_led " + slots.side + " 00FF00 0";
        }
        
        case Intent::LED_SET_COLOR: {
            if (slots.color.empty()) {
                std::cout << "[SLOT-ERROR] No color specified" << std::endl;
                return "";
            }
            return "config_led " + slots.side + " " + color_to_hex(slots.color) + " 100";
        }
        
        case Intent::BRIGHTNESS_SET: {
            if (slots.brightness.empty()) {
                std::cout << "[SLOT-ERROR] No brightness level specified" << std::endl;
                return "";
            }
            int level = brightness_to_value(slots.brightness, true);
            return "set_lcd brightness " + std::to_string(level);
        }
        
        case Intent::AUDIO_PORT_SET: {
            if (slots.audio_port.empty()) {
                std::cout << "[SLOT-ERROR] No audio port specified" << std::endl;
                return "";
            }
            return "set_default_speaker " + slots.audio_port;
        }
        
        default:
            return "";
    }
}

// =============================================================================
// Main Processing Function (Grammar + Slot-Filling)
// =============================================================================

std::string process_command_with_grammar_slots(const std::string& input_text) {
    if (input_text.empty()) return "";
    
    // Step 1: Normalize
    std::string normalized = normalize_string(input_text);
    normalized = remove_filler_words(normalized);
    normalized = apply_synonyms(normalized);
    
    std::cout << "[NORMALIZED] \"" << input_text << "\" -> \"" << normalized << "\"" << std::endl;
    
    // Step 2: Detect Intent
    Intent intent = detect_intent(normalized);
    
    if (intent == Intent::UNKNOWN) {
        std::cout << "[INTENT] Unknown - no match" << std::endl;
        return "";
    }
    
    const char* intent_names[] = {
        "UNKNOWN", "LED_POWER_ON", "LED_POWER_OFF", 
        "LED_SET_COLOR", "BRIGHTNESS_SET", "AUDIO_PORT_SET"
    };
    std::cout << "[INTENT] " << intent_names[static_cast<int>(intent)] << std::endl;
    
    // Step 3: Extract Slots
    CommandSlots slots = extract_slots(normalized);
    
    std::cout << "[SLOTS] action=" << slots.action 
              << " target=" << slots.target 
              << " side=" << slots.side 
              << " color=" << slots.color 
              << " brightness=" << slots.brightness 
              << " audio=" << slots.audio_port << std::endl;
    
    // Step 4: Map to Command
    std::string command = map_to_command(intent, slots);
    
    if (!command.empty()) {
        std::cout << "[COMMAND] " << command << std::endl;
    }
    
    return command;
}

     

// =============================================================================
// TCP Client for Sending Commands
// =============================================================================
bool send_command_to_server(const std::string& command, const std::string& host = "127.0.0.1", int port = 6975) {
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
        std::cerr << "[COMMAND-CLIENT] Socket creation failed" << std::endl;
        return false;
    }
    
    // Set socket timeout
    struct timeval timeout;
    timeout.tv_sec = 2;
    timeout.tv_usec = 0;
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
    
    struct sockaddr_in server_addr;
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(port);
    
    if (inet_pton(AF_INET, host.c_str(), &server_addr.sin_addr) <= 0) {
        std::cerr << "[COMMAND-CLIENT] Invalid address" << std::endl;
        close(sock);
        return false;
    }
    
    if (connect(sock, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        std::cerr << "[COMMAND-CLIENT] Connection failed to " << host << ":" << port << std::endl;
        close(sock);
        return false;
    }
    
    // Send command (add newline if needed)
    std::string msg = command + "\n";
    ssize_t sent = send(sock, msg.c_str(), msg.length(), 0);
    
    close(sock);
    
    if (sent < 0) {
        std::cerr << "[COMMAND-CLIENT] Send failed" << std::endl;
        return false;
    }
    
    return true;
}
// =============================================================================
// File Modification Time
// =============================================================================
time_t get_file_mod_time(const std::string& path) {
    struct stat result;
    if (stat(path.c_str(), &result) == 0) {
        return result.st_mtime;
    }
    return 0;
}

// Read Vosk mode from file
int read_vosk_mode(const std::string& mode_file) {
    std::ifstream file(mode_file);
    if (!file.is_open()) {
        std::cerr << "[VOSK-MODE] Cannot open " << mode_file << ", defaulting to mode 0" << std::endl;
        return 0;
    }
    
    std::string line;
    std::getline(file, line);
    file.close();
    
    // Trim whitespace
    line.erase(std::remove_if(line.begin(), line.end(), ::isspace), line.end());
    
    int mode = std::atoi(line.c_str());
    if (mode != 0 && mode != 1) {
        std::cerr << "[VOSK-MODE] Invalid mode '" << line << "', defaulting to 0" << std::endl;
        return 0;
    }
    
    return mode;
}

void getVadConfig(VadIterator &vad)
{
    const int len = 100;
    std::string val_str(len, '\0'); 
    char* val = &val_str[0];

    getEnvVar("word_gate_timeout", "2000,2000", val, len);
    std::cout << "[VAD Config] word_gate_timeout: " << val << std::endl;
    val_str.assign(val);
    std::string::size_type pos = val_str.find(',');
    if (pos == std::string::npos) {
        std::cerr << "[VAD Config] Invalid word_gate_timeout format." << std::endl;
        return;
    }
    std::string start_timeout_str = val_str.substr(0, pos);
    std::string end_timeout_str = val_str.substr(pos + 1);
    int start_timeout = atoi(start_timeout_str.data());
    int end_timeout = atoi(end_timeout_str.data());
    
    getEnvVar("word_gate_threshold", "0.5", val, len);
    float g_thres = atof(val);
    
    getEnvVar("word_gate_mode", "enable", val, len);
    bool g_mode = (strcmp(val, "enable") == 0) || (strcmp(val, "1") == 0);
    
    vad.set_speech_timeout(start_timeout, end_timeout);
    vad.set_speech_threshold(g_thres);
    vad.set_word_gate_enable(g_mode);
    
    printf("[VAD Config] Mode:%d, start:%d ms, end:%d ms, threshold:%.2f\n",
           g_mode, start_timeout, end_timeout, g_thres);

    if (g_vad_event_server) {
        std::ifstream hostname_file("/etc/hostname");
        std::string hostname;
        if (hostname_file.is_open()) {
            std::getline(hostname_file, hostname);
            hostname_file.close();
            hostname.erase(std::remove_if(hostname.begin(), hostname.end(), ::isspace), hostname.end());
        } else {
            hostname = "unknown_device";
        }
        if (hostname != g_last_hostname) {
            g_vad_event_server->set_hostname(hostname);
            g_last_hostname = hostname;
        }
    }
}

// =============================================================================
// ALSA Capture Thread
// =============================================================================
void alsa_capture_thread(const std::string& device_name) {
    std::cout << "[ALSA] Capture thread started" << std::endl;
    
    snd_pcm_t* pcm_handle;
    snd_pcm_hw_params_t* hw_params;
    unsigned int sample_rate = 16000;
    unsigned int channels = 2;
    snd_pcm_uframes_t frames_per_period = 512;
    int err;
    
    if ((err = snd_pcm_open(&pcm_handle, device_name.c_str(), 
                            SND_PCM_STREAM_CAPTURE, 0)) < 0) {
        std::cerr << "[ALSA] Cannot open PCM device: " << snd_strerror(err) << std::endl;
        keep_running = false;
        return;
    }
    
    snd_pcm_hw_params_alloca(&hw_params);
    snd_pcm_hw_params_any(pcm_handle, hw_params);
    snd_pcm_hw_params_set_access(pcm_handle, hw_params, SND_PCM_ACCESS_RW_INTERLEAVED);
    snd_pcm_hw_params_set_format(pcm_handle, hw_params, SND_PCM_FORMAT_S16_LE);
    snd_pcm_hw_params_set_channels(pcm_handle, hw_params, channels);
    
    unsigned int actual_rate = sample_rate;
    snd_pcm_hw_params_set_rate_near(pcm_handle, hw_params, &actual_rate, 0);
    snd_pcm_hw_params_set_period_size_near(pcm_handle, hw_params, &frames_per_period, 0);
    
    snd_pcm_uframes_t buffer_size_frames = frames_per_period * 16;
    snd_pcm_hw_params_set_buffer_size_near(pcm_handle, hw_params, &buffer_size_frames);
    
    snd_pcm_hw_params(pcm_handle, hw_params);
    snd_pcm_prepare(pcm_handle);
    
    std::cout << "[ALSA] Configured: " << actual_rate << " Hz, " << channels 
              << " channels, " << frames_per_period << " frames/period" << std::endl;
    
    std::vector<int16_t> buffer(frames_per_period * channels);
    std::vector<int16_t> accumulated_buffer;
    
    while (keep_running) {
        err = snd_pcm_readi(pcm_handle, buffer.data(), frames_per_period);
        
        if (err == -EAGAIN) {
            usleep(1000);
            continue;
        }
        if (err == -EPIPE) {
            std::cerr << "[ALSA] Overrun occurred, recovering..." << std::endl;
            snd_pcm_prepare(pcm_handle);
            continue;
        }
        if (err < 0) {
            std::cerr << "[ALSA] Read error: " << snd_strerror(err) << std::endl;
            break;
        }
        
        if (err > 0 && err < (int)frames_per_period) {
            size_t samples_read = err * channels;
            accumulated_buffer.insert(accumulated_buffer.end(), 
                                     buffer.begin(), buffer.begin() + samples_read);
            if (accumulated_buffer.size() < frames_per_period * channels) {
                continue;
            }
            buffer = accumulated_buffer;
            accumulated_buffer.clear();
        }
        
        size_t mono_frames = buffer.size() / channels;
        std::vector<float> mono_float(mono_frames);
        std::vector<int16_t> mono_int16(mono_frames);
        
        for (size_t i = 0; i < mono_frames; ++i) {
            mono_int16[i] = buffer[i * channels];
            mono_float[i] = mono_int16[i] / 32768.0f;
        }
        
        {
            std::lock_guard<std::mutex> lock(vad_queue_mutex);
            vad_audio_queue.push(mono_float);
        }
        vad_queue_cv.notify_one();
        
        {
            std::lock_guard<std::mutex> lock(vosk_queue_mutex);
            vosk_audio_queue.push(mono_int16);
        }
        vosk_queue_cv.notify_one();
    }
    
    snd_pcm_drain(pcm_handle);
    snd_pcm_close(pcm_handle);
    std::cout << "[ALSA] Capture thread stopped" << std::endl;
}
// Extract text that comes AFTER the trigger phrase
std::string extract_command_after_trigger(const std::string& text) {
    std::string lower_text = text;
    std::transform(lower_text.begin(), lower_text.end(), lower_text.begin(), ::tolower);
    
    size_t pos = lower_text.find(TRIGGER_PHRASE);
    if (pos == std::string::npos) {
        return "";
    }
    
    // Get everything after "hey aurora"
    size_t start = pos + TRIGGER_PHRASE.length();
    
    // Skip whitespace
    while (start < text.length() && std::isspace(text[start])) {
        start++;
    }
    
    if (start >= text.length()) {
        return "";  // Nothing after trigger phrase
    }
    
    return text.substr(start);
}
// =============================================================================
// VAD Processing Thread
// =============================================================================
void vad_processing_thread(VadIterator& vad, JsonEventServer& volume_server) {
    std::cout << "[VAD] Processing thread started" << std::endl;
    int cnt = 0;
    
    while (keep_running) {
        if (vad_config_update_requested) {
            vad_config_update_requested = 0;
            getVadConfig(vad);
        }
        
        std::vector<float> audio_data;
        {
            std::unique_lock<std::mutex> lock(vad_queue_mutex);
            if (vad_audio_queue.empty()) {
                if (vad_queue_cv.wait_for(lock, std::chrono::milliseconds(100)) == std::cv_status::timeout) {
                    continue;
                }
                if (!keep_running || vad_audio_queue.empty()) {
                    continue;
                }
            }
            audio_data = vad_audio_queue.front();
            vad_audio_queue.pop();
        }
        
        if (!audio_data.empty()) {
            vad.audio_length_samples = audio_data.size();
            vad.predict(audio_data);
            
            if (cnt % 16 == 0) {
                std::vector<float> stereo_data(audio_data.size() * 2);
                for (size_t i = 0; i < audio_data.size(); ++i) {
                    stereo_data[i * 2] = audio_data[i];
                    stereo_data[i * 2 + 1] = audio_data[i];
                }
                volumeData_t vol = calculateVolumeStereo(stereo_data);
                volume_server.broadcast_volume(vol);
                cnt = 0;
            }
            cnt++;
        }
    }
    
    std::cout << "[VAD] Processing thread stopped" << std::endl;
}

// =============================================================================
// Extract text from JSON
// =============================================================================
std::string extractPartialText(const std::string& json) {
    size_t pos = json.find("\"partial\"");
    if (pos == std::string::npos) return "";
    
    size_t start = json.find(":", pos) + 1;
    while (start < json.length() && (json[start] == ' ' || json[start] == '\"')) start++;
    
    size_t end = start;
    while (end < json.length() && json[end] != '\"') end++;
    
    if (start < json.length() && end > start) {
        return json.substr(start, end - start);
    }
    return "";
}

std::string extractFinalText(const std::string& json) {
    size_t pos = json.find("\"text\"");
    if (pos == std::string::npos) return "";
    
    size_t start = json.find(":", pos) + 1;
    while (start < json.length() && (json[start] == ' ' || json[start] == '\"')) start++;
    
    size_t end = start;
    while (end < json.length() && json[end] != '\"') end++;
    
    if (start < json.length() && end > start) {
        return json.substr(start, end - start);
    }
    return "";
}

// =============================================================================
// NEW: Check if text contains trigger phrase
// =============================================================================
bool contains_trigger_phrase(const std::string& text) {
    std::string lower_text = text;
    std::transform(lower_text.begin(), lower_text.end(), lower_text.begin(), ::tolower);
    return (lower_text.find(TRIGGER_PHRASE) != std::string::npos);
}
void vosk_processing_thread(const std::string& model_path, 
                            const std::string& mode_file,
                            const std::string& grammar_file,
                            const std::string& commands_file) 
{
    std::cout << "[VOSK] Processing thread started (DYNAMIC MODE + HEY AURORA + COMMAND LOOKUP)" << std::endl;
    
    // Initialize Vosk
    vosk_set_log_level(0);
    VoskModel* model = vosk_model_new(model_path.c_str());
    if (!model) {
        std::cerr << "[VOSK] Failed to load model from: " << model_path << std::endl;
        return;
    }
    
    // Read initial mode
    int current_mode = read_vosk_mode(mode_file);
    std::vector<std::string> phrases;
    VoskRecognizer* recognizer = nullptr;
    
    // NEW: Trigger state for Mode 1
    bool command_triggered = false;
    std::chrono::steady_clock::time_point trigger_time;
    std::cout << "[GRAMMAR] Slot-filling system ready" << std::endl;
    
    // Create recognizer based on initial mode
    if (current_mode == 0) {
        std::cout << "[VOSK-MODE] Starting in MODE 0: Full vocabulary transcription" << std::endl;
        recognizer = vosk_recognizer_new(model, 16000.0);
        vosk_recognizer_set_max_alternatives(recognizer, 0);
    } else {
        std::cout << "[VOSK-MODE] Starting in MODE 1: Grammar-based command detection" << std::endl;
        phrases = load_phrases_from_file(grammar_file);
        std::string grammar = build_grammar_json(phrases);
        recognizer = vosk_recognizer_new_grm(model, 16000.0, grammar.c_str());
        std::cout << "[GRAMMAR] Loaded " << phrases.size() << " phrases from " << grammar_file << std::endl;
        std::cout << "[TRIGGER] Mode 1 requires \"" << TRIGGER_PHRASE << "\" before accepting commands" << std::endl;
    }
    
    if (!recognizer) {
        std::cerr << "[VOSK] Failed to create recognizer" << std::endl;
        vosk_model_free(model);
        return;
    }
    
    std::cout << "[VOSK] Model loaded successfully" << std::endl;
    
    std::string last_partial_text = "";
    
    while (keep_running) {
        // Check for mode update request (SIGUSR2)
        if (vosk_mode_update_requested) {
            vosk_mode_update_requested = 0;
            
            int new_mode = read_vosk_mode(mode_file);
            
            if (new_mode != current_mode) {
                std::cout << "\n[VOSK-MODE] *** MODE SWITCH REQUESTED: " 
                          << current_mode << " -> " << new_mode << " ***" << std::endl;
                
                // Free old recognizer
                if (recognizer) {
                    vosk_recognizer_free(recognizer);
                    recognizer = nullptr;
                }
                
                // Reset trigger state on mode switch
                command_triggered = false;
                
                // Create new recognizer based on mode
                if (new_mode == 0) {
                    std::cout << "[VOSK-MODE] Switching to MODE 0: Full vocabulary transcription" << std::endl;
                    recognizer = vosk_recognizer_new(model, 16000.0);
                    vosk_recognizer_set_max_alternatives(recognizer, 0);
                } else {
                    std::cout << "[VOSK-MODE] Switching to MODE 1: Grammar-based command detection" << std::endl;
                    phrases = load_phrases_from_file(grammar_file);
                    std::string grammar = build_grammar_json(phrases);
                    recognizer = vosk_recognizer_new_grm(model, 16000.0, grammar.c_str());
                    std::cout << "[GRAMMAR] Loaded " << phrases.size() << " phrases:" << std::endl;
                    for (const auto& phrase : phrases) {
                        std::cout << "  - \"" << phrase << "\"" << std::endl;
                    }
                    std::cout << "[TRIGGER] Mode 1 requires \"" << TRIGGER_PHRASE << "\" before accepting commands" << std::endl;
                }
                
                if (!recognizer) {
                    std::cerr << "[VOSK-MODE] ERROR: Failed to create new recognizer!" << std::endl;
                    break;
                }
                
                current_mode = new_mode;
                last_partial_text = "";
                
                std::cout << "[VOSK-MODE] *** MODE SWITCH COMPLETE ***\n" << std::endl;
            } else {
                std::cout << "[VOSK-MODE] Mode unchanged (already in mode " << current_mode << ")" << std::endl;
            }
        }
        
        // Check for commands reload
        if (commands_update_requested) {
            commands_update_requested = 0;
            std::cout << "[GRAMMAR] Grammar system does not require file reload" << std::endl;
        }
        
        // Check trigger timeout in Mode 1
        if (current_mode == 1 && command_triggered) {
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - trigger_time).count();
            
            if (elapsed >= TRIGGER_TIMEOUT_MS) {
                command_triggered = false;
                std::cout << "[TRIGGER] *** Trigger window expired (5s timeout) - Waiting for \"" 
                          << TRIGGER_PHRASE << "\" again ***" << std::endl;
            }
        }
        
        std::vector<int16_t> audio_data;
        {
            std::unique_lock<std::mutex> lock(vosk_queue_mutex);
            if (vosk_audio_queue.empty()) {
                if (vosk_queue_cv.wait_for(lock, std::chrono::milliseconds(100)) == std::cv_status::timeout) {
                    continue;
                }
                if (!keep_running || vosk_audio_queue.empty()) {
                    continue;
                }
            }
            audio_data = vosk_audio_queue.front();
            vosk_audio_queue.pop();
        }
        
        if (!audio_data.empty()) {
            int final = vosk_recognizer_accept_waveform(
                recognizer,
                (const char*)audio_data.data(),
                audio_data.size() * sizeof(int16_t)
            );
            
            if (final) {
                std::string result = vosk_recognizer_result(recognizer);
                std::cout << "[VOSK] FINAL (Mode " << current_mode << "): " << result << std::endl;
                
                if (g_vosk_final_server) {
                    g_vosk_final_server->broadcast_message(result);
                }
                
                std::string final_text = extractFinalText(result);
                if (!final_text.empty()) {
                    if (current_mode == 1) {
                        // MODE 1: Grammar-based command mode with trigger
                        
                        // 1. Check if trigger phrase is in the final text
                        if (contains_trigger_phrase(final_text)) {
                            // Check if there's a command AFTER "hey aurora"
                            std::string command_after_trigger = extract_command_after_trigger(final_text);
                            
                            if (!command_after_trigger.empty()) {
                                // ONE-STEP: "hey aurora turn on the front led"
                                std::cout << "[TRIGGER] One-step command detected: \"" << command_after_trigger << "\"" << std::endl;
                                
                                std::string cmd = process_command_with_grammar_slots(command_after_trigger);
                                if (!cmd.empty()) {
                                    if (send_command_to_server(cmd, "127.0.0.1", COMMAND_PORT)) {
                                        std::cout << "[SENT] " << cmd << std::endl;
                                    }
                                } else {
                                    std::cout << "[COMMAND] ✗ No match for one-step command: \"" << command_after_trigger << "\"" << std::endl;
                                }
                                // Do NOT open trigger window - execute and close
                            } else {
                                // TWO-STEP: Just "hey aurora" - open trigger window
                                command_triggered = true;
                                trigger_time = std::chrono::steady_clock::now();
                                std::cout << "[TRIGGER] *** \"" << TRIGGER_PHRASE 
                                          << "\" DETECTED! Command window OPEN for " 
                                          << TRIGGER_TIMEOUT_MS << "ms ***" << std::endl;
                            }
                        }
                        // 2. Check if we're in triggered state
                        else if (command_triggered) {
                            // SCENARIO: Trigger active, User says "<Command>"
                            std::string cmd = process_command_with_grammar_slots(final_text);
                            if (!cmd.empty()) {
                                if (send_command_to_server(cmd, "127.0.0.1", COMMAND_PORT)) {
                                    std::cout << "[SENT] " << cmd << std::endl;
                                    command_triggered = false;
                                }
                            } else {
                                std::cout << "[FAIL] No match during trigger: " << final_text << std::endl;
                            }
                            
                            // Close trigger window after processing command
                            command_triggered = false;
                        } 
                        // 3. Not triggered
                        else {
                            std::cout << "[TRIGGER] Grammar phrase detected but NOT triggered: \"" 
                                      << final_text << "\" (waiting for \"" << TRIGGER_PHRASE << "\")" << std::endl;
                        }
                    } else {
                        // MODE 0: Full transcription mode
                        std::cout << "[TRANSCRIPTION] " << final_text << std::endl;
                    }
                }
                last_partial_text = "";
            } else {
                std::string partial = vosk_recognizer_partial_result(recognizer);
                std::string current_partial_text = extractPartialText(partial);

                // Send partial if changed
                if (!current_partial_text.empty() && current_partial_text != last_partial_text) {
                    std::cout << "[VOSK] PARTIAL (Mode " << current_mode << "): " << partial << std::endl;
                    if (g_vosk_partial_server) {
                        g_vosk_partial_server->broadcast_message(partial);
                    }
                    
                    last_partial_text = current_partial_text;
                }
            }
        }
    }
    
    // Get final result
    std::string final_result = vosk_recognizer_final_result(recognizer);
    std::cout << "[VOSK] FINAL RESULT: " << final_result << std::flush;

    if (!final_result.empty() && final_result.find("\"text\" : \"\"") == std::string::npos) {
        if (g_vosk_final_server) {
            g_vosk_final_server->broadcast_message(final_result);
        }
    }
    
    vosk_recognizer_free(recognizer);
    vosk_model_free(model);
    
    std::cout << "[VOSK] Processing thread stopped" << std::endl;
}

// =============================================================================
// Main Function
// =============================================================================
int main(int argc, char* argv[]) {
    std::cout << "=== Dynamic Mode VAD + Vosk Speech Recognition System ===" << std::endl;
    std::cout << "=== With 'Hey Aurora' Trigger + Command Lookup ===" << std::endl;
    
Config cfg = parse_args(argc, argv);
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "Configuration:" << std::endl;
    std::cout << "  ALSA Device: " << cfg.alsa_device << std::endl;
    std::cout << "  VAD Model: " << cfg.vad_model_path << std::endl;
    std::cout << "  Vosk Model: " << cfg.vosk_model_path << std::endl;
    std::cout << "  Mode File: " << cfg.mode_file << std::endl;
    std::cout << "  Grammar File: " << cfg.grammar_file << std::endl;
    std::cout << "  VAD Event Port: " << cfg.vad_event_port << std::endl;
    std::cout << "  Volume Port: " << cfg.volume_port << std::endl;
    std::cout << "  Vosk Finals TCP: " << cfg.vosk_tcp_host << ":" << cfg.vosk_final_port << std::endl;
    std::cout << "  Vosk Partials TCP: " << cfg.vosk_tcp_host << ":" << cfg.vosk_partial_port << std::endl;

    signal(SIGINT, signal_handler);
    signal(SIGUSR2, unified_config_handler);
    
    // Write PID file
    std::ofstream pidfile("/tmp/combined_vad_vosk.pid");
    pidfile << getpid() << std::endl;
    pidfile.close();
    printf("PID: %d\n", getpid());
    
    // Create mode file if it doesn't exist
    std::ifstream mode_check(cfg.mode_file);
    if (!mode_check.is_open()) {
        std::ofstream mode_create(cfg.mode_file);
        mode_create << "0" << std::endl;
        mode_create.close();
        std::cout << "[VOSK-MODE] Created mode file: " << cfg.mode_file << " (default: 0)" << std::endl;
    } else {
        mode_check.close();
    }
    

    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "Configuration:" << std::endl;
    std::cout << "  ALSA Device: " << cfg.alsa_device << std::endl;
    std::cout << "  VAD Model: " << cfg.vad_model_path << std::endl;
    std::cout << "  Vosk Model: " << cfg.vosk_model_path << std::endl;
    std::cout << "  Mode File: " << cfg.mode_file << std::endl;
    std::cout << "  Grammar File: " << cfg.grammar_file << std::endl;
    std::cout << "  Commands File: " << cfg.commands_file << std::endl;
    std::cout << "  VAD Event Port: " << cfg.vad_event_port << std::endl;
    std::cout << "  Volume Port: " << cfg.volume_port << std::endl;
    std::cout << "  Commands TCP Client: 127.0.0.1:6975" << std::endl;
    std::cout << "  Vosk Finals TCP: " << cfg.vosk_tcp_host << ":" << cfg.vosk_final_port << std::endl;
    std::cout << "  Vosk Partials TCP: " << cfg.vosk_tcp_host << ":" << cfg.vosk_partial_port << std::endl;
    std::cout << std::string(80, '=') << "\n" << std::endl;

    signal(SIGINT, signal_handler);
    signal(SIGUSR2, unified_config_handler);
    // Initialize VAD
    VadIterator vad(cfg.vad_model_path);
    vad.reset();
    getVadConfig(vad);
    
    // Initialize JSON Event Servers
    JsonEventServer vad_event_server(cfg.vad_event_port);
    JsonEventServer volume_server(cfg.volume_port);
    
    g_vad_event_server = &vad_event_server;
    g_volume_server = &volume_server;
    
    if (!vad_event_server.start()) {
        std::cerr << "[VAD] Failed to start event server on port " 
                  << cfg.vad_event_port << std::endl;
        return 1;
    }
    std::cout << "[VAD] Event server started on port " << cfg.vad_event_port << std::endl;
    
    if (!volume_server.start()) {
        std::cerr << "[VAD] Failed to start volume server on port " 
                  << cfg.volume_port << std::endl;
        return 1;
    }
    std::cout << "[VAD] Volume server started on port " << cfg.volume_port << std::endl;
    
    JsonEventServer vosk_final_server(cfg.vosk_final_port);
    JsonEventServer vosk_partial_server(cfg.vosk_partial_port);
    
    g_vosk_final_server = &vosk_final_server;
    g_vosk_partial_server = &vosk_partial_server;
    
    if (!vosk_final_server.start()) {
        std::cerr << "[VOSK] Failed to start finals server on port " 
                  << cfg.vosk_final_port << std::endl;
        return 1;
    }
    std::cout << "[VOSK] Finals server started on port " << cfg.vosk_final_port << std::endl;
    
    if (!vosk_partial_server.start()) {
        std::cerr << "[VOSK] Failed to start partials server on port " 
                  << cfg.vosk_partial_port << std::endl;
        return 1;
    }
    std::cout << "[VOSK] Partials server started on port " << cfg.vosk_partial_port << std::endl;

    std::cout << "\nStarting processing threads..." << std::endl;
    
    std::thread alsa_thread(alsa_capture_thread, cfg.alsa_device);
    std::thread vad_thread(vad_processing_thread, std::ref(vad), std::ref(volume_server));
    std::thread vosk_thread(vosk_processing_thread, cfg.vosk_model_path, 
                           cfg.mode_file, cfg.grammar_file, cfg.commands_file);

    std::cout << "\n" << std::string(80, '#') << std::endl;
    std::cout << "SPEECH RECOGNITION SYSTEM RUNNING WITH COMMAND LOOKUP" << std::endl;
    std::cout << "Press Ctrl+C to stop." << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    std::cout << "  VAD events: port " << cfg.vad_event_port << std::endl;
    std::cout << "  Volume data: port " << cfg.volume_port << std::endl;
    std::cout << "  Vosk FINALS: " << cfg.vosk_tcp_host << ":" << cfg.vosk_final_port << std::endl;
    std::cout << "  Vosk PARTIALS: " << cfg.vosk_tcp_host << ":" << cfg.vosk_partial_port << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    std::cout << "MODE SWITCHING:" << std::endl;
    std::cout << "  Mode file: " << cfg.mode_file << std::endl;
    std::cout << "  Grammar file: " << cfg.grammar_file << std::endl;
    std::cout << "  Commands file: " << cfg.commands_file << std::endl;
    std::cout << "  Current PID: " << getpid() << std::endl;
    std::cout << "  To update configuration:" << std::endl;
    std::cout << "    1. Edit " << cfg.mode_file << " (0=transcription, 1=grammar)" << std::endl;
    std::cout << "    2. Edit " << cfg.commands_file << " (phrase=command mappings)" << std::endl;
    std::cout << "    3. Edit VAD config files (word_gate settings)" << std::endl;
    std::cout << "    4. Send signal: kill -SIGUSR2 " << getpid() << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    std::cout << "COMMAND LOOKUP (Mode 1 only):" << std::endl;
    std::cout << "  Commands file: " << cfg.commands_file << std::endl;
    std::cout << "  Matching: Fuzzy (case-insensitive, removes filler words)" << std::endl;
    std::cout << "  Output: Raw command string to port " << std::endl;
    std::cout << "  Unmatched phrases: Logged and ignored" << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    std::cout << "HEY AURORA TRIGGER (Mode 1 only):" << std::endl;
    std::cout << "  Trigger phrase: \"" << TRIGGER_PHRASE << "\" (hardcoded)" << std::endl;
    std::cout << "  Trigger timeout: " << TRIGGER_TIMEOUT_MS << " ms (5 seconds)" << std::endl;
    std::cout << "  Behavior:" << std::endl;
    std::cout << "    1. Say \"" << TRIGGER_PHRASE << "\" to activate" << std::endl;
    std::cout << "    2. Commands from " << cfg.grammar_file << " will be matched to " 
              << cfg.commands_file << std::endl;
    std::cout << "    3. Matched commands sent to port for 5 seconds" << std::endl;
    std::cout << "    4. After timeout, say \"" << TRIGGER_PHRASE << "\" again" << std::endl;
    std::cout << std::string(80, '#') << std::endl << std::endl;
    
    // Wait for threads to finish
    std::cout << "Waiting for threads..." << std::endl;
    
    if (alsa_thread.joinable()) {
        alsa_thread.join();
        std::cout << "ALSA thread joined" << std::endl;
    }
    
    if (vad_thread.joinable()) {
        vad_thread.join();
        std::cout << "VAD thread joined" << std::endl;
    }
    
    if (vosk_thread.joinable()) {
        vosk_thread.join();
        std::cout << "Vosk thread joined" << std::endl;
    }
    
    // Cleanup
    std::cout << "\nStopping servers..." << std::endl;
    vad_event_server.stop();
    volume_server.stop();
    vosk_final_server.stop();
    vosk_partial_server.stop();

    std::cout << "System stopped cleanly." << std::endl;
    return 0;
}