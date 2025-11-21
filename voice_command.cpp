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
#include <alsa/asoundlib.h>

// Vosk
#include <vosk_api.h>

// Networking
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <getopt.h>

// Local headers (Assumed to exist in your project)
#include "JsonEventServer.h"
#include "wav.h"

// =============================================================================
// Global Variables and Signal Handling
// =============================================================================
std::atomic<bool> keep_running{true};
std::atomic<bool> signal_received{false};
volatile sig_atomic_t vad_config_update_requested = 0; // Legacy support
volatile sig_atomic_t vosk_mode_update_requested = 0;
volatile sig_atomic_t commands_update_requested = 0;

static JsonEventServer *g_vosk_final_server = nullptr;
static JsonEventServer *g_vosk_partial_server = nullptr;

std::condition_variable vosk_queue_cv;
std::mutex vosk_queue_mutex;
std::queue<std::vector<int16_t>> vosk_audio_queue;

std::string g_last_hostname;
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

        vosk_queue_cv.notify_all();
        
        if (g_vosk_final_server) {
            g_vosk_final_server->stop();
        }
        if (g_vosk_partial_server) {
            g_vosk_partial_server->stop();
        }
    }
}

void unified_config_handler(int sig) {
    vosk_mode_update_requested = 1;
    commands_update_requested = 1;
    // Re-arm signal
    signal(SIGUSR2, unified_config_handler);
}

// =============================================================================
// Configuration
// =============================================================================
struct Config {
    std::string alsa_device = "hw:0,7";
    std::string vosk_model_path = "models/vosk_model";
    std::string vosk_tcp_host = "127.0.0.1";
    int vosk_final_port = 9999;
    int vosk_partial_port = 9998;
    std::string mode_file = "mode.txt";
    std::string grammar_file = "keywords.txt";
    std::string commands_file = "commands.txt";
    int command_port = 6975;
};

Config parse_args(int argc, char *argv[]) {
    Config cfg;
    
    static struct option long_opts[] = {
        {"device", required_argument, nullptr, 'd'},
        {"vosk-model", required_argument, nullptr, 'k'},
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
    while ((opt = getopt_long(argc, argv, "d:k:h:f:a:m:g:c:p:", long_opts, nullptr)) != -1) {
        switch (opt) {
            case 'd': cfg.alsa_device = optarg; break;
            case 'k': cfg.vosk_model_path = optarg; break;
            case 'h': cfg.vosk_tcp_host = optarg; break;
            case 'f': cfg.vosk_final_port = std::stoi(optarg); break;
            case 'a': cfg.vosk_partial_port = std::stoi(optarg); break;
            case 'm': cfg.mode_file = optarg; break;
            case 'g': cfg.grammar_file = optarg; break;
            case 'c': cfg.commands_file = optarg; break;
            case 'p': cfg.command_port = std::stoi(optarg); break;
            default:
                std::cerr << "Usage: " << argv[0] << " [options]\n"
                          << "  -d, --device <dev>            ALSA device (default: hw:0,7)\n"
                          << "  -k, --vosk-model <path>       Vosk model path\n"
                          << "  -h, --vosk-host <ip>          Vosk TCP host\n"
                          << "  -f, --vosk-final-port <port>  Finals port\n"
                          << "  -a, --vosk-partial-port <port> Partials port\n"
                          << "  -m, --mode-file <path>        Mode file\n"
                          << "  -g, --grammar-file <path>     Grammar file\n"
                          << "  -c, --commands-file <path>    Commands file\n"
                          << "  -p, --command-port <port>     Port to send matched commands\n";
                exit(1);
        }
    }
    return cfg;
}

// =============================================================================
// Grammar-Based Phrase Loading
// =============================================================================
std::vector<std::string> load_phrases_from_file(const std::string& filepath) {
    std::vector<std::string> phrases;
    std::ifstream file(filepath);
    
    if (!file.is_open()) {
        std::cerr << "[GRAMMAR] Failed to open " << filepath << ". Using defaults.\n";
        return {
            "on", "off", "down the", "up the", "brightness",
            "set led to color", "green", "magenta", "cyan", "orange", "white", "blue", "red",
            "turn off the screen", "hey aurora", "change the", "from"
        };
    }
    
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        
        std::string phrase = line;
        // Trim trailing whitespace
        while (!phrase.empty() && (std::isspace(phrase.back()))) {
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
bool send_command_to_server(const std::string& command, const std::string& host, int port) {
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
    
    // Send command (add newline)
    std::string msg = command + "\n";
    // Use MSG_NOSIGNAL to prevent SIGPIPE if server disconnects during send
    ssize_t sent = send(sock, msg.c_str(), msg.length(), MSG_NOSIGNAL);
    
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
int read_vosk_mode(const std::string& mode_file) {
    std::ifstream file(mode_file);
    if (!file.is_open()) {
        return 0; // Default mode
    }
    
    std::string line;
    std::getline(file, line);
    file.close();
    
    // Trim whitespace
    line.erase(std::remove_if(line.begin(), line.end(), ::isspace), line.end());
    
    int mode = std::atoi(line.c_str());
    if (mode != 0 && mode != 1) {
        return 0;
    }
    
    return mode;
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
    
    if ((err = snd_pcm_open(&pcm_handle, device_name.c_str(), SND_PCM_STREAM_CAPTURE, 0)) < 0) {
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
    
    snd_pcm_hw_params(pcm_handle, hw_params);
    snd_pcm_prepare(pcm_handle);
    
    std::vector<int16_t> buffer(frames_per_period * channels);
    std::vector<int16_t> accumulated_buffer;
    
    while (keep_running) {
        err = snd_pcm_readi(pcm_handle, buffer.data(), frames_per_period);
        
        if (err == -EAGAIN) { usleep(1000); continue; }
        if (err == -EPIPE) { snd_pcm_prepare(pcm_handle); continue; }
        if (err < 0) break;
        
        if (err > 0) {
            // Simple accumulate if we get partial reads (rare with readi but good to handle)
            size_t samples_read = err * channels;
            accumulated_buffer.insert(accumulated_buffer.end(), buffer.begin(), buffer.begin() + samples_read);
            
            if (accumulated_buffer.size() >= frames_per_period * channels) {
                // Process mono conversion
                size_t mono_frames = accumulated_buffer.size() / channels;
                std::vector<int16_t> mono_int16(mono_frames);
                
                for (size_t i = 0; i < mono_frames; ++i) {
                    mono_int16[i] = accumulated_buffer[i * channels];
                }
                
                {
                    std::lock_guard<std::mutex> lock(vosk_queue_mutex);
                    vosk_audio_queue.push(mono_int16);
                }
                vosk_queue_cv.notify_one();
                accumulated_buffer.clear();
            }
        }
    }
    
    snd_pcm_drain(pcm_handle);
    snd_pcm_close(pcm_handle);
    std::cout << "[ALSA] Capture thread stopped" << std::endl;
}

// =============================================================================
// String Extraction Helpers
// =============================================================================

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

bool contains_trigger_phrase(const std::string& text) {
    std::string lower_text = text;
    std::transform(lower_text.begin(), lower_text.end(), lower_text.begin(), ::tolower);
    return (lower_text.find(TRIGGER_PHRASE) != std::string::npos);
}

// =============================================================================
// Vosk Processing Thread
// =============================================================================
void vosk_processing_thread(const std::string& model_path, 
                            const std::string& mode_file,
                            const std::string& grammar_file,
                            const std::string& commands_file,
                            int command_port) 
{
    std::cout << "[VOSK] Processing thread started" << std::endl;
    
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
    
    // State for Mode 1
    bool command_triggered = false;
    std::chrono::steady_clock::time_point trigger_time;

    // No initialization needed for grammar+slots system
    std::cout << "[GRAMMAR] Slot-filling system ready" << std::endl;    
    
    // Initialize Recognizer based on Mode
    auto init_recognizer = [&](int mode) {
        if (recognizer) vosk_recognizer_free(recognizer);
        
        if (mode == 0) {
            std::cout << "[VOSK-MODE] MODE 0: Full transcription" << std::endl;
            recognizer = vosk_recognizer_new(model, 16000.0);
            vosk_recognizer_set_max_alternatives(recognizer, 0);
        } else {
            std::cout << "[VOSK-MODE] MODE 1: Grammar Command Detection" << std::endl;
            phrases = load_phrases_from_file(grammar_file);
            std::string grammar = build_grammar_json(phrases);
            recognizer = vosk_recognizer_new_grm(model, 16000.0, grammar.c_str());
            std::cout << "[TRIGGER] Waiting for \"" << TRIGGER_PHRASE << "\"" << std::endl;
        }
    };
    
    init_recognizer(current_mode);
    
    if (!recognizer) {
        std::cerr << "[VOSK] Failed to create recognizer" << std::endl;
        vosk_model_free(model);
        return;
    }
    
    std::string last_partial_text = "";
    
    while (keep_running) {
        // Check for mode update
        if (vosk_mode_update_requested) {
            vosk_mode_update_requested = 0;
            int new_mode = read_vosk_mode(mode_file);
            if (new_mode != current_mode) {
                current_mode = new_mode;
                command_triggered = false;
                init_recognizer(current_mode);
            }
        }

        // Check for commands reload (not needed for grammar system, but kept for signal compatibility)
        if (commands_update_requested) {
            commands_update_requested = 0;
            std::cout << "[GRAMMAR] Grammar system does not require file reload" << std::endl;
        }
        
        // Check trigger timeout
        if (current_mode == 1 && command_triggered) {
            auto now = std::chrono::steady_clock::now();
            if (std::chrono::duration_cast<std::chrono::milliseconds>(now - trigger_time).count() >= TRIGGER_TIMEOUT_MS) {
                command_triggered = false;
                std::cout << "[TRIGGER] Timeout. Waiting for \"" << TRIGGER_PHRASE << "\" again." << std::endl;
            }
        }
        
        std::vector<int16_t> audio_data;
        {
            std::unique_lock<std::mutex> lock(vosk_queue_mutex);
            if (vosk_audio_queue.empty()) {
                vosk_queue_cv.wait_for(lock, std::chrono::milliseconds(100));
            }
            if (!vosk_audio_queue.empty()) {
                audio_data = vosk_audio_queue.front();
                vosk_audio_queue.pop();
            }
        }
        
        if (audio_data.empty()) continue;
        
        if (vosk_recognizer_accept_waveform(recognizer, (const char*)audio_data.data(), audio_data.size() * sizeof(int16_t))) {
            std::string result = vosk_recognizer_result(recognizer);
            std::cout << "[VOSK] FINAL: " << result << std::endl;
            
            if (g_vosk_final_server) g_vosk_final_server->broadcast_message(result);
            
            std::string final_text = extractFinalText(result);
            if (!final_text.empty()) {
                if (current_mode == 1) {
                    // --- MODE 1 LOGIC ---
                    
                    if (contains_trigger_phrase(final_text)) {
                        std::string command_after = extract_command_after_trigger(final_text);
                        
                        if (!command_after.empty()) {
                            // SCENARIO: "Hey Aurora <Command>" (One-shot)
                            std::cout << "[TRIGGER] One-step: \"" << command_after << "\"" << std::endl;
                            std::string cmd = process_command_with_grammar_slots(command_after);
                            if (!cmd.empty()) {
                                if (send_command_to_server(cmd, "127.0.0.1", command_port))
                                    std::cout << "[SENT] " << cmd << std::endl;
                            } else {
                                std::cout << "[FAIL] No match for: " << command_after << std::endl;
                            }
                                                    } else {
                            // SCENARIO: "Hey Aurora" (Enter Trigger State)
                            command_triggered = true;
                            trigger_time = std::chrono::steady_clock::now();
                            std::cout << "[TRIGGER] Window OPEN (5s)" << std::endl;
                        }
                    } else if (command_triggered) {
                        // SCENARIO: Trigger active, User says "<Command>"
// SCENARIO: Trigger active, User says "<Command>"
                        std::string cmd = process_command_with_grammar_slots(final_text);
                        if (!cmd.empty()) {
                            if (send_command_to_server(cmd, "127.0.0.1", command_port)) {
                                std::cout << "[SENT] " << cmd << std::endl;
                                command_triggered = false;
                            }
                        } else {
                            std::cout << "[FAIL] No match during trigger: " << final_text << std::endl;
                        }
                                            }
                } else {
                    // --- MODE 0 LOGIC ---
                    std::cout << "[TRANSCRIPT] " << final_text << std::endl;
                }
            }
            last_partial_text = "";
        } else {
            // Handle Partials
            std::string partial = vosk_recognizer_partial_result(recognizer);
            std::string current_partial = extractPartialText(partial);
            if (!current_partial.empty() && current_partial != last_partial_text) {
                // std::cout << "[PARTIAL] " << current_partial << std::endl; // Optional verbose logging
                if (g_vosk_partial_server) g_vosk_partial_server->broadcast_message(partial);
                last_partial_text = current_partial;
            }
        }
    }
    
    if (recognizer) vosk_recognizer_free(recognizer);
    if (model) vosk_model_free(model);
    std::cout << "[VOSK] Processing thread stopped" << std::endl;
}

// =============================================================================
// Main Function
// =============================================================================
int main(int argc, char* argv[]) {
    Config cfg = parse_args(argc, argv);
    
    std::cout << "=== Dynamic Vosk Speech System ===" << std::endl;
    std::cout << "  Device: " << cfg.alsa_device << std::endl;
    std::cout << "  Model: " << cfg.vosk_model_path << std::endl;
    std::cout << "  Command Port: " << cfg.command_port << std::endl;
    
    // Create default files if missing
    {
        std::ofstream f(cfg.mode_file, std::ios::app); 
        if (f.tellp() == 0) f << "0" << std::endl;
    }
    // Grammar+Slots system doesn't need commands file
std::cout << "[INIT] Using grammar+slot-filling (no commands file needed)" << std::endl;
    signal(SIGINT, signal_handler);
    signal(SIGUSR2, unified_config_handler);

    JsonEventServer vosk_final_server(cfg.vosk_final_port);
    JsonEventServer vosk_partial_server(cfg.vosk_partial_port);
    
    g_vosk_final_server = &vosk_final_server;
    g_vosk_partial_server = &vosk_partial_server;
    
    if (!vosk_final_server.start() || !vosk_partial_server.start()) {
        std::cerr << "Failed to start TCP servers." << std::endl;
        return 1;
    }

    std::thread alsa_thread(alsa_capture_thread, cfg.alsa_device);
    std::thread vosk_thread(vosk_processing_thread, cfg.vosk_model_path, 
                           cfg.mode_file, cfg.grammar_file, cfg.commands_file, cfg.command_port);
    
    if (alsa_thread.joinable()) alsa_thread.join();
    if (vosk_thread.joinable()) vosk_thread.join();
    
    vosk_final_server.stop();
    vosk_partial_server.stop();
    return 0;
}