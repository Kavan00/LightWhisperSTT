# LightWhisperSTT

A lightweight near real-time offline speech-to-text for low-resource systems (e.g. Raspberry Pi) using OpenAI's Whisper models with [pywhispercpp](https://github.com/absadiki/pywhispercpp) using [whisper.cpp](https://github.com/ggml-org/whisper.cpp).

## Features

- **Real-time transcription**: Continuously records audio and transcribes speech in configurable time windows
- **Multi-threaded processing**: Separate threads for audio recording and transcription to prevent blocking
- **Flexible configuration**: Customizable model size, language, and processing parameters
- **Callback support**: Optional callback function for handling transcriptions as they arrive
- **Buffer management**: Efficient circular buffer for audio data with automatic overflow handling

## Memory Usage

Approximate RAM consumption by model size (tested on macOS):

| Model    | RAM Usage |
|----------|-----------|
| base     | ~0.2GB    |
| small    | ~0.6GB    |
| medium   | ~1.9GB    |
| large-v3 | ~3.3GB    |

## Installation

```bash
git clone https://github.com/Kavan00/LightWhisperSTT
cd LightWhisperSTT
pip install sounddevice numpy pywhispercpp
python3 example.py
```

**Note**: You may need to install additional system dependencies for audio recording depending on your platform.

## Quick Start

### Basic Usage

```python
from LightWhisperSTT.LightWhisperSTT import LightWhisperSTT

# Create STT instance with default settings
stt = LightWhisperSTT()

# Start transcription (blocks until stopped)
try:
    stt.start()
except KeyboardInterrupt:
    stt.stop()
    print("Transcription stopped")
```

### With Custom Configuration

```python
def handle_transcription(text):
    print(f"Transcribed: {text['text']}")
    # Your custom processing here

stt = LightWhisperSTT(
    model_name="small",           # See available models at: https://absadiki.github.io/pywhispercpp/#pywhispercpp.constants.AVAILABLE_MODELS
    language="en",                # Language code or "auto" for detection
    window_seconds=5,             # Audio chunk duration
    print_debug=True,                   # Enable debug output
    on_transcription=handle_transcription  # Custom callback
)

stt.start()
```

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name` | "medium" | Whisper model size (see [available models](https://absadiki.github.io/pywhispercpp/#pywhispercpp.constants.AVAILABLE_MODELS)) |
| `language` | "auto" | Target language code or "auto" for automatic detection |
| `chunk_size` | 4096 | Audio buffer chunk size |
| `window_seconds` | 10 | Duration of audio segments to transcribe |
| `model_threads` | 4 | Number of threads for Whisper model processing |
| `print_debug` | False | Enable debug output |
| `on_transcription` | None | Callback function called for each transcription |

## Methods

### `start()`
Begins the recording and transcription process. This method blocks until `stop()` is called or the process is interrupted.

### `stop()`
Stops the recording and transcription process.

### `get_transcripts()`
Returns a copy of all transcribed segments as a list of dictionaries with `index` and `text` fields.

### `get_model_languages()`
Returns supported languages of the choosen model.

## How It Works

1. **Audio Recording**: Continuously records audio in chunks using sounddevice
2. **Buffering**: Maintains a circular buffer of the most recent audio data
3. **Segmentation**: Every `window_seconds`, extracts an audio segment for processing
4. **Transcription**: Processes audio segments through Whisper model using worker threads
5. **Output**: Prints transcriptions or calls custom callback function

## Performance Notes

- **Model Size**: Larger models (medium, large) provide better accuracy but require more processing time and memory
- **Window Size**: Shorter windows provide faster response but may miss context; longer windows improve accuracy but increase latency
- **Threading**: The system uses separate threads for recording and transcription to maintain real-time performance

## Example Output

```
Hello, this is a test of the speech recognition system.
I'm speaking into my microphone right now.
The transcription should appear in real time.
```

### Model Loading
The first run may take time as Whisper models are downloaded and loaded. Subsequent runs will be faster.

## Requirements

- Python 3.7+
- sounddevice
- numpy
- pywhispercpp
- Working microphone
- Sufficient RAM for chosen Whisper model

## License

This project is licensed under the same license as [pywhispercpp](https://github.com/absadiki/pywhispercpp/blob/main/LICENSE)/[whisper.cpp](https://github.com/ggml-org/whisper.cpp/blob/master/LICENSE) (MIT License).