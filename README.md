# LightWhisperSTT

A lightweight near real-time offline speech-to-text for low-resource systems (e.g. Raspberry Pi) using OpenAI's Whisper models with [pywhispercpp](https://github.com/absadiki/pywhispercpp) a python binding for [whisper.cpp](https://github.com/ggml-org/whisper.cpp). And [WebRTCVad](https://github.com/wiseman/py-webrtcvad) for Voice Activity Detection.

## Features

- **Voice Activity Detection**: Automatically detects speech start/end using WebRTC VAD and RMS thresholding
- **Near Real-time transcription**: Continuously records audio and transcribes speech segments as they occur
- **Smart buffering**: Pre-buffer system captures speech from the beginning, post-silence detection ensures complete phrases
- **Multi-threaded processing**: Separate threads for audio recording and transcription to prevent blocking
- **Flexible configuration**: Customizable model size, language, and voice detection parameters
- **Callback support**: Optional callback function for handling transcriptions as they arrive
- **Efficient memory usage**: Circular buffers with automatic overflow handling

## Voice Activity Detection

The system uses voice activity detection to trigger transcription only when speech is detected:

- **WebRTC VAD**: Industry-standard voice activity detection
- **RMS threshold**: Minimum loudness level to consider as potential speech
- **Speech ratio thresholds**: Configurable start/end detection sensitivity
- **Pre-buffer**: Captures 1 second before speech detection to avoid cutting off words
- **Post-silence**: Waits 0.3 seconds after speech ends to ensure complete phrases

## Memory Usage

Approximate RAM consumption by model size (tested on macOS15):

| Model    | RAM Usage |
|----------|-----------|
| base     | ~0.2GB    |
| small    | ~0.6GB    |
| medium   | ~1.9GB    |
| large-v3 | ~3.3GB    |

## Installation

```bash
pip install lightwhisperstt
```

**Note**: You may need to install additional system dependencies for audio recording depending on your platform.

## Quick Start

### Basic Usage

```python
from lightwhisperstt.core import LightWhisperSTT

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
    window_seconds=15,            # Maximum audio buffer duration
    print_debug=True,             # Enable debug output
    on_transcription=handle_transcription,  # Custom callback
    rms_threshold=0.015,          # Adjust voice sensitivity
    start_threshold=0.4,          # Speech detection sensitivity
    end_threshold=0.1             # End-of-speech sensitivity
)

stt.start()
```

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name` | "medium" | Whisper model (see [available models](https://absadiki.github.io/pywhispercpp/#pywhispercpp.constants.AVAILABLE_MODELS)) |
| `language` | "auto" | Target language code or "auto" for automatic detection |
| `chunk_size` | 4096 | Audio buffer chunk size |
| `window_seconds` | 15 | Maximum duration of audio buffer before forced flush |
| `model_threads` | 4 | Number of threads for Whisper model processing |
| `print_debug` | False | Enable debug output |
| `on_transcription` | None | Callback function called for each transcription |
| `rms_threshold` | 0.01 | Minimum loudness (RMS) to consider speech |
| `start_threshold` | 0.3 | Speech ratio threshold to trigger transcription start |
| `end_threshold` | 0.15 | Speech ratio threshold to trigger transcription end |

## Methods

### `start()`
Begins the recording and transcription process. This method blocks until `stop()` is called or the process is interrupted.

### `stop()`
Stops the recording and transcription process.

### `get_transcripts()`
Returns a copy of all transcribed segments as a list of dictionaries with `index` and `text` fields.

## How It Works

1. **Audio Recording**: Continuously records audio in chunks using sounddevice
2. **Voice Activity Detection**: WebRTC VAD combined with RMS analysis detects speech start/end
3. **Smart Buffering**: 
   - Pre-buffer captures 1s before speech detection
   - Main buffer accumulates speech audio
   - Post-silence detection waits 0.3s after speech ends
4. **Automatic Transcription**: When speech ends or buffer reaches maximum size, audio is queued for transcription
5. **Multi-threaded Processing**: Worker threads process audio segments through Whisper model
6. **Output**: Prints transcriptions or calls custom callback function

## Performance Notes

- **Model Size**: Larger models (medium, large) provide better accuracy but require more processing time and memory
- **Buffer Size**: Longer windows allow for longer continuous speech but increase memory usage
- **Voice Detection**: Proper tuning of thresholds improves accuracy and reduces false triggers
- **Threading**: The system uses separate threads for recording and transcription to maintain real-time performance


## Model Loading
The first run may take time as Whisper models are downloaded and loaded. Subsequent runs will be faster.

## Requirements

- Python 3.10+
- sounddevice
- numpy
- pywhispercpp
- webrtcvad
- Working microphone
- Sufficient RAM for chosen Whisper model

## License

This project is licensed under the same license as [pywhispercpp](https://github.com/absadiki/pywhispercpp/blob/main/LICENSE)/[whisper.cpp](https://github.com/ggml-org/whisper.cpp/blob/master/LICENSE) ([MIT License](https://github.com/Kavan00/LightWhisperSTT/blob/release/LICENSE)).