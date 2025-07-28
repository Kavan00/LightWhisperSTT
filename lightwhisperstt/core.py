import sounddevice as sd
import numpy as np
import threading
import queue
import webrtcvad
from collections import deque
from pywhispercpp.model import Model
import time
import librosa


class LightWhisperSTT:
    def __init__(self,
                 model_name="medium",
                 language="auto",
                 window_seconds=15, # TimeWindow used for buffer
                 model_threads=4, # Threads
                 on_transcription=None, # Callback function
                 rms_threshold=0.01, # Adjust voice sensitivity
                 start_threshold=0.3, # Speech detection sensitivity
                 end_threshold=0.15, # End-of-speech sensitivity
                 vad_aggressiveness=3, # VoiceActivationDetection aggressiveness (0-3 - 0 ultra sensitive, 3 - moderate)
                 chunk_size=4096,  # ChunkSize
                 print_debug=False,  # Enable debug output
                 ):

        self.input_sample_rate = self._detect_sample_rate()
        self.target_sample_rate = 16000  # Whisper's preferred rate
        self.channels = 1
        self.chunk_size = chunk_size
        self.window_seconds = window_seconds
        self.print_debug = print_debug
        self.vad = webrtcvad.Vad(vad_aggressiveness)

        self.triggered = False
        self.silence_counter = 0

        self.rms_threshold = rms_threshold
        self.start_threshold = start_threshold
        self.end_threshold = end_threshold

        self.pre_buffer = deque(maxlen=self.input_sample_rate * 1)  # 1s PreBuffer
        self.post_silence_frames = int(0.3 * self.input_sample_rate)  # 0.3s Post
        self.speech_window = deque(maxlen=2)  # 2 Frames

        self.buffer = deque(maxlen=self.input_sample_rate * window_seconds)
        self.audio_queue = queue.Queue()
        self.index = 0
        self.running = False

        self.model = Model(model_name, print_progress=False, n_threads=model_threads, language=language)
        self.on_transcription = on_transcription
        self.transcripts = []

    def _detect_sample_rate(self):
        test_rates = [16000, 44100, 48000]
        for rate in test_rates:
            try:
                sd.check_input_settings(samplerate=rate, channels=1)
                return rate
            except:
                continue
        return 16000

    def is_speech(self, frame_bytes):
        return self.vad.is_speech(frame_bytes, 16000)  # VAD expects 16kHz

    def audio_callback(self, indata, frames, time_info, status):
        frame = indata[:, 0]
        audio_frame = frame.tobytes()[:960]  # 30ms * 16kHz * 2 Bytes

        # Should calculate faster but is not precise
        volume = np.abs(frame).mean()
        # volume = np.sqrt(np.mean(frame.astype(np.float32) ** 2))
        is_speech = self.is_speech(audio_frame)
        self.speech_window.append(is_speech and volume > self.rms_threshold)
        speech_ratio = sum(self.speech_window) / len(self.speech_window)

        if not self.triggered and speech_ratio > self.start_threshold:
            if self.print_debug: print("▶️ Speech started")
            self.triggered = True
            self.buffer.extend(self.pre_buffer)
            self.pre_buffer.clear()

        if self.triggered:
            self.buffer.extend(frame)

            if speech_ratio < self.end_threshold:
                self.silence_counter += len(frame)
                if self.silence_counter >= self.post_silence_frames:
                    self.flush_buffer()
        else:
            self.pre_buffer.extend(frame)
            self.silence_counter = 0

    def flush_buffer(self):
        if self.print_debug: print("⏹️ Speech ended → enqueue")
        self.triggered = False
        snippet = np.array(self.buffer)

        # Resample to 16kHz for Whisper if needed
        if self.input_sample_rate != self.target_sample_rate:
            snippet = librosa.resample(snippet.astype(np.float32),
                                       orig_sr=self.input_sample_rate,
                                       target_sr=self.target_sample_rate)

        self.audio_queue.put((self.index, snippet))
        self.buffer.clear()
        self.silence_counter = 0
        self.speech_window.clear()
        self.index += 1

    def recorder_loop(self):
        with sd.InputStream(callback=self.audio_callback,
                            samplerate=self.input_sample_rate,
                            channels=self.channels,
                            blocksize=self.chunk_size):
            if self.print_debug:
                print("🎙️ Recording started")
            self.running = True
            try:
                while self.running:
                    if len(self.buffer) >= self.input_sample_rate * self.window_seconds:
                        if self.print_debug:
                            print("Max buffer reached → forced flush")
                        self.flush_buffer()
                    time.sleep(0.01)
            except KeyboardInterrupt:
                if self.print_debug:
                    print("Interrupted")

    def transcriber_worker(self):
        while True:
            index, audio = self.audio_queue.get()
            try:
                segments = self.model.transcribe(audio)
                for segment in segments:
                    entry = {"index": index, "text": segment.text.strip()}
                    self.transcripts.append(entry)
                    if self.on_transcription:
                        self.on_transcription(entry)
                    elif self.print_debug:
                        print(f"[{index}] {entry['text']}")
                    else:
                        print(entry["text"])
            except Exception as e:
                print(f"[{index}] Error: {e}")
            finally:
                self.audio_queue.task_done()

    def get_model_languages(self):
        return self.model.available_languages()

    def start(self):
        threading.Thread(target=self.transcriber_worker, daemon=True).start()
        self.recorder_loop()

    def stop(self):
        self.running = False

    def get_transcripts(self):
        return self.transcripts.copy()
