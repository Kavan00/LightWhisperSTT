import sounddevice as sd
import numpy as np
import threading
import queue
from collections import deque
from pywhispercpp.model import Model
import time


class LightWhisperSTT:
    def __init__(self,
                 model_name="medium",
                 language="auto",
                 chunk_size=4096,
                 window_seconds=10, #package size -> will listen for 10s and then transcribe
                 model_threads=4,
                 print_debug=False,
                 on_transcription=None):  # optional callback

        self.sample_rate = 16000  # Whisper supports 16 kHz
        self.channels = 1
        self.chunk_size = chunk_size
        self.window_seconds = window_seconds
        self.model_threads = model_threads
        self.num_workers = 2
        self.print_debug = print_debug

        self.model = Model(model_name, print_progress=False, n_threads=model_threads, language=language)
        self.audio_queue = queue.Queue()
        self.model_lock = threading.Lock()
        self.buffer = deque(maxlen=self.sample_rate * window_seconds)
        self.index = 0
        self.running = False

        self.transcripts = []  # stores all segments
        self.on_transcription = on_transcription  # optional callback

    def audio_callback(self, indata, frames, time_info, status):
        self.buffer.extend(indata[:, 0])

    def recorder_loop(self):
        with sd.InputStream(callback=self.audio_callback,
                            samplerate=self.sample_rate,
                            channels=self.channels,
                            blocksize=self.chunk_size):
            if self.print_debug: print("Starting recording...")
            self.running = True
            try:
                while self.running:
                    if len(self.buffer) >= self.sample_rate * self.window_seconds:
                        snippet = np.array(self.buffer).copy()
                        self.audio_queue.put((self.index, snippet))
                        self.index += 1
                        time.sleep(self.window_seconds)
                    else:
                        time.sleep(0.1)
            except KeyboardInterrupt:
                if self.print_debug: print("Stopped recording..")

    def transcriber_worker(self):
        while True:
            index, audio = self.audio_queue.get()
            try:
                with self.model_lock:
                    segments = self.model.transcribe(audio)
                for segment in segments:
                    entry = {
                        "index": index,
                        "text": segment.text.strip()
                    }
                    self.transcripts.append(entry)

                    if self.on_transcription:
                        self.on_transcription(entry)
                    else:
                        if self.print_debug:
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
        for _ in range(self.num_workers):
            threading.Thread(target=self.transcriber_worker, daemon=True).start()
        self.recorder_loop()

    def stop(self):
        self.running = False

    def get_transcripts(self):
        return self.transcripts.copy()
