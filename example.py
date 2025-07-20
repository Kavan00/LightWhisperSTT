from LightWhisperSTT.LightWhisperSTT import LightWhisperSTT

def handle_transcription(text):
    print(f"Transcribed: {text['text']}")
    # Your custom processing here

test = LightWhisperSTT(model_name="medium", language="de", on_transcription=handle_transcription)
test.start()