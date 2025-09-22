import sounddevice as sd
from scipy.io.wavfile import write
from faster_whisper import WhisperModel
from TTS.api import TTS
import playsound
import tkinter as tk
from tkinter import messagebox
import os
import simpleaudio as sa

import time

# === Config ===
SAMPLE_RATE = 44100
DURATION = 5
AUDIO_PATH = "input.wav"
RESPONSE_PATH = "response.wav"

# === Step 1: Record Audio ===
def record_audio():
    print("🎙️ Recording...")
    audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
    sd.wait()
    write(AUDIO_PATH, SAMPLE_RATE, audio)
    print("✅ Recording complete.")

# === Step 2: Transcribe with Whisper ===
def transcribe_audio():
    print("📝 Transcribing...")
    model = WhisperModel("C:/Users/dariu/Models/tiny.en", device="cpu", compute_type="int8")
    segments, _ = model.transcribe(AUDIO_PATH)
    full_text = " ".join(segment.text for segment in segments)
    print("📝 You said:", full_text)
    return full_text

# === Step 3: Basic Sentiment Analysis ===
def analyze_sentiment(text):
    text = text.lower()
    negative_keywords = ["sad", "depressed", "tired", "alone", "hate", "angry", "upset"]
    positive_keywords = ["happy", "excited", "great", "joy", "thankful", "relieved", "good"]

    if any(word in text for word in negative_keywords):
        sentiment = "negative"
    elif any(word in text for word in positive_keywords):
        sentiment = "positive"
    else:
        sentiment = "neutral"

    print(f"🧠 Sentiment: {sentiment}")
    return sentiment

# === Step 4: Get Response Based on Emotion ===
def get_response(emotion):
    responses = {
        "positive": "I'm happy to hear that. Keep going!",
        "neutral": "Thanks for sharing. I'm always here to listen.",
        "negative": "I'm here for you. You are not alone."
    }
    return responses.get(emotion, "I'm listening. Tell me more.")

# === Step 5: Speak Response ===
def speak_response(text):
    print("🔊 STEP 5: Generating speech with Coqui TTS...")
    tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False)
    tts.tts_to_file(text=text, file_path=RESPONSE_PATH)

    print("🔊 Playing response...")
    wave_obj = sa.WaveObject.from_wave_file(RESPONSE_PATH)
    play_obj = wave_obj.play()
    play_obj.wait_done()  # Wait for playback to finish

    try:
        os.remove(RESPONSE_PATH)
        print("🗑️ Deleted response.wav after playback.")
    except Exception as e:
        print(f"⚠️ Could not delete response.wav: {e}")


# === Main MindMate Process ===
def run_mindmate():
    try:
        print("\n⏱️ Starting performance timer...\n")
        start_total = time.time()

        # Step 1: Record Audio
        start_voice = time.time()
        record_audio()
        end_voice = time.time()

        # Step 2: Transcribe
        start_transcribe = time.time()
        text = transcribe_audio()
        end_transcribe = time.time()

        # Step 3: Analyze Sentiment
        start_sentiment = time.time()
        emotion = analyze_sentiment(text)
        end_sentiment = time.time()

        # Step 4 + 5: Generate and Play Response
        start_tts = time.time()
        response = get_response(emotion)
        speak_response(response)
        end_tts = time.time()

        end_total = time.time()

        # Show performance summary
        print("\n📊 Performance Results")
        print(f"Voice Recording:\t{end_voice - start_voice:.2f} seconds")
        print(f"Whisper Transcription:\t{end_transcribe - start_transcribe:.2f} seconds")
        print(f"Sentiment Detection:\t{end_sentiment - start_sentiment:.2f} seconds")
        print(f"TTS Generation:\t\t{end_tts - start_tts:.2f} seconds")
        print(f"Total Loop Time:\t{end_total - start_total:.2f} seconds\n")

    except Exception as e:
        messagebox.showerror("Error", f"Something went wrong:\n{e}")

        

        # === GUI ===
root = tk.Tk()
root.title("MindMate")
root.geometry("300x200")

label = tk.Label(root, text="Press to Talk to MindMate", font=("Arial", 14))
label.pack(pady=20)

button = tk.Button(root, text="🎙️ Speak", font=("Arial", 12), command=run_mindmate)
button.pack(pady=10)

root.mainloop()
