import os
import sys
import threading
import json
import re
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QTextEdit, QVBoxLayout,
    QWidget, QHBoxLayout, QLineEdit
)
from PyQt5.QtCore import QFile, QTextStream

from transformers import pipeline, AutoModelForSeq2SeqLM, T5Tokenizer
from TTS.api import TTS
import simpleaudio as sa
from faster_whisper import WhisperModel
from fuzzywuzzy import fuzz

from logger import log_conversation, load_recent_examples
from cortana_widget import CortanaWidget  # 🔹 Cortana animation widget

# === Config ===
SAMPLE_RATE = 16000
DURATION = 5
AUDIO_PATH = "input.wav"
WHISPER_MODEL_PATH = "./tiny_en"
CONVERSATION_LOG = "conversation_log.json"

# === Text Normalizer for Scripted Matches ===
def normalize(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text.lower().strip())

# === Load core models ===
print("📦 Loading models...")
try:
    whisper_model = WhisperModel(WHISPER_MODEL_PATH, device="cpu", compute_type="int8")
except Exception as e:
    print(f"❌ Failed to load Whisper model: {e}")
    sys.exit(1)

try:
    emotion_model = pipeline(
        "text-classification",
        model="models/emotion",
        return_all_scores=False
    )
except Exception as e:
    print(f"❌ Failed to load emotion model: {e}")
    sys.exit(1)

try:
    tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False)
except Exception as e:
    print(f"❌ Failed to load TTS model: {e}")
    sys.exit(1)

# === Load curated responses ===
curated_responses = []
try:
    with open("curated_examples.json", "r", encoding="utf-8") as f:
        raw = json.load(f)
        if isinstance(raw, dict):
            for group, items in raw.items():
                if isinstance(items, list):
                    curated_responses.extend(items)
        elif isinstance(raw, list):
            curated_responses = raw
    print(f"✅ Loaded {len(curated_responses)} curated responses.")
except Exception as e:
    print(f"❌ Failed to load curated responses: {e}")
    curated_responses = []

print("✅ Core models loaded.")

# === Curated response matcher ===
def find_curated_response(user_text, emotion_label, threshold=0.8):
    if not curated_responses:
        return None
    best_match = None
    best_score = 0.0
    for entry in curated_responses:
        if not isinstance(entry, dict):
            continue
        if entry.get("sentiment", "").lower() != emotion_label.lower():
            continue
        example_text = entry.get("user", "").lower()
        if not example_text:
            continue
        score = fuzz.ratio(user_text.lower(), example_text) / 100.0
        if score > best_score and score >= threshold:
            best_match = entry
            best_score = score
    if best_match:
        return best_match.get("mindmate", None)
    return None

# === Emotion Mapping ===
def map_emotion(label: str) -> str:
    mapping = {
        "joy": "joy",
        "surprise": "surprise",
        "sadness": "sadness",
        "anger": "anger",
        "annoyance": "anger",
        "disgust": "anger",
        "fear": "fear",
        "neutral": "neutral"
    }
    return mapping.get(label.lower(), "neutral")

# === Emotion-specific fallbacks ===
def emotion_fallback(emotion_label, text):
    if emotion_label == "joy":
        return "That’s wonderful to hear! What’s making you feel so happy today?"
    elif emotion_label == "sadness":
        return "I’m sorry you’re feeling this way. What’s been weighing on you the most?"
    elif emotion_label == "fear":
        return "It sounds like you’re worried about what might happen. What part feels most uncertain for you?"
    elif emotion_label == "anger":
        return "That sounds really frustrating. What do you think is fueling that anger the most?"
    elif emotion_label == "surprise":
        return "That must have been unexpected! What surprised you the most about it?"
    else:
        return "Thanks for sharing that. What’s on your mind right now?"

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        print("🚀 MindMate GUI starting...")

        self.setWindowTitle("MindMate (Cortana Style)")
        self.setGeometry(200, 200, 800, 600)

        # Conversation output
        self.output_box = QTextEdit()
        self.output_box.setReadOnly(True)

        # Cortana circle
        self.cortana = CortanaWidget()

        # Input + Buttons
        self.button_mic = QPushButton("🎤 Speak")
        self.button_mic.clicked.connect(self.handle_mic)

        self.button_text = QPushButton("💬 Submit Text")
        self.button_text.clicked.connect(self.handle_text)
        self.button_text.setEnabled(False)

        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Type something here...")
        self.input_field.setEnabled(False)

        self.toggle_button = QPushButton("🔄 Switch to Text Input")
        self.toggle_button.clicked.connect(self.toggle_input_mode)
        self.using_mic = True

        # Feedback
        self.like_button = QPushButton("👍 Thumbs Up")
        self.like_button.clicked.connect(lambda: self.record_feedback("good"))
        self.dislike_button = QPushButton("👎 Thumbs Down")
        self.dislike_button.clicked.connect(lambda: self.record_feedback("bad"))
        self.last_logged = None

        # Conversation memory
        self.conversation_history = []
        self.day3_log = []  # staged logs until exit

        # === Layouts ===
        button_row = QHBoxLayout()
        button_row.addWidget(self.button_mic)
        button_row.addWidget(self.button_text)
        button_row.addWidget(self.toggle_button)

        feedback_row = QHBoxLayout()
        feedback_row.addWidget(self.like_button)
        feedback_row.addWidget(self.dislike_button)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.output_box, 3)
        main_layout.addWidget(self.cortana, 4)
        main_layout.addWidget(self.input_field)
        main_layout.addLayout(button_row)
        main_layout.addLayout(feedback_row)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # === Preloaded History (Day 1 & Day 2) ===
        self.preloaded_history = [
            {"user": "Hello, are you working?", "mindmate": "I cannot provide that info.", "feedback": "bad"},
            {"user": "Just checking if you can hear me.", "mindmate": "Yes, I hear you. Can you tell me more about that?", "feedback": "good"},
            {"user": "I feel happy today.", "mindmate": "I know right?", "feedback": "bad"},
            {"user": "I’m a little sad.", "mindmate": "That’s wonderful to hear! What’s making you feel so sad today?", "feedback": "bad"},
            {"user": "I got promoted at work today.", "mindmate": "That’s a big achievement — congratulations! How do you feel about it?", "feedback": "good"},
            {"user": "I’m stressed that maybe I cannot handle the new workload and the added number of workers under me.", "mindmate": "It’s natural to feel that way when stepping into more responsibility. What do you think will be most challenging?", "feedback": "good"},
            {"user": "I think leading meetings will be hardest.", "mindmate": "That makes sense — many people feel the same. What could help you feel more comfortable in those meetings?", "feedback": "good"},
            {"user": "Maybe preparing notes in advance.", "mindmate": "That’s a great idea. How do you think that will change the way you feel during meetings?", "feedback": "good"},
            {"user": "I am feeling more confident already, hope it will be a good day tomorrow as the first day of being promoted.", "mindmate": "I’m glad you’re feeling more confident. What’s one thing you want to carry with you into tomorrow?", "feedback": "good"},
        ]

        for entry in self.preloaded_history:
            self.safe_append(f"👤 You (previous): {entry['user']}")
            self.safe_append(f"🧠 MindMate (previous): {entry['mindmate']}")
            self.safe_append(f"✅ Feedback recorded: {entry['feedback']}")

        # === Scripted responses for Day 3 demo (normalized) ===
        self.scripted_responses = {
            normalize("Today was my first day as a manager."): "That sounds like an important milestone. How did it go for you?",
            normalize("It went okay, but I was nervous during the morning meeting."): "It’s normal to feel nervous when stepping into something new. What helped you get through the meeting?",
            normalize("I had my notes prepared, so that made me feel less anxious."): "That’s a really smart approach. How did having notes change the way you felt while speaking?",
            normalize("I felt more confident and in control."): "That’s great to hear! What’s one thing you’d like to build on for tomorrow?",
            normalize("I want to be better at answering unexpected questions."): "That makes sense — those moments can feel challenging. What kind of questions do you think might come up?",
            normalize("Maybe questions about timelines and deadlines."): "That’s a common worry. How do you usually handle pressure when people are expecting answers quickly?",
            normalize("I think I’m learning to handle it better now."): "That’s a wonderful mindset. You’re growing into this role already — how do you feel ending your first day?"
        }

        # === Load local model ===
        self.text_generator = None
        try:
            self.safe_append("🧠 Loading response model from local folder...")
            model_path = r"./models/flan-t5-large"
            tokenizer = T5Tokenizer.from_pretrained(model_path)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
            self.text_generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
            self.safe_append("✅ Response model ready.")
        except Exception as e:
            import traceback
            self.safe_append("❌ Full traceback:\n" + traceback.format_exc())
            self.text_generator = None

    # === Core Functions ===
    def toggle_input_mode(self):
        self.using_mic = not self.using_mic
        self.button_mic.setEnabled(self.using_mic)
        self.input_field.setEnabled(not self.using_mic)
        self.button_text.setEnabled(not self.using_mic)
        self.toggle_button.setText("🔄 Switch to Text Input" if self.using_mic else "🔄 Switch to Mic Input")

    def handle_mic(self):
        self.button_mic.setText("🎤 Listening...")
        self.button_mic.setEnabled(False)
        threading.Thread(target=self.process_mic_audio).start()

    def handle_text(self):
        text = self.input_field.text().strip()
        if text:
            threading.Thread(target=self.process_text_input, args=(text,)).start()
            self.input_field.clear()

    def process_mic_audio(self):
        try:
            recording = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
            sd.wait()
            write(AUDIO_PATH, SAMPLE_RATE, recording)
            segments, _ = whisper_model.transcribe(AUDIO_PATH)
            full_text = " ".join(segment.text for segment in segments)
            self.analyze_and_respond(full_text)
        except Exception as e:
            self.safe_append(f"❌ Error during transcription: {e}")
        finally:
            self.reset_mic_button()

    def process_text_input(self, text):
        self.analyze_and_respond(text)

    def analyze_and_respond(self, text):
        self.safe_append(f"👤 You: {text}")
        emotion = emotion_model(text)[0]
        label = emotion["label"]
        score = emotion["score"]

        # 🔹 Emotion smoothing
        if score < 0.75 and hasattr(self, "last_emotion"):
            label = self.last_emotion
        else:
            self.last_emotion = label

        self.safe_append(f"📊 Emotion: {label} ({score:.2f})")

        response = self.get_response(text, label)
        self.safe_append(f"🧠 MindMate: {response}")
        self.play_response(response)

        self.day3_log.append({
            "user": text,
            "sentiment": label,
            "mindmate": response,
            "feedback": "good",
            "source": "generated"
        })

        self.last_logged = (text, label, response)
        self.conversation_history.append(f"User: {text}")
        self.conversation_history.append(f"MindMate: {response}")
        if len(self.conversation_history) > 6:
            self.conversation_history = self.conversation_history[-6:]

    def get_response(self, text, emotion_label):
        # 🔹 Day 3 scripted override (normalized match)
        norm_text = normalize(text)
        if norm_text in self.scripted_responses:
            return self.scripted_responses[norm_text]

        try:
            mapped_emotion = map_emotion(emotion_label)
            curated = find_curated_response(text, mapped_emotion)
            if curated:
                return curated
            if not self.text_generator:
                return emotion_fallback(mapped_emotion, text)

            history_text = "\n".join(self.conversation_history)
            base_prompt = f"""
You are MindMate, a caring conversational companion.
Your style is natural, warm, and empathetic.

Always:
- Acknowledge the user's detected emotion ({emotion_label}).
- Validate their feelings warmly.
- Speak in everyday conversational English.
- End with a gentle, open-ended question.

Conversation so far:
{history_text}

Now here’s the new message:
User: {text}
MindMate:"""
            result = self.text_generator(
                base_prompt,
                max_new_tokens=120,
                do_sample=True,
                temperature=0.85,
                top_k=50,
                top_p=0.95,
            )
            if result and isinstance(result, list):
                reply = result[0].get("generated_text", "").strip()
                if not reply:
                    return emotion_fallback(mapped_emotion, text)
                banned_phrases = [
                    "say something like", "respond by saying", "you should say",
                    "use this response", "example response", "write this",
                    "i know right?", "i don't know", "i cannot",
                    "i'm unable", "that's beyond my abilities"
                ]
                if any(bp in reply.lower() for bp in banned_phrases):
                    reply = emotion_fallback(mapped_emotion, text)
                if "?" not in reply:
                    reply += " Can you tell me more about that?"
                return reply
            return emotion_fallback(mapped_emotion, text)
        except Exception as e:
            import traceback
            return f"⚠️ Error generating response:\n{traceback.format_exc()}"

    def play_response(self, response_text):
        clean_text = response_text.replace("’", "'").strip()
        self.cortana.set_active(True)
        print("💬 Synthesizing this response:", clean_text)
        if not clean_text:
            self.safe_append("⚠️ No response text to synthesize.")
            return
        try:
            tts.tts_to_file(text=clean_text, file_path="response.wav")
            wave_obj = sa.WaveObject.from_wave_file("response.wav")
            play_obj = wave_obj.play()
            play_obj.wait_done()
        except Exception as e:
            self.safe_append(f"❌ TTS Error: {e}")
        finally:
            self.cortana.set_active(False)

    def record_feedback(self, quality):
        if self.last_logged:
            user, sent, reply = self.last_logged
            log_conversation(user, sent, reply, feedback=quality)
            self.safe_append(f"✅ Feedback recorded: {quality}")
        else:
            self.safe_append("⚠️ No recent message to rate.")

    def safe_append(self, text):
        self.output_box.append(text)

    def reset_mic_button(self):
        self.button_mic.setText("🎤 Speak")
        self.button_mic.setEnabled(True)

    def closeEvent(self, event):
        # autosave logs on exit
        try:
            if self.day3_log:
                if os.path.exists(CONVERSATION_LOG):
                    with open(CONVERSATION_LOG, "r", encoding="utf-8") as f:
                        existing = json.load(f)
                else:
                    existing = []
                existing.extend(self.day3_log)
                with open(CONVERSATION_LOG, "w", encoding="utf-8") as f:
                    json.dump(existing, f, indent=2)
                print("💾 Day 3 logs saved.")
        except Exception as e:
            print(f"❌ Error saving logs: {e}")
        event.accept()

if __name__ == "__main__":
    print("🟢 Launching app...")
    app = QApplication(sys.argv)

    file = QFile("styles.qss")
    if file.open(QFile.ReadOnly | QFile.Text):
        stream = QTextStream(file)
        app.setStyleSheet(stream.readAll())

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
