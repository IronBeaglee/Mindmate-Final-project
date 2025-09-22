import json
from datetime import datetime
import os

LOG_FILE = "conversation_log.json"
CURATED_FILE = "curated_examples.json"

# === Log a conversation entry ===
def log_conversation(user_text, sentiment, mindmate_reply, feedback="unrated", source="generated"):
    """
    Log a conversation entry with timestamp, user input, sentiment, response,
    feedback (👍/👎/unrated), and source (curated/generated).
    """
    entry = {
        "timestamp": datetime.now().isoformat(),
        "user": user_text,
        "sentiment": sentiment,
        "mindmate": mindmate_reply,
        "feedback": feedback,
        "source": source   # 🔹 new: curated or generated
    }

    # Load existing logs
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            try:
                logs = json.load(f)
            except json.JSONDecodeError:
                logs = []
    else:
        logs = []

    # Append and save
    logs.append(entry)
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2, ensure_ascii=False)


# === Load recent conversation examples ===
def load_recent_examples(n=3):
    """
    Load the last n exchanges and format them as few-shot examples for the prompt.
    """
    if not os.path.exists(LOG_FILE):
        return ""

    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            logs = json.load(f)
    except json.JSONDecodeError:
        return ""

    if not logs:
        return ""

    recent = logs[-n:]
    examples = []
    for e in recent:
        user = e.get("user", "")
        mindmate = e.get("mindmate", "")
        examples.append(f"User: {user}\nMindMate: {mindmate}\n")

    return "\n".join(examples)
