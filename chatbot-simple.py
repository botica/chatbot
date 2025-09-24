import sys
import time
import json
from collections import deque
from datetime import datetime
import ollama

# ----------------- Configuration -----------------
MODEL_NAME = "mistral"
SAVE_FILE = "conversation.json" # conversation history
MEMORY_SIZE = 6  # number of turns to remember

SYSTEM_PROMPT = '''
You are a friendly, relaxed, and conversational chatbot.
Your goal is to keep the user engaged and respond like a thoughtful friend.
Keep responses clear, natural, and casual. Show understanding, curiosity, or light humor.
'''

# ----------------- Utility Functions -----------------
def timestamp():
    return datetime.utcnow().isoformat()

def load_json(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, ValueError, json.JSONDecodeError):
        return default

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def typewriter_stream(text, delay=0.01):
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)

def human_readable_time_diff(last_time_str):
    try:
        last_time = datetime.fromisoformat(last_time_str)
        diff = datetime.utcnow() - last_time
        if diff.days > 0:
            return f"{diff.days} day(s) ago"
        hours = diff.seconds // 3600
        if hours > 0:
            return f"{hours} hour(s) ago"
        minutes = (diff.seconds % 3600) // 60
        if minutes > 0:
            return f"{minutes} minute(s) ago"
        return "just now"
    except Exception:
        return None

def build_prompt(user_input, history):
    conversation = [f"user: {t['user']}\nbot: {t['bot']}" for t in history]
    conversation.append(f"user: {user_input}\nbot:")
    return "Current conversation:\n" + "\n".join(conversation)

# ----------------- Main Chat Loop -----------------
def chat():
    save_data = load_json(SAVE_FILE, {})
    history = deque(save_data.get("history", []), maxlen=MEMORY_SIZE)

    last_chat_time = save_data.get("last_chat_time")
    if last_chat_time:
        diff_str = human_readable_time_diff(last_chat_time)
        if diff_str:
            print(f"(last chat was {diff_str})")
    else:
        print("(this looks like your first chat!)")

    while True:
        try:
            user_input = input("you: ")
        except (EOFError, KeyboardInterrupt):
            save_data["history"] = list(history)
            save_data["last_chat_time"] = timestamp()
            save_json(SAVE_FILE, save_data)
            sys.exit(0)

        prompt = build_prompt(user_input, history)
        ai_output = ""
        first_chunk = True

        for event in ollama.chat(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            stream=True
        ):
            if "message" in event and "content" in event["message"]:
                if first_chunk:
                    sys.stdout.write("\r" + " " * len("bot is thinking...") + "\r")
                    sys.stdout.flush()
                    first_chunk = False
                chunk = event["message"]["content"]
                ai_output += chunk
                typewriter_stream(chunk)
        print()

        history.append({"user": user_input, "bot": ai_output.strip()})

        # Save conversation after each turn
        save_data["history"] = list(history)
        save_data["last_chat_time"] = timestamp()
        save_json(SAVE_FILE, save_data)

# ----------------- Entry Point -----------------
if __name__ == "__main__":
    chat()
