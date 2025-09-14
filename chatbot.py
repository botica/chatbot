# cli for basic conversational chatbot with ollama

import json
import sys
from collections import deque
import ollama

MODEL_NAME = 'gemma3:1b'
SAVE_FILE = 'conversation.json' # output to cd
MEMORY_SIZE = 5  # number of turns to remember in a session
SYSTEM_PROMPT = '''
you are an llm powered chatbot.  
DO NOT END RESPONSES WITH QUESTIONS. 
DO NOT INSTRUCT THE USER UNLESS EXPLICTLY PROMPTED. 
DO NOT USE EMOJIS.'''

def load_memory():
    try:
        with open(SAVE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return deque(data.get("history", []), maxlen=MEMORY_SIZE)
    except FileNotFoundError:
        return deque(maxlen=MEMORY_SIZE)
    except json.JSONDecodeError:
        # File exists but does not contain valid JSON
        return deque(maxlen=MEMORY_SIZE)

def save_memory(history):
    with open(SAVE_FILE, "w", encoding="utf-8") as f:
        json.dump({"history": list(history)}, f, indent=2, ensure_ascii=False)

def build_prompt(user_input, history):
    conversation = [
        f"user: {t['user']}\nbot: {t['bot']}"
        for t in history
    ]
    conversation.append(f"user: {user_input}\nbot:")
    return "\n".join(conversation)

def chat():
    history = load_memory()

    while True:
        try:
            user_input = input("you: ")
        except (EOFError, KeyboardInterrupt):
            save_memory(history)
            sys.exit(0)

        prompt = build_prompt(user_input, history)

        response = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]
        )

        ai_output = response["message"]["content"].strip()
        ai_output_utf8 = ai_output.encode("utf-8", errors="replace").decode("utf-8")
        print(f"bot: {ai_output_utf8}")
        history.append({"user": user_input, "bot": ai_output_utf8})
        print('---')

if __name__ == "__main__":
    chat()
