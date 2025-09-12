# command-line script for conversational chatbot with ollama in python

import json
import sys
from collections import deque
import ollama

MODEL_NAME = 'gemma3:1b' # put the ollama model name here
SAVE_FILE = 'conversation.json' # outputs to this file in cd
MEMORY_SIZE = 4  # number of past turns to remember (user input + bot output is one turn)
SYSTEM_PROMPT = '''You are a professional social worker working to provide therapy to users.
 You are empathetic, kind, and a good listener.
 You help users work through their problems and provide emotional support using techniques like C.B.T.
 Do not deviate from this role.'''

def load_memory():
    try:
        with open(SAVE_FILE, "r") as f:
            data = json.load(f)
            return deque(data.get("history", []), maxlen=MEMORY_SIZE)
    except FileNotFoundError:
        return deque(maxlen=MEMORY_SIZE)

def save_memory(history):
    with open(SAVE_FILE, "w") as f:
        json.dump({"history": list(history)}, f, indent=2)

def build_prompt(user_input, history):
    conversation = []
    for t in history:
        conversation.append("User: " + str(t['user']) + "\nbot: " + str(t['ai']))
    return "\n".join(conversation) + "\nUser: " + str(user_input) + "\nbot:"

def chat():
    history = load_memory()

    while True:
        try:
            user_input = input("you: ")
        except (EOFError, KeyboardInterrupt):
            save_memory(history)
            sys.exit(0)

        if user_input.lower() in ["quit", "exit"]:
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
        print(f"bot: {ai_output}")
        history.append({"user": user_input, "ai": ai_output})
        print('\n\n')

if __name__ == "__main__":
    chat()
