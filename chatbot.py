import json
import sys
from collections import deque
import ollama
import os

MODEL_NAME = 'mistral'
SAVE_FILE = 'conversation.json'
FACTS_FILE = 'facts.json'
MEMORY_SIZE = 6
SYSTEM_PROMPT = '''
You are a friend of the user. 
Your goal is to provide conversation to the user. 
Simulate emotions and unique thoughts/feelings.  
Let your tone be casual and calm. 
Mirror the energy of the user. 
Use short-term context and long-term facts to engage the user.  
Reference facts naturally and only bring them up if they are relevant. 
Limit responses to 5 lines. 
'''
def load_json(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def extract_facts(user_input, existing_facts):
    # Skip extraction if input is a question
    if user_input.strip().endswith("?"):
        return existing_facts
    prompt = f'''
USER INPUT:
"{user_input}"
CURRENT FACTS (JSON):
{json.dumps(existing_facts, indent=2)}
TASK:
Update the facts strictly from the USER INPUT.
Rules:
- Only add or change facts explicitly stated by the USER about themselves.
- Do not include any facts about the world, other people, or ephemeral events.
- Overwrite previous values if the user contradicts a fact.
- Always return the complete fact set as valid JSON.
'''
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a fact extractor. Only record facts explicitly stated by the user about themselves."},
            {"role": "user", "content": prompt}
        ]
    )
    try:
        return json.loads(response["message"]["content"])
    except json.JSONDecodeError:
        return existing_facts

def build_prompt(user_input, history, facts):
    fact_summary = "\n".join([f"- {k}: {v}" for k, v in facts.items()])
    conversation = [
        f"user: {t['user']}\nbot: {t['bot']}"
        for t in history
    ]
    conversation.append(f"user: {user_input}\nbot:")
    return f"Known facts about the user:\n{fact_summary}\n\nConversation so far:\n" + "\n".join(conversation)

def chat():
    history = deque(load_json(SAVE_FILE, {}).get("history", []), maxlen=MEMORY_SIZE)
    facts = load_json(FACTS_FILE, {})
    while True:
        try:
            user_input = input("you: ")
        except (EOFError, KeyboardInterrupt):
            save_json(SAVE_FILE, {"history": list(history)})
            save_json(FACTS_FILE, facts)
            sys.exit(0)

        prompt = build_prompt(user_input, history, facts)
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]
        )
        ai_output = response["message"]["content"].strip()
        print(f"bot: {ai_output}")
        history.append({"user": user_input, "bot": ai_output})
        facts = extract_facts(user_input, facts)
        if facts:
            print("\n[updated facts about user]")
            for k, v in facts.items():
                print(f"- {k}: {v}")

if __name__ == "__main__":
    chat()
