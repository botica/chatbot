#refactor
#extraxt facts before outputing response

# keeps running convo
# exports facts about user to json and keeps in context as well
# call llm to get convo resp
# call llm to extact facts

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

def merge_facts(existing, new):
    '''
    merge new facts into existing facts. 
    If a fact already exists and is different, store as a list of values. 
    If it's the same, ignore.
    '''
    for k, v in new.items():
        if k in existing:
            if isinstance(existing[k], list):
                if v not in existing[k]:
                    existing[k].append(v)
            else:
                if existing[k] != v:
                    existing[k] = [existing[k], v]
        else:
            existing[k] = v
    return existing

def extract_facts(user_input, existing_facts):
    '''
    extracts self-related facts from user input using llm
    normalizes categories and merges with existing facts.
    '''
    extract_system_prompt = '''
You are a fact extractor.
Only record facts the USER explicitly states about THEMSELVES.

You may use these categories (or invent a new one if nothing fits):
- Name
- Age
- Gender
- Location
- Nationality
- Ethnicity
- FormerJob
- CurrentJob
- Hobby (activities the user *does* for enjoyment, e.g. "Hiking", "Gaming")
- Interest (subjects the user *follows or learns about*, e.g. "History", "Astronomy")
- Preference (foods, drinks, music, or other things the user *likes or dislikes*, e.g. "Potatoes", "Jazz")
- Trait (adjectives describing the user’s personality, e.g. "Kind", "Curious")
- Skill (things the user *can do*, e.g. "Cooking", "Programming")
- Education
- RelationshipStatus
- Pet
- Mood

Guidelines:
- If the user mentions an *activity they do*, classify it as a Hobby.
- If the user mentions something they *enjoy or consume* (like food, drinks, music), classify it as a Preference.
- If the user describes their *personality*, classify it as a Trait.
- If the fact doesn’t fit any, create a short new category.

Facts must be concise and general (e.g., "FormerJob": "Cook").
Ignore vague memories, opinions, or random details.

Return valid JSON.
If no facts, return {}.

Examples:

User: "I love potatoes."
Output: {"Preference": "Potatoes"}

User: "I like hiking."
Output: {"Hobby": "Hiking"}

User: "I'm Irish."
Output: {"Nationality": "Irish"}

User: "I'm good at programming."
Output: {"Skill": "Programming"}

User: "I’m a happy person."
Output: {"Trait": "Happy"}
    '''

    extract_prompt = f'''
USER INPUT:
"{user_input}"
CURRENT FACTS (JSON):
{json.dumps(existing_facts, indent=2)}
Return only the new or updated facts (not the full set).
Return valid JSON.
'''

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": extract_system_prompt},
            {"role": "user", "content": extract_prompt}
        ]
    )

    try:
        new_facts = json.loads(response["message"]["content"])
        return merge_facts(existing_facts, new_facts)
    except json.JSONDecodeError:
        return existing_facts

def build_prompt(user_input, history, facts):
    fact_summary = "\n".join([f"- {k}: {v}" for k, v in facts.items()])
    conversation = [
        f"user: {t['user']}\nbot: {t['bot']}"
        for t in history
    ]
    conversation.append(f"user: {user_input}\nbot:")
    return f"Saved facts about user:\n{fact_summary}\n\nCurrent conversation:\n" + "\n".join(conversation)

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
            print("\n[facts about user]")
            for k, v in facts.items():
                print(f"- {k}: {v}")

if __name__ == "__main__":
    chat()
