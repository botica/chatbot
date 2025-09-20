import json
import sys
from collections import deque
import ollama
from datetime import datetime

MODEL_NAME = 'mistral'
SAVE_FILE = 'conversation.json'
FACTS_FILE = 'facts.json'
MEMORY_SIZE = 6

SYSTEM_PROMPT = '''
You are a friend of the user. 
Your goal is simply to talk to the user. 
Simulate emotions and unique thoughts/feelings.  
Let your tone be casual and calm. 
Mirror the energy of the user. 
Use short-term context and long-term facts to engage the user naturally. 
Limit responses to 5 lines. 
'''

def timestamp():
    return datetime.utcnow().isoformat()

def load_json(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def normalize_facts(facts):
    """
    Normalize all facts to dicts with {"value": ..., "timestamp": ...}.
    Flattens nested values recursively.
    Handles strings, dicts, and lists of strings/dicts.
    """
    def normalize_entry(entry):
        if isinstance(entry, str):
            return {"value": entry, "timestamp": timestamp()}
        elif isinstance(entry, dict):
            if "value" in entry and "timestamp" in entry:
                val = entry["value"]
                ts = entry["timestamp"]
                if isinstance(val, dict):
                    val = normalize_entry(val)["value"]
                return {"value": val, "timestamp": ts}
            return {k: normalize_entry(v) for k, v in entry.items()}
        elif isinstance(entry, list):
            flat_list = []
            for e in entry:
                ne = normalize_entry(e)
                if isinstance(ne, list):
                    flat_list.extend(ne)
                else:
                    flat_list.append(ne)
            return flat_list
        return entry

    return {k: normalize_entry(v) for k, v in facts.items()}

def merge_facts(existing, new):
    for k, v in new.items():
        # If new fact is a list, merge items individually
        if isinstance(v, list):
            for item in v:
                merge_facts(existing, {k: item})
            continue

        # Extract the actual value
        if isinstance(v, dict) and "value" in v:
            fact_value = v["value"]
        else:
            fact_value = v

        new_fact = {"value": fact_value, "timestamp": timestamp()}

        # Preference/Dislike contradictions
        if k == "Preference" and "Dislike" in existing:
            if isinstance(existing["Dislike"], list):
                existing["Dislike"] = [d for d in existing["Dislike"] if d["value"] != fact_value]
                if not existing["Dislike"]:
                    del existing["Dislike"]
            elif existing["Dislike"]["value"] == fact_value:
                del existing["Dislike"]

        elif k == "Dislike" and "Preference" in existing:
            if isinstance(existing["Preference"], list):
                existing["Preference"] = [p for p in existing["Preference"] if p["value"] != fact_value]
                if not existing["Preference"]:
                    del existing["Preference"]
            elif existing["Preference"]["value"] == fact_value:
                del existing["Preference"]

        elif k == "Mood":
            existing["Mood"] = new_fact
            continue

        # Merge normally with flattening
        if k in existing:
            if isinstance(existing[k], dict):
                existing[k] = [existing[k]]

            flat_list = []
            for entry in existing[k]:
                if isinstance(entry, list):
                    flat_list.extend(entry)
                else:
                    flat_list.append(entry)
            existing[k] = flat_list

            # Add new_fact if not duplicate
            if all(entry["value"] != fact_value for entry in existing[k]):
                existing[k].append(new_fact)
        else:
            existing[k] = new_fact

    return existing

def extract_facts(user_input, existing_facts):
    extract_system_prompt = '''
You are a fact extractor.
ONLY record facts the USER explicitly states about THEMSELVES.

Strict rules:
- Do NOT infer facts. Only record if the user clearly says:
    "I am...", "I like...", "I love...", "I enjoy...", "I eat...", "I drink...", or "my name is...".
- Do NOT record anything the user mentions about other people, animals, or things in general.
- Do NOT record facts if the user is asking a question, speculating, or making general statements.
- Only record facts explicitly about the USER.
- Do not assume preferences, dislikes, or traits unless explicitly stated by the user.

Categories you may use:
- Name, Age, Gender, Location, Nationality, Ethnicity, FormerJob, CurrentJob
- Hobby, Interest, Preference, Dislike, Trait, Skill, Education
- RelationshipStatus, Pet, Mood

Return valid JSON with only NEW or UPDATED facts.
If no facts, return {}.
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
    def format_fact(k, v):
        if isinstance(v, dict):
            return f"- {k}: {v['value']} (timestamp: {v['timestamp']})"
        elif isinstance(v, list):
            return "\n".join([f"- {k}: {entry['value']} (timestamp: {entry['timestamp']})" for entry in v])
        return f"- {k}: {v}"

    fact_summary = "\n".join([format_fact(k, v) for k, v in facts.items()])
    conversation = [f"user: {t['user']}\nbot: {t['bot']}" for t in history]
    conversation.append(f"user: {user_input}\nbot:")
    return f"Saved facts about user:\n{fact_summary}\n\nCurrent conversation:\n" + "\n".join(conversation)

def extract_value(v):
    """Extract value and timestamp for printing."""
    if isinstance(v, dict):
        val = v.get("value", "")
        ts = v.get("timestamp", "")
        return val, ts
    return v, ""

def chat():
    history = deque(load_json(SAVE_FILE, {}).get("history", []), maxlen=MEMORY_SIZE)
    facts = normalize_facts(load_json(FACTS_FILE, {}))
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

        # Print facts with timestamp
        if facts:
            print("\n[facts about user]")
            for k, v in facts.items():
                if isinstance(v, list):
                    for entry in v:
                        val, ts = extract_value(entry)
                        print(f"- {k}: {val} (timestamp: {ts})")
                else:
                    val, ts = extract_value(v)
                    print(f"- {k}: {val} (timestamp: {ts})")

if __name__ == "__main__":
    chat()
