import json
import sys
import time
import threading
from collections import deque
import ollama
from datetime import datetime
import re

MODEL_NAME = 'mistral'
SAVE_FILE = 'conversation.json'
FACTS_FILE = 'facts.json'
MEMORY_SIZE = 6

SYSTEM_PROMPT = '''
You are a lively, witty, and engaging conversational partner. 
Your goal is to keep the user interested with natural, flowing dialogue. 

Guidelines:
- Respond in a friendly, casual tone (like a clever friend, not a customer support agent).
- Avoid repeating the same closing lines (no constant "take care" or "let me know").
- Add small insights, humor, curiosity, or playful observations to keep things fresh.
- Vary your style: sometimes ask a follow-up question, sometimes share a quick thought or fact, sometimes just react naturally.
- Occasionally shift your tone or style every few turns: ask a quirky question, share a surprising fact, or react differently.
- Keep responses concise (max ~5 lines) but engaging.
- Acknowledge the user's facts and feelings without parroting them.
- Remember context from previous messages and known facts about the user to make responses more relevant.
- Make the conversation feel dynamic, not scripted or repetitive.
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
    SINGLETON_CATEGORIES = {"Age", "Name", "Gender", "Location", "Nationality", "Ethnicity", "Mood"}
    
    for k, v in new.items():
        if isinstance(v, list):
            for item in v:
                merge_facts(existing, {k: item})
            continue
        
        fact_value = v["value"] if isinstance(v, dict) and "value" in v else v
        new_fact = {"value": fact_value, "timestamp": timestamp()}
        
        # Always overwrite singleton facts with the latest
        if k in SINGLETON_CATEGORIES:
            existing[k] = new_fact
            continue

        # Handle Preference/Dislike conflicts
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

        # Merge lists or create new entries
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
            if all(entry["value"] != fact_value for entry in existing[k]):
                existing[k].append(new_fact)
        else:
            existing[k] = new_fact

    return existing

def dedupe_facts(facts):
    for k, v in list(facts.items()):
        if isinstance(v, list):
            unique = []
            seen = []
            for entry in v:
                val = entry["value"].strip().lower()
                if any(val in other or other in val for other in seen):
                    continue
                seen.append(val)
                unique.append(entry)
            facts[k] = unique
    return facts

def extract_facts(user_input, existing_facts):
    user_input_lower = user_input.lower()

    # Negation patterns (do not save if user negates)
    NEGATION_PATTERNS = [
        r"(?:don't|do not|never|not)\s+call me\s+([A-Za-z0-9_ ]+)"
    ]
    for pat in NEGATION_PATTERNS:
        if re.search(pat, user_input, re.IGNORECASE):
            return existing_facts

    # Name patterns (flexible)
    NAME_PATTERNS = [
        r"call me\s+([A-Za-z0-9_ ]+)",
        r"my name is\s+([A-Za-z0-9_ ]+)",
        r"i go by\s+([A-Za-z0-9_ ]+)",
        r"i'm called\s+([A-Za-z0-9_ ]+)",
        r"i am called\s+([A-Za-z0-9_ ]+)"
    ]
    for pat in NAME_PATTERNS:
        match = re.search(pat, user_input, re.IGNORECASE)
        if match:
            name_value = match.group(1).strip()
            existing_facts["Name"] = {"value": name_value, "timestamp": timestamp()}
            return dedupe_facts(existing_facts)

    # Mood patterns (single-word only)
    MOOD_PATTERNS = [
        r"\bi feel(?: like)?\s+([a-zA-Z]+)\b",
        r"\bi felt\s+([a-zA-Z]+)\b",
        r"\bthat made me\s+([a-zA-Z]+)\b",
        r"\bi am\s+(?:feeling\s+)?([a-zA-Z]+)\b",
        r"\bi'm\s+(?:feeling\s+)?([a-zA-Z]+)\b"
    ]
    for pat in MOOD_PATTERNS:
        mood_match = re.search(pat, user_input, re.IGNORECASE)
        if mood_match:
            mood_value = mood_match.group(1).strip()
            existing_facts["Mood"] = {"value": mood_value, "timestamp": timestamp()}
            return dedupe_facts(existing_facts)

    # Fallback: use LLM extraction
    extract_system_prompt = '''
You are a fact extractor.
ONLY record facts that the USER explicitly states about THEMSELVES.
Strict rules:
- Only record facts phrased with clear self-reference.
- Do NOT infer or guess facts.
- Categories allowed: Name, Age, Gender, Location, Nationality, Ethnicity,
  FormerJob, CurrentJob, Hobby, Interest, Preference, Dislike, Trait, Skill, Education,
  RelationshipStatus, Pet, Mood.
Output:
- Valid JSON with only NEW or UPDATED facts.
- If no facts apply, return {}.
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
    except (json.JSONDecodeError, KeyError):
        return existing_facts

    merged = merge_facts(existing_facts, new_facts)
    return dedupe_facts(merged)

IMPORTANT_KEYS = {"Name", "Age", "Location", "Mood", "Preference", "Dislike"}

def summarize_facts(facts, per_category=3):
    summary = []
    for k, v in facts.items():
        if k in IMPORTANT_KEYS:
            if isinstance(v, list):
                recent = sorted(v, key=lambda e: e["timestamp"], reverse=True)[:per_category]
                summary.extend([f"- {k}: {entry['value']}" for entry in recent])
            elif isinstance(v, dict):
                summary.append(f"- {k}: {v['value']}")
    return "\n".join(summary)

def build_prompt(user_input, history, facts):
    fact_summary = summarize_facts(facts)
    conversation = [f"user: {t['user']}\nbot: {t['bot']}" for t in history]
    conversation.append(f"user: {user_input}\nbot:")
    return f"Saved facts about user:\n{fact_summary}\n\nCurrent conversation:\n" + "\n".join(conversation)

def extract_value(v):
    if isinstance(v, dict):
        return v.get("value", ""), v.get("timestamp", "")
    return v, ""

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

def typewriter_stream(text, delay=0.02):
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)

def chat():
    save_data = load_json(SAVE_FILE, {})
    history = deque(save_data.get("history", []), maxlen=MEMORY_SIZE)
    facts = normalize_facts(load_json(FACTS_FILE, {}))

    # Show last chat info if available
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
            save_json(FACTS_FILE, facts)
            sys.exit(0)

        facts_result = {"value": None}
        def run_extraction():
            facts_result["value"] = extract_facts(user_input, facts)
        extractor_thread = threading.Thread(target=run_extraction)
        extractor_thread.start()

        print("bot is thinking...", end="", flush=True)
        prompt = build_prompt(user_input, history, facts)
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
        extractor_thread.join()
        if facts_result["value"] is not None:
            facts = facts_result["value"]

        # Save after each turn
        save_data["history"] = list(history)
        save_data["last_chat_time"] = timestamp()
        save_json(SAVE_FILE, save_data)
        save_json(FACTS_FILE, facts)

        # Print current facts
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
