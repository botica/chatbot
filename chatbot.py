import json
import sys
from collections import deque
import ollama
from datetime import datetime
import re

MODEL_NAME = 'mistral'
SAVE_FILE = 'conversation.json'
FACTS_FILE = 'facts.json'
MEMORY_SIZE = 6

SYSTEM_PROMPT = '''
You are a friendly, empathetic companion for the user.
Your goal is to have natural, casual conversations.
Mirror the user’s energy and current mood when appropriate.
Use short, engaging responses (max 5 lines).
Acknowledge explicit user facts politely.
If you know the user’s PreferredName, use it instead of full Name.
'''

# ---------------- Timestamp ----------------
def timestamp():
    return datetime.utcnow().isoformat()

# ---------------- JSON Helpers ----------------
def load_json(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# ---------------- Facts Helpers ----------------
def normalize_facts(facts):
    """Normalize all facts to dicts with value + timestamp, flattening recursively."""
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
    """Merge new facts into existing, handling singleton categories and duplicates."""
    SINGLETON_CATEGORIES = {"Age", "Name", "Gender", "Location", "Nationality", "Ethnicity", "Mood"}

    for k, v in new.items():
        if isinstance(v, list):
            for item in v:
                merge_facts(existing, {k: item})
            continue

        fact_value = v["value"] if isinstance(v, dict) and "value" in v else v
        new_fact = {"value": fact_value, "timestamp": timestamp()}

        # Handle contradictions
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

        # Mood overwrites
        if k == "Mood":
            existing["Mood"] = new_fact
            continue

        # Singleton overwrite
        if k in SINGLETON_CATEGORIES:
            existing[k] = new_fact
            continue

        # Merge multi-value
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
    """Remove near-duplicates in multi-value categories (substrings etc.)."""
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

# ---------------- Explicit Self-Fact Patterns ----------------
SELF_FACT_PATTERNS = [
    r"\bi am\b", r"\bi'm\b", r"\bmy name is\b", r"\bi like\b", r"\bi love\b",
    r"\bi enjoy\b", r"\bi hate\b", r"\bi dislike\b", r"\bi play\b", r"\bi watch\b",
    r"\bi read\b", r"\bi eat\b", r"\bi drink\b", r"\bi live in\b", r"\bi'm from\b",
    r"\bi work as\b", r"\bmy job is\b", r"\bi know how to\b", r"\bi'm good at\b",
]

def is_explicit_self_fact(user_input: str) -> bool:
    """Check if user input matches any explicit self-fact pattern."""
    text = user_input.lower()
    return any(re.search(pat, text) for pat in SELF_FACT_PATTERNS)

# ---------------- Fact Extraction ----------------
def extract_facts(user_input, existing_facts):
    """
    Extract explicit self-facts from user input.
    Handles 'call me X' as PreferredName.
    Captures moods from common patterns.
    """
    user_input_lower = user_input.lower()

    # Explicit self-fact patterns
    SELF_FACT_PATTERNS = [
        r"\bi am\b", r"\bi'm\b", r"\bmy name is\b", r"\bi like\b", r"\bi love\b",
        r"\bi enjoy\b", r"\bi hate\b", r"\bi dislike\b", r"\bi play\b", r"\bi watch\b",
        r"\bi read\b", r"\bi eat\b", r"\bi drink\b", r"\bi live in\b", r"\bi'm from\b",
        r"\bi work as\b", r"\bmy job is\b", r"\bi know how to\b", r"\bi'm good at\b",
        r"\bcall me\b"
    ]

    # Mood-specific patterns (fixed)
    MOOD_PATTERNS = [
        r"\bi feel\s+([a-zA-Z ]+)",                   # I feel happy
        r"\bi felt\s+([a-zA-Z ]+)",                   # I felt sad
        r"\bthat made me\s+([a-zA-Z ]+)",            # That made me excited
        r"\bi am\s+(?:feeling\s+)?([a-zA-Z]{1,20})\b",   # I am anxious / I am feeling tired
        r"\bi'm\s+(?:feeling\s+)?([a-zA-Z]{1,20})\b"     # I'm anxious / I'm feeling tired
    ]

    # Skip extraction if no patterns match
    if not any(re.search(pat, user_input_lower) for pat in SELF_FACT_PATTERNS + MOOD_PATTERNS):
        return existing_facts

    # Handle "call me X" as PreferredName
    match = re.search(r"call me\s+([A-Za-z0-9_ ]+)", user_input, re.IGNORECASE)
    if match:
        nickname = match.group(1).strip()
        existing_facts["PreferredName"] = {"value": nickname, "timestamp": timestamp()}
        return dedupe_facts(existing_facts)

    # Handle mood patterns
    for pat in MOOD_PATTERNS:
        mood_match = re.search(pat, user_input, re.IGNORECASE)
        if mood_match:
            mood_value = mood_match.group(1).strip()
            existing_facts["Mood"] = {"value": mood_value, "timestamp": timestamp()}
            return dedupe_facts(existing_facts)

    # Fallback: use LLM for other explicit self-facts
    extract_system_prompt = '''
You are a fact extractor.
ONLY record facts that the USER explicitly states about THEMSELVES.

Strict rules:
- Only record facts phrased with clear self-reference:
    • "I am...", "I'm...", "My name is..."
    • "I like/love/enjoy...", "I hate/dislike..."
    • "I play...", "I watch...", "I read..."
    • "I eat...", "I drink..."
    • "I live in...", "I'm from..."
    • "I work as...", "My job is..."
    • "I know how to...", "I'm good at..."
- Do NOT infer or guess facts.
- Do NOT record facts about other people, fictional characters, or places unless explicitly tied to the user.
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

# ---------------- Prompt Builder ----------------
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

# ---------------- Chat Loop ----------------
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

        # Extract only explicit self-facts
        facts = extract_facts(user_input, facts)

        # Print facts
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
