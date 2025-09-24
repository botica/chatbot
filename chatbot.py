import json
import sys
import time
import threading
from collections import deque
from datetime import datetime
import ollama

# ----------------- Configuration -----------------
MODEL_NAME = "mistral"
SAVE_FILE = "conversation.json"  # conversation history
FACTS_FILE = "important_facts.json"  # long-term important facts
MEMORY_SIZE = 2  # short-term conversation turns
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
    except (FileNotFoundError, json.JSONDecodeError):
        return default

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def typewriter_stream(text, delay=0.02):
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

# ----------------- Memory Management -----------------
def merge_facts(existing, new):
    """Merge new facts with per-item timestamps, handling nested dicts correctly."""
    for k, v in new.items():
        if not v:
            continue
        # Determine timestamp and value
        if isinstance(v, dict) and "value" in v:
            val = v["value"]
            ts = v.get("timestamp", timestamp())
        else:
            val = v
            ts = timestamp()

        # Flatten list values into individual items
        if isinstance(val, list):
            for item in val:
                # If item is already a dict with timestamp, pass as-is
                if isinstance(item, dict) and "value" in item and "timestamp" in item:
                    merge_facts(existing, {k: item})
                else:
                    merge_facts(existing, {k: {"value": item, "timestamp": ts}})
            continue

        # Handle existing multi-item categories
        if k in existing:
            if isinstance(existing[k], list):
                if not any(entry["value"] == val for entry in existing[k]):
                    existing[k].append({"value": val, "timestamp": ts})
            elif isinstance(existing[k], dict):
                existing[k] = [existing[k], {"value": val, "timestamp": ts}]
        else:
            # If val is already a dict with its own timestamp, store as-is
            if isinstance(val, dict) and "timestamp" in val:
                existing[k] = val
            else:
                existing[k] = {"value": val, "timestamp": ts}
    return existing

def dedupe_facts(facts):
    """Remove duplicate values for multi-valued facts (if any)."""
    for k, v in list(facts.items()):
        if isinstance(v, dict):
            continue
        elif isinstance(v, list):
            unique = {}
            for entry in v:
                if isinstance(entry, dict):
                    val = entry.get("value")
                    ts = entry.get("timestamp", "")
                    val_key = str(val).lower()
                    if val_key not in unique or ts > unique[val_key].get("timestamp", ""):
                        unique[val_key] = entry
            facts[k] = list(unique.values())
    return facts

def summarize_facts(facts):
    summary = []
    for k, v in facts.items():
        if isinstance(v, dict) and v.get("value"):
            val = v["value"]
            if isinstance(val, dict):
                val = json.dumps(val)
            summary.append(f"- {k}: {val}")
        elif isinstance(v, list):
            for entry in v:
                val = entry.get("value")
                ts = entry.get("timestamp")
                if isinstance(val, dict):
                    val = json.dumps(val)
                if val:
                    summary.append(f"- {k}: {val} (timestamp: {ts})")
    return "\n".join(summary)

def build_prompt(user_input, history, facts):
    fact_summary = summarize_facts(facts)
    conversation = [f"user: {t['user']}\nbot: {t['bot']}" for t in history]
    conversation.append(f"user: {user_input}\nbot:")
    return f"Saved important facts about the user:\n{fact_summary}\n\nCurrent conversation:\n" + "\n".join(conversation)

# ----------------- Fact Extraction -----------------
def extract_important_facts(user_input, existing_facts):
    """
    Ask the LLM to identify only personally meaningful facts about the user.
    Conservative: do NOT record facts about other people, shows, or events.
    """
    extract_system_prompt = '''
You are a careful fact extractor.
- Record facts that the USER explicitly states about THEMSELVES.
- If the input does not contain any facts, return {}.
- Return ONLY valid JSON with new or updated facts.
'''
    extract_prompt = f'''
USER INPUT:
"{user_input}"

CURRENT IMPORTANT FACTS (JSON):
{json.dumps(existing_facts, indent=2)}

Return new or updated facts only in JSON.
'''
    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": extract_system_prompt},
                {"role": "user", "content": extract_prompt}
            ]
        )
        # Debug: print raw LLM output
        print("\n[DEBUG] Extracted raw facts from LLM:")
        print(response["message"]["content"])
        new_facts = json.loads(response["message"]["content"])
    except Exception:
        return existing_facts

    merged = merge_facts(existing_facts, new_facts)
    return dedupe_facts(merged)

# ----------------- Main Chat Loop -----------------
def chat():
    save_data = load_json(SAVE_FILE, {})
    history = deque(save_data.get("history", []), maxlen=MEMORY_SIZE)
    facts = load_json(FACTS_FILE, {})

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

        # Extract important facts in a separate thread
        def run_extraction():
            facts_result["value"] = extract_important_facts(user_input, facts)

        extractor_thread = threading.Thread(target=run_extraction)
        extractor_thread.start()

        # Build prompt for response
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

        # Save state
        save_data["history"] = list(history)
        save_data["last_chat_time"] = timestamp()
        save_json(SAVE_FILE, save_data)
        save_json(FACTS_FILE, facts)

        # Print important facts
        if facts:
            print("\n[important facts about user]")
            for k, v in facts.items():
                if isinstance(v, dict):
                    val = v.get("value")
                    ts = v.get("timestamp")
                    if isinstance(val, dict):
                        val = json.dumps(val)
                    if val:
                        print(f"- {k}: {val} (timestamp: {ts})")
                elif isinstance(v, list):
                    for entry in v:
                        val = entry.get("value")
                        ts = entry.get("timestamp")
                        if isinstance(val, dict):
                            val = json.dumps(val)
                        if val:
                            print(f"- {k}: {val} (timestamp: {ts})")

# ----------------- Entry Point -----------------
if __name__ == "__main__":
    chat()
