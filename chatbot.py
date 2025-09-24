import json
import sys
import time
import threading
from collections import deque
from datetime import datetime
import ollama

MODEL_NAME = "mistral" # whatever ollama model name
SAVE_FILE = "conversation.json"  # conversational history
FACTS_FILE = "important_facts.json"  # long-term important facts, pruned by significance
MEMORY_SIZE = 10  # short-term conversation turns
MAX_FACTS = 10   # maximum facts saved before pruning
SYSTEM_PROMPT = '''
You are a friendly, relaxed, and conversational chatbot.
Your goal is to keep the user engaged and respond like a thoughtful friend.
Keep responses clear, natural, and casual. Show understanding, curiosity, or light humor.
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

def score_fact(key, value, provided_score=None):
    """
    Assigns significance to a fact.
    - Use LLM-provided score if available.
    - If value == 'Unknown', force very low score.
    - Otherwise fallback to neutral default.
    """
    if not value or str(value).lower() == "unknown":
        return 10
    if provided_score is not None:
        return provided_score
    return 50  # neutral default

def prune_facts(facts):
    """Keep only the top N facts by significance (breaking ties by recency)."""
    all_facts = []
    for k, v in facts.items():
        if isinstance(v, dict):
            all_facts.append((k, v))
        elif isinstance(v, list):
            for entry in v:
                all_facts.append((k, entry))

    # Sort by significance, then timestamp (newest first)
    all_facts.sort(
        key=lambda x: (x[1].get("significance", 0), x[1].get("timestamp", "")),
        reverse=True
    )

    # Trim to MAX_FACTS
    kept = all_facts[:MAX_FACTS]

    # Rebuild dict
    new_facts = {}
    for k, entry in kept:
        if k not in new_facts:
            new_facts[k] = entry
        else:
            if not isinstance(new_facts[k], list):
                new_facts[k] = [new_facts[k]]
            new_facts[k].append(entry)
    return new_facts

def merge_facts(existing, new):
    """Merge new facts with per-item timestamps and significance."""
    for k, v in new.items():
        if not v:
            continue

        if isinstance(v, dict) and "value" in v:
            val = v["value"]
            ts = v.get("timestamp", timestamp())
            sig = score_fact(k, val, v.get("significance"))
        else:
            val = v
            ts = timestamp()
            sig = score_fact(k, val)

        # Flatten list values into individual items
        if isinstance(val, list):
            for item in val:
                if isinstance(item, dict) and "value" in item and "timestamp" in item:
                    merge_facts(existing, {k: item})
                else:
                    merge_facts(existing, {k: {"value": item, "timestamp": ts, "significance": sig}})
            continue

        # Handle existing multi-item categories
        if k in existing:
            if isinstance(existing[k], list):
                if not any(entry["value"] == val for entry in existing[k]):
                    existing[k].append({"value": val, "timestamp": ts, "significance": sig})
            elif isinstance(existing[k], dict):
                existing[k] = [existing[k], {"value": val, "timestamp": ts, "significance": sig}]
        else:
            if isinstance(val, dict) and "timestamp" in val:
                if "significance" not in val:
                    val["significance"] = sig
                existing[k] = val
            else:
                existing[k] = {"value": val, "timestamp": ts, "significance": sig}
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
            summary.append(f"- {k}: {val} (sig: {v.get('significance',0)})")
        elif isinstance(v, list):
            for entry in v:
                val = entry.get("value")
                ts = entry.get("timestamp")
                if val:
                    summary.append(f"- {k}: {val} (timestamp: {ts}, sig: {entry.get('significance',0)})")
    return "\n".join(summary)

def build_prompt(user_input, history, facts):
    fact_summary = summarize_facts(facts)
    conversation = [f"user: {t['user']}\nbot: {t['bot']}" for t in history]
    conversation.append(f"user: {user_input}\nbot:")
    return f"Saved important facts about the user:\n{fact_summary}\n\nCurrent conversation:\n" + "\n".join(conversation)

def extract_important_facts(user_input, existing_facts):
    """
    Ask the LLM to identify only personally meaningful facts about the user.
    - Each fact must have { "value": ..., "significance": int 0-100, "timestamp": optional }.
    - Do not fabricate facts.
    - Do not insert placeholders like "Unknown".
    - Return only valid JSON with new or updated facts.
    """
    extract_system_prompt = '''
You are a careful fact extractor.
Rules:
- Record facts the USER explicitly states about THEMSELVES.
- For each fact, include:
  - "value": the stated information
  - "significance": integer 0–100 (100 = extremely important, 0 = trivial)
  - "timestamp": current UTC time in ISO 8601 format
- Return facts as a flat JSON object, e.g.:

{
  "Name": { "value": "Alice", "significance": 100, "timestamp": "2025-09-24T14:27:25.553946" },
  "Age": { "value": "32", "significance": 80, "timestamp": "2025-09-24T14:27:25.553946" }
}

- Never invent or insert placeholder values like "Unknown".
- If there are no new facts, return {}.
- Do not wrap the output in extra keys like "CURRENT_IMPORTANT_FACTS".
- Return ONLY valid JSON.
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
        print("\n[DEBUG] Extracted raw facts from LLM:")
        print(response["message"]["content"])
        new_facts = json.loads(response["message"]["content"])
    except Exception:
        return existing_facts

    merged = merge_facts(existing_facts, new_facts)
    deduped = dedupe_facts(merged)
    return prune_facts(deduped)

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
                    sig = v.get("significance", 0)
                    if val:
                        print(f"- {k}: {val} (timestamp: {ts}, sig: {sig})")
                elif isinstance(v, list):
                    for entry in v:
                        val = entry.get("value")
                        ts = entry.get("timestamp")
                        sig = entry.get("significance", 0)
                        if val:
                            print(f"- {k}: {val} (timestamp: {ts}, sig: {sig})")

if __name__ == "__main__":
    chat()
