"""Agent tools — get_current_datetime, calculator, note_taker."""

import json
import os
from datetime import datetime

from langchain_core.tools import tool

NOTES_FILE = os.path.join(os.path.dirname(__file__), "notes.json")


@tool
def get_current_datetime() -> str:
    """Returns current date and time in YYYY-MM-DD HH:MM:SS format."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def calculator(expression: str) -> str:
    """Evaluates a mathematical expression and returns the result.
    Examples: '2 + 2', '3 * (4 + 5)', '100 / 7'
    """
    if "**" in expression:
        return "Error: exponentiation is not allowed"
    allowed = set("0123456789+-*/.() ")
    if not all(c in allowed for c in expression):
        return "Error: expression contains invalid characters"
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {e}"


@tool
def note_taker(action: str, content: str = "") -> str:
    """Saves or retrieves notes from a local JSON file.
    action: 'save' to save a new note (requires content), 'list' to list all notes.
    """
    if action == "save":
        if not content:
            return "Error: content is required to save a note."
        notes = []
        if os.path.exists(NOTES_FILE):
            with open(NOTES_FILE, "r") as f:
                notes = json.load(f)
        note = {
            "id": len(notes) + 1,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        }
        notes.append(note)
        with open(NOTES_FILE, "w") as f:
            json.dump(notes, f, indent=2)
        return f"Note #{note['id']} saved."

    elif action == "list":
        if not os.path.exists(NOTES_FILE):
            return "No notes found."
        with open(NOTES_FILE, "r") as f:
            notes = json.load(f)
        if not notes:
            return "No notes found."
        return "\n".join(f"[{n['id']}] {n['content']} ({n['timestamp']})" for n in notes)

    return f"Unknown action '{action}'. Use 'save' or 'list'."
