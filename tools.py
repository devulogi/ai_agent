"""Agent tools — get_current_datetime, calculator, note_taker, wikipedia_search."""

import ast
import json
import operator
import os
import urllib.parse
import urllib.request
from datetime import datetime, timezone

from langchain_core.tools import tool

NOTES_FILE = os.path.join(os.path.dirname(__file__), "notes.json")


@tool
def get_current_datetime() -> str:
    """Returns current date and time in YYYY-MM-DD HH:MM:SS UTC format."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


# Safe math evaluator — no eval(), no exec(), no code injection.
_MATH_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
}


def _safe_eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _MATH_OPS:
        return _MATH_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _MATH_OPS:
        return _MATH_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"Unsupported operation: {ast.dump(node)}")


@tool
def calculator(expression: str) -> str:
    """Evaluates a mathematical expression and returns the result.
    Examples: '2 + 2', '3 * (4 + 5)', '100 / 7', '348 * 0.15'
    """
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        result = _safe_eval(tree.body)
        return str(round(result, 10))
    except ZeroDivisionError:
        return "Error: division by zero"
    except Exception as e:
        return f"Error: {e}"


@tool
def note_taker(action: str, content: str = "") -> str:
    """Saves, retrieves, searches, or deletes notes from a local JSON file.
    action: 'save' to save a new note (requires content),
            'list' to list all notes,
            'search' to find notes matching a keyword (requires content as the search term),
            'delete' to delete a note by ID (requires content as the note ID number).
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

    elif action == "search":
        if not content:
            return "Error: content is required as the search keyword."
        if not os.path.exists(NOTES_FILE):
            return "No notes found."
        with open(NOTES_FILE, "r") as f:
            notes = json.load(f)
        keyword = content.lower()
        matches = [n for n in notes if keyword in n["content"].lower()]
        if not matches:
            return f"No notes matching '{content}'."
        return "\n".join(f"[{n['id']}] {n['content']} ({n['timestamp']})" for n in matches)

    elif action == "delete":
        if not content:
            return "Error: content must be the note ID to delete."
        if not os.path.exists(NOTES_FILE):
            return "No notes found."
        with open(NOTES_FILE, "r") as f:
            notes = json.load(f)
        try:
            target_id = int(content)
        except ValueError:
            return f"Error: '{content}' is not a valid note ID."
        updated = [n for n in notes if n["id"] != target_id]
        if len(updated) == len(notes):
            return f"No note with ID {target_id} found."
        with open(NOTES_FILE, "w") as f:
            json.dump(updated, f, indent=2)
        return f"Note #{target_id} deleted."

    return f"Unknown action '{action}'. Use 'save', 'list', 'search', or 'delete'."


@tool
def wikipedia_search(query: str) -> str:
    """Searches Wikipedia and returns a summary for the given query.
    Use this to ground factual answers and support multi-hop reasoning.
    Examples: 'boiling point of water', 'first iPhone release date'
    """
    try:
        encoded = urllib.parse.quote(query)
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded}"
        req = urllib.request.Request(url, headers={"User-Agent": "ReActAgent/1.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode())
        if data.get("type") == "disambiguation":
            return f"Disambiguation page. Try a more specific query. Suggestions: {data.get('title')}"
        extract = data.get("extract", "")
        return extract[:1200] if extract else "No summary found."
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return f"No Wikipedia article found for '{query}'."
        return f"HTTP error {e.code} fetching Wikipedia."
    except Exception as e:
        return f"Error: {e}"
