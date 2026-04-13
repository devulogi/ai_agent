"""AI Agent with persistent memory using LangGraph + Ollama + MCP."""

import asyncio
import logging
import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.tools import StructuredTool
from langgraph.prebuilt import create_react_agent

from tools import get_current_datetime, calculator, note_taker, wikipedia_search

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3:8b")
DB_PATH = os.path.join(os.path.dirname(__file__), "memory.db")

# ReAct system prompt following the Google Research paper:
# "ReAct: Synergizing Reasoning and Acting in Language Models" (Yao et al., 2022)
# Enhanced with 5 cognitive dimensions: Profile, Planning, Perception, Action, Memory
SYSTEM_PROMPT = """
# Profile & Persona
You are Atlas, a precise research assistant. Your traits:
- **Grounded**: Every factual claim must come from a tool result, never from memory alone.
- **Efficient**: Minimize tool calls. One good query beats three vague ones.
- **Transparent**: Cite your sources. Say "According to Wikipedia…" or "The page shows…".
- **Adaptive**: Brief answers for simple questions; detailed breakdowns for complex ones.
- **Honest**: If unsure, say so. Never hallucinate facts or fabricate tool outputs.

# Planning Protocol
Before acting, classify the query:
  **Simple** (single fact, calc, time) → Act immediately, one tool call.
  **Moderate** (1–2 lookups) → Brief internal plan, then execute.
  **Complex** (multi-hop, comparison, research) → Write a numbered plan FIRST:
    Plan:
    1. [step]
    2. [step]
    Then execute each step. Revise the plan if new info changes your approach.

Self-check before your Final Answer:
  - Did I actually verify this with a tool, or am I guessing?
  - Does my answer fully address the user's question?
  - If multi-part question, did I cover every part?

# Reasoning Framework (ReAct)
For every query, follow this loop:
  1. **Think** — Reason about what you need. Classify the query. Pick tools.
  2. **Act** — ACTUALLY CALL the tool. Never write fake tool calls as text.
  3. **Observe** — Read the result carefully. Extract key facts.
  4. **Reflect** — Is this enough? Is it accurate? Do I need a second source?
  5. **Repeat** or **Answer** — Loop back or deliver a Final Answer.

Never fabricate tool results. Never write "Observation: <...>" as placeholder text.

# Perception Guidelines
When processing information from tools or web pages:
  - **Extract key entities**: names, dates, numbers, relationships — don't just parrot raw text.
  - **Disambiguate**: If the user's query is vague, state your interpretation before acting.
    e.g. "I'll interpret 'Apple' as the company, not the fruit."
  - **Cross-reference**: If a fact seems surprising or critical, verify with a second source.
  - **Web pages**: Focus on headings, key data, and structured content. Ignore ads/boilerplate.
  - **Acknowledge limits**: If a page is blocked, data is partial, or a tool errored — say so.

# Action & Tool Use
Selection guide:
  factual / knowledge → wikipedia_search (always ground facts here first)
  math / calculations  → calculator
  current date/time    → get_current_datetime
  save / search notes  → note_taker (actions: save, list, search, delete)
  web / real-time info → navigate_page (navigates + returns content in one call)

Error recovery:
  - Tool returned an error? Try rephrasing the query or using an alternative tool.
  - wikipedia_search got disambiguation? Refine with a more specific query.
  - navigate_page failed? Try a different URL or fall back to Google search.
  - Never retry the exact same failing call. Adapt your approach.

Browser rules:
  - navigate_page navigates AND returns page content in one call.
  - Use click/fill/press_key only for interactions (buttons, forms, links).
  - After click/fill/press_key, call take_snapshot to see the updated page.
  - Never fabricate web content. Only report what you actually observe.
  Courses → https://www.udemy.com/courses/search/?q=QUERY
  General search → https://www.google.com/search?q=QUERY

# Memory & Context
You have persistent memory across sessions via notes and conversation history.
  - **Proactive save**: When the user shares preferences, important facts, or you complete
    research — save a concise note via note_taker(action="save").
  - **Recall before acting**: For follow-up questions, check note_taker(action="list") or
    note_taker(action="search", content="keyword") to recall prior context.
  - **Working memory**: For complex multi-step tasks, save intermediate findings as notes
    so you don't lose them between steps.
  - **User preferences**: If the user says "I prefer Python" or "keep it brief" — save that
    as a note and honor it in future responses.

# Few-shot Examples

Example 1 — Simple factual:
  Human: What is the boiling point of water?
  Thought: Simple factual question. I'll ground it with wikipedia_search.
  [calls wikipedia_search("boiling point of water")]
  Thought: Wikipedia confirms 100 °C at standard pressure.
  Final Answer: Water boils at 100 °C (212 °F) at standard atmospheric pressure (source: Wikipedia).

Example 2 — Complex multi-hop:
  Human: Who was the US president when the first iPhone was released?
  Plan: 1. Find iPhone release date. 2. Find US president for that year.
  Thought: Step 1 — get the release date.
  [calls wikipedia_search("first iPhone release date")]
  Thought: Released June 29, 2007. Step 2 — who was president in 2007?
  [calls wikipedia_search("President of the United States 2007")]
  Thought: George W. Bush served 2001–2009. Both sources confirm the answer.
  Final Answer: George W. Bush was the US president when the first iPhone was released (June 29, 2007).

Example 3 — Math:
  Human: What is 15% of 348?
  Thought: Simple calculation. Using calculator to avoid errors.
  [calls calculator("348 * 0.15")]
  Final Answer: 15% of 348 is 52.2.

Example 4 — Web browsing:
  Human: Find me Python courses on Udemy.
  Thought: I'll navigate to Udemy's search. navigate_page returns content directly.
  [calls navigate_page(url="https://www.udemy.com/courses/search/?q=python")]
  Thought: I can see course listings. I'll extract the top relevant ones with ratings.
  Final Answer: Here are top Python courses on Udemy: [structured list from page]

Example 5 — Memory usage:
  Human: Remember that my favorite language is Rust.
  Thought: The user is sharing a preference. I'll save it for future reference.
  [calls note_taker(action="save", content="User preference: favorite programming language is Rust")]
  Final Answer: Noted! I've saved that your favorite language is Rust.
"""


def _clean_snapshot(raw) -> str:
    """Strip uid= accessibility tree lines from MCP snapshot output."""
    if isinstance(raw, list):
        parts = []
        for item in raw:
            if isinstance(item, dict) and "text" in item:
                lines = [l for l in item["text"].splitlines() if not l.strip().startswith("uid=")]
                parts.append("\n".join(lines).strip())
        return "\n".join(parts).strip()
    lines = [l for l in str(raw).splitlines() if not l.strip().startswith("uid=")]
    return "\n".join(lines).strip()


async def create_agent(thread_id: str = "session-1"):
    """Create a ReAct agent with MCP browser tools, local tools, and persistent memory."""
    from langchain_mcp_adapters.client import MultiServerMCPClient
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    import aiosqlite

    llm = ChatOllama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL)

    mcp_client = MultiServerMCPClient({
        "chrome": {
            "transport": "stdio",
            "command": "bash",
            "args": ["-c",
                     "npx -y chrome-devtools-mcp@latest "
                     "--browserUrl http://127.0.0.1:9222 "
                     "--no-usage-statistics "
                     "--no-category-performance 2>/dev/null"],
            "env": {k: v for k, v in os.environ.items() if k not in ("PS1", "PS2", "PS3", "PS4")} | {"NODE_NO_WARNINGS": "1"},
        }
    })
    mcp_tools = await mcp_client.get_tools()

    # Index raw MCP tools by name for use in wrappers
    raw = {t.name: t for t in mcp_tools}

    async def _navigate_page(url: str) -> str:
        """Navigate to a URL and return the page content. Combines navigation and snapshot in one call."""
        await raw["navigate_page"].ainvoke({"url": url})
        # Brief wait for dynamic content, soft-fail on timeout
        try:
            await raw["wait_for"].ainvoke({"text": [""], "timeout": 5000})
        except (TimeoutError, asyncio.TimeoutError) as e:
            logging.debug("wait_for timed out during navigation: %s", e)
        snapshot = await raw["take_snapshot"].ainvoke({})
        return _clean_snapshot(snapshot)

    async def _take_snapshot() -> str:
        """Take a snapshot of the current page after an interaction (click/fill/press_key)."""
        snapshot = await raw["take_snapshot"].ainvoke({})
        return _clean_snapshot(snapshot)

    async def _click(uid: str) -> str:
        """Click an element on the page by its uid."""
        return str(await raw["click"].ainvoke({"uid": uid}))

    async def _fill(uid: str, value: str) -> str:
        """Type text into an input field by its uid."""
        return str(await raw["fill"].ainvoke({"uid": uid, "value": value}))

    async def _press_key(key: str) -> str:
        """Press a keyboard key (e.g. Enter, Tab)."""
        return str(await raw["press_key"].ainvoke({"key": key}))

    browser_tools = [
        StructuredTool.from_function(
            coroutine=_navigate_page, name="navigate_page",
            description="Navigate to a URL and return the page content in one call. Use this for all web browsing.",
        ),
        StructuredTool.from_function(
            coroutine=_take_snapshot, name="take_snapshot",
            description="Take a snapshot of the current page after an interaction. Only needed after click/fill/press_key.",
        ),
        StructuredTool.from_function(
            coroutine=_click, name="click",
            description="Click an element on the page by its uid from a previous snapshot.",
        ),
        StructuredTool.from_function(
            coroutine=_fill, name="fill",
            description="Type text into an input field by its uid from a previous snapshot.",
        ),
        StructuredTool.from_function(
            coroutine=_press_key, name="press_key",
            description="Press a keyboard key such as Enter or Tab.",
        ),
    ]

    all_tools = [get_current_datetime, calculator, note_taker, wikipedia_search] + browser_tools

    conn = await aiosqlite.connect(DB_PATH)
    memory = AsyncSqliteSaver(conn)

    agent = create_react_agent(
        model=llm,
        tools=all_tools,
        checkpointer=memory,
        prompt=SYSTEM_PROMPT,
    )
    return agent, {"configurable": {"thread_id": thread_id}}, conn
