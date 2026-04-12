"""AI Agent with persistent memory using LangGraph + Ollama + MCP."""

import os
import sqlite3
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.sqlite import SqliteSaver

from tools import get_current_datetime, calculator, note_taker

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3.5:9b")
DB_PATH = os.path.join(os.path.dirname(__file__), "memory.db")

SYSTEM_PROMPT = """
You are a helpful assistant with browser and utility tools.

Think step by step before acting:

1. UNDERSTAND the user's intent — what do they need and what kind of source would best answer it? divide the user's query to understand the main components and understand the context.
   - Real-time or web info → use browser tools
   - Math → use calculator
   - Time → use get_current_datetime
   - Notes → use note_taker

2. PLAN the right URL if browsing is needed:
   - Weather → https://weather.com/weather/today/l/LOCATION
   - Courses → https://www.udemy.com/courses/search/?q=QUERY
   - General → https://www.google.com/search?q=QUERY

3. BROWSE using this exact sequence:
   a. take_snapshot — identify available pages
   b. navigate_page — go to the planned URL (requires page number from snapshot)
   c. wait_for — confirm the page loaded by waiting for expected text
   d. take_snapshot — read the loaded page content
   e. click/fill/press_key — interact with the page if needed to get the answer

4. RESPOND concisely with only the relevant result.
   - VERY IMPORTANT: Do NOT use any language other than English in your response, even if the user input is in another language. Always respond in English.
   - Do NOT describe page structure
   - Do NOT suggest code
   - Do NOT make up information available on the web
   - Use the user query and combine it with the information you found to give a direct answer e.g. query: "What's the weather in Paris?" → "The weather in Paris is currently 18°C with light rain." (instead of "I found the weather information on this page...")
"""


def create_agent(thread_id: str = "default"):
    """Create a ReAct agent with persistent SQLite memory."""
    llm = ChatOllama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL)
    tools = [get_current_datetime, calculator, note_taker]

    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    memory = SqliteSaver(conn)

    agent = create_react_agent(
        model=llm,
        tools=tools,
        checkpointer=memory,
        prompt=SYSTEM_PROMPT,
    )
    return agent, {"configurable": {"thread_id": thread_id}}, conn


async def create_agent_with_mcp(thread_id: str = "default"):
    """Create a ReAct agent with MCP browser tools + local tools."""
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

    # Essential browser tools
    keep = {"navigate_page", "take_snapshot", "click", "fill", "press_key", "wait_for", "list_pages"}
    mcp_tools = [t for t in mcp_tools if t.name in keep]

    all_tools = [get_current_datetime, calculator, note_taker] + mcp_tools

    aio_conn = await aiosqlite.connect(DB_PATH)
    memory = AsyncSqliteSaver(aio_conn)

    agent = create_react_agent(
        model=llm,
        tools=all_tools,
        checkpointer=memory,
        prompt=SYSTEM_PROMPT,
    )
    return agent, {"configurable": {"thread_id": thread_id}}, aio_conn
