"""CLI entry point for the AI Agent."""

import asyncio
import glob
import os
import re
from agent import create_agent, DB_PATH
from langchain_core.messages import AIMessage, ToolMessage


def fresh_db():
    """Delete memory DB files for a clean start."""
    for f in glob.glob(DB_PATH + "*"):
        os.remove(f)


def strip_think(text: str) -> str:
    """Remove <think>...</think> blocks from Qwen's output."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _clean_observation(content) -> str:
    """Extract only human-readable text from a tool message, stripping raw snapshot trees."""
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                text = item["text"]
                # Strip the accessibility tree dump (lines starting with uid=)
                lines = [l for l in text.splitlines() if not l.strip().startswith("uid=")]
                parts.append("\n".join(lines).strip())
        text = "\n".join(parts).strip()
    else:
        text = str(content)
        lines = [l for l in text.splitlines() if not l.strip().startswith("uid=")]
        text = "\n".join(lines).strip()
    return text[:300] + "..." if len(text) > 300 else text


def print_react_trajectory(messages: list) -> None:
    """Print the Thought/Action/Observation trajectory per the ReAct paper."""
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                print(f"  💡 Action      → {tc['name']}({tc['args']})")
        elif isinstance(msg, ToolMessage):
            print(f"  🔍 Observation → {_clean_observation(msg.content)}")


async def main():
    print("=" * 50)
    print("🤖 Atlas — AI Agent with Memory + Browser (MCP)")
    print("=" * 50)
    print("Commands: 'quit' to exit, 'new' to switch session, 'reset' to wipe memory.\n")

    thread_id = "session-1"
    agent, config, conn = await create_agent(thread_id)

    while True:
        try:
            loop = asyncio.get_event_loop()
            user_input = await loop.run_in_executor(
                None, lambda: input("You: ").strip()
            )
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Bye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("👋 Bye!")
            break
        if user_input.lower() == "reset":
            await conn.close()
            fresh_db()
            agent, config, conn = await create_agent(thread_id)
            print("🧹 Memory wiped. Fresh start.\n")
            continue
        if user_input.lower() == "new":
            await conn.close()
            thread_id = input("New session ID: ").strip() or "session-1"
            agent, config, conn = await create_agent(thread_id)
            print(f"🔄 Switched to session: {thread_id}\n")
            continue

        try:
            response = await agent.ainvoke(
                {"messages": [("human", user_input)]},
                config=config,
            )
            messages = response["messages"]
            ai_message = messages[-1]

            print(f"\n{'─' * 50}")
            print(f"📝 Query: {user_input}")
            print_react_trajectory(messages)
            print(f"🤖 Agent: {strip_think(ai_message.content)}")
            print(f"{'─' * 50}\n")
        except Exception as e:
            if "INVALID_CHAT_HISTORY" in str(e):
                print("\n⚠️  Corrupt session history detected. Starting fresh session...\n")
                await conn.close()
                fresh_db()
                agent, config, conn = await create_agent(thread_id)
            else:
                print(f"\n❌ Error: {e}\n")

    await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
