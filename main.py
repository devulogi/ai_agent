"""CLI entry point for the AI Agent."""

import asyncio
import glob
import os
import re
import sys
from agent import create_agent, create_agent_with_mcp, DB_PATH


def fresh_db():
    """Delete memory DB files for a clean start."""
    for f in glob.glob(DB_PATH + "*"):
        os.remove(f)


def strip_think(text: str) -> str:
    """Remove <think>...</think> blocks from Qwen's output."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


async def main_mcp():
    print("=" * 50)
    print("🤖 AI Agent with Memory + Browser (MCP)")
    print("=" * 50)
    print("Tools: local tools + Chrome DevTools (full)")
    print("Type 'quit' to exit, 'new' for a new session.\n")

    thread_id = "session-1"
    agent, config, conn = await create_agent_with_mcp(thread_id)

    while True:
        try:
            user_input = await asyncio.get_event_loop().run_in_executor(
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
        if user_input.lower() == "new":
            await conn.close()
            thread_id = input("New session ID: ").strip() or "session-1"
            agent, config, conn = await create_agent_with_mcp(thread_id)
            print(f"🔄 Switched to session: {thread_id}\n")
            continue

        try:
            response = await agent.ainvoke(
                {"messages": [("human", user_input)]},
                config=config,
            )
            ai_message = response["messages"][-1]
            print(f"\n{'─' * 50}")
            print(f"📝 Query: {user_input}")
            print(f"🤖 Agent: {strip_think(ai_message.content)}")
            print(f"{'─' * 50}\n")
        except Exception as e:
            if "INVALID_CHAT_HISTORY" in str(e):
                print("\n⚠️  Corrupt session history detected. Starting fresh session...\n")
                await conn.close()
                fresh_db()
                agent, config, conn = await create_agent_with_mcp(thread_id)
            else:
                print(f"\n❌ Error: {e}\n")

    await conn.close()


def main_local():
    print("=" * 50)
    print("🤖 AI Agent with Memory")
    print("=" * 50)
    print("Tools: get_current_datetime, calculator, note_taker")
    print("Type 'quit' to exit, 'new' for a new session.\n")

    thread_id = "session-1"
    agent, config, conn = create_agent(thread_id)

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Bye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("👋 Bye!")
            break
        if user_input.lower() == "new":
            conn.close()
            thread_id = input("New session ID: ").strip() or "session-1"
            agent, config, conn = create_agent(thread_id)
            print(f"🔄 Switched to session: {thread_id}\n")
            continue

        try:
            response = agent.invoke(
                {"messages": [("human", user_input)]},
                config=config,
            )
            ai_message = response["messages"][-1]
            print(f"\n{'─' * 50}")
            print(f"📝 Query: {user_input}")
            print(f"🤖 Agent: {strip_think(ai_message.content)}")
            print(f"{'─' * 50}\n")
        except Exception as e:
            if "INVALID_CHAT_HISTORY" in str(e):
                print("\n⚠️  Corrupt session history detected. Starting fresh session...\n")
                conn.close()
                fresh_db()
                agent, config, conn = create_agent(thread_id)
            else:
                print(f"\n❌ Error: {e}\n")

    conn.close()


if __name__ == "__main__":
    fresh_db()
    if "--mcp" in sys.argv:
        asyncio.run(main_mcp())
    else:
        main_local()
