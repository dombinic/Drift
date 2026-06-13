"""Quickstart: Using Drift with a LangChain agent.

Requires:
    pip install driftguard[langchain] langchain langchain-openai

Run with:
    OPENAI_API_KEY=... python examples/quickstart_langchain.py
"""

from drift import DriftGuard
from drift.callbacks.langchain import DriftCallbackHandler

# 1. Create the guard
guard = DriftGuard(
    on_anomaly=lambda a: print(f"🚨 {a}")  # Real-time alerts
)

# 2. Create the callback handler
handler = DriftCallbackHandler(guard)

# 3. Use with any LangChain agent, chain, or LLM
# 
# from langchain_openai import ChatOpenAI
# from langchain.agents import create_react_agent, AgentExecutor
#
# llm = ChatOpenAI(model="gpt-4o", callbacks=[handler])
# agent = AgentExecutor(agent=..., tools=..., callbacks=[handler])
# 
# # Run your agent as normal — Drift monitors in the background
# for query in queries:
#     agent.invoke({"input": query})

# 4. Get the report
guard.report()
