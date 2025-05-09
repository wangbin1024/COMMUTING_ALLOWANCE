import streamlit as st
import sys
import os
from langchain_core.messages import HumanMessage
from langchain.callbacks import LangChainTracer

# LangSmith API設定
LANGSMITH_API_KEY = "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"
LANGSMITH_PROJECT = "agent-eval-project"
os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_API_KEY
os.environ["LANGCHAIN_PROJECT"] = LANGSMITH_PROJECT

# agent.py があるディレクトリを追加
module_dir = "/Workspace/Users/wang-b2@itec.hankyu-hanshin.co.jp/COMMUTING_ALLOWANCE/commuting_allowance"
if module_dir not in sys.path:
  sys.path.append(module_dir)

from agent import agent  # agent = create_tool_calling_agent(...)

st.title("AIエージェント Demo")

user_input = st.text_input("質問を入力してください:")

if st.button("送信") and user_input:
  st.write(f"あなたの入力: {user_input}")
  with st.spinner("AIエージェントが考え中..."):
    tracer = LangChainTracer()
    config = {"configurable": {"thread_id": "session_00001"}, "callbacks": [tracer]}

    reply = ""
    events = agent.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        config,
        stream_mode="values",
    )

    for event in events:
      if "messages" in event:
        for msg in event["messages"]:
          role = msg.get("role", "unknown")
          content = msg.get("content", "")
          reply += f"[{role}] {content}\n"

    if reply:
      st.success(reply)
    else:
      st.warning("応答がありませんでした。")
