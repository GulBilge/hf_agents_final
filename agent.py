import os, json, difflib
from typing import List, TypedDict, Optional
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langchain_huggingface import HuggingFaceEndpoint

from tools import multiply, add, subtract, divide, modulus, wiki_search, web_search, arvix_search, get_youtube_transcript, transcribe_audio

# --- Ayarlar ---
load_dotenv()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SYSTEM_PROMPT_PATH = os.path.join(BASE_DIR, "system_prompt.txt")
MEMORY_PATH = os.path.join(BASE_DIR, "memory.json")
MAX_MEMORY = 1000

# --- System prompt ---
with open(SYSTEM_PROMPT_PATH, "r") as f:
    system_prompt = f.read()
sys_msg = SystemMessage(content=system_prompt)

# --- Tools ve Tool Executor ---
tools = [multiply, add, subtract, divide, modulus, wiki_search, web_search, arvix_search, get_youtube_transcript, transcribe_audio]

class SimpleToolExecutor:
    def __init__(self, tools):
        self.tools = {tool.name: tool for tool in tools}
    def invoke(self, tool_calls):
        results = []
        for call in tool_calls:
            tool = self.tools.get(call['name'])
            if tool:
                try:
                    result = tool.run(call['args'])
                except Exception as e:
                    result = str(e)
                results.append(result)
            else:
                results.append(f"Tool {call['name']} not found.")
        return results

tool_executor = SimpleToolExecutor(tools)

# --- LLM ayarı (zephyr) ---
llm_with_tools = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    huggingfacehub_api_token=os.getenv("HF_TOKEN")
)

# --- Hafıza işlemleri ---
def load_memory():
    if os.path.exists(MEMORY_PATH):
        with open(MEMORY_PATH, "r") as f:
            return json.load(f)
    return []

def save_memory(entry):
    memory = load_memory()
    memory.append(entry)
    if len(memory) > MAX_MEMORY:
        memory = memory[-MAX_MEMORY:]
    with open(MEMORY_PATH, "w") as f:
        json.dump(memory, f, indent=2)

# --- State Tanımı ---
class MessagesStateWithFlag(TypedDict):
    messages: List[BaseMessage]
    retrieved_answer_found: Optional[bool]

# --- Retriever Node ---
def retriever(state: MessagesStateWithFlag) -> MessagesStateWithFlag:
    query = state["messages"][-1].content.strip().lower()
    memory = load_memory()
    for entry in memory:
        question = entry["question"].strip().lower()
        similarity = difflib.SequenceMatcher(None, question, query).ratio()
        if similarity > 0.9:
            return {
                "messages": [sys_msg, AIMessage(content=f"{entry['answer']}")],
                "retrieved_answer_found": True
            }
    return {"messages": state["messages"], "retrieved_answer_found": False}

# --- Assistant Node ---
def assistant(state: MessagesStateWithFlag) -> MessagesStateWithFlag:
    messages = [sys_msg] + state["messages"]
    prompt = "\n".join([m.content for m in messages if hasattr(m, "content")])
    response = llm_with_tools.invoke(prompt)
    if isinstance(response, dict):
        response_text = response.get("generated_text", str(response))
    else:
        response_text = str(response)
    return {"messages": state["messages"] + [AIMessage(content=response_text)], "retrieved_answer_found": False}
# --- Tool Executor Node ---
def tool_executor_node(state: MessagesStateWithFlag) -> MessagesStateWithFlag:
    last_msg = state["messages"][-1]
    if not isinstance(last_msg, AIMessage) or not last_msg.tool_calls:
        return state
    results = tool_executor.invoke(last_msg.tool_calls)
    tool_messages = [
        ToolMessage(content=json.dumps(result), tool_call_id=call['id'])
        for call, result in zip(last_msg.tool_calls, results)
    ]
    return {"messages": state["messages"] + tool_messages, "retrieved_answer_found": False}

# --- Save Answer Node ---
def save_answer_node(state: MessagesStateWithFlag) -> MessagesStateWithFlag:
    question = next((m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), "")
    answer = state["messages"][-1].content
    save_memory({"question": question, "answer": answer})
    return state

# --- Geçiş Kuralı ---
def should_continue(state: MessagesStateWithFlag) -> str:
    if state.get("retrieved_answer_found"):
        return "end"
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tool_executor"
    return "save_answer"

# --- Graph Tanımı ---
def build_graph():
    builder = StateGraph(MessagesStateWithFlag)
    builder.set_entry_point("retriever")
    builder.add_node("retriever", retriever)
    builder.add_node("assistant", assistant)
    builder.add_node("tool_executor", tool_executor_node)
    builder.add_node("save_answer", save_answer_node)
    builder.add_conditional_edges("retriever", lambda state: "end" if state["retrieved_answer_found"] else "assistant", {"assistant": "assistant", "end": END})
    builder.add_conditional_edges("assistant", should_continue, {"tool_executor": "tool_executor", "save_answer": "save_answer"})
    builder.add_edge("tool_executor", "assistant")
    builder.add_edge("save_answer", END)
    return builder.compile()

# --- CLI ---
if __name__ == "__main__":
    app = build_graph()
    print("🤖 Chatbot başladı. Çıkmak için 'exit' yaz.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Bot: Görüşmek üzere!")
            break
        result = app.invoke({"messages": [HumanMessage(content=user_input)]})
        print(f"Bot: {result['messages'][-1].content}")