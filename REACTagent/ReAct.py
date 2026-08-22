import os
from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_mistralai import ChatMistralAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

load_dotenv()

# Initialize LLM
llm = ChatMistralAI(
    model="mistral-small-latest",
    temperature=0.7,
    api_key=os.getenv("MISTRAL_API_KEY")
)

# Define Tools & Bind to LLM
@tool
def add(a: int, b: int) -> int:
    """Adds two numbers together."""
    return a + b

@tool
def multiply(a:any,b:any) ->any:
    """multiply the two numbers"""
    return a*b




tools = [add,multiply]
llm_with_tools = llm.bind_tools(tools)





# State Definition
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]




# Node: Call Model
def model_call(state: AgentState) -> dict:
    system_prompt = SystemMessage(content="You are a helpful AI assistant.")
    # Invoke bound model with history
    response = llm_with_tools.invoke([system_prompt] + list(state["messages"]))
    return {"messages": [response]}


# Conditional Edge Logic
def should_continue(state: AgentState) -> str:
    messages = state["messages"]
    last_message = messages[-1]
    
    # If the model produced tool calls, route to tools; otherwise end
    if getattr(last_message, "tool_calls", None):
        return "continue"
    return "end"




# Build Graph
graph = StateGraph(AgentState)

graph.add_node("ouragent", model_call)
graph.add_node("tools", ToolNode(tools=tools))

graph.add_edge(START, "ouragent")
graph.add_conditional_edges(
    "ouragent",
    should_continue,
    {
        "continue": "tools",
        "end": END,
    },
)
graph.add_edge("tools", "ouragent")

app = graph.compile()



# Stream Processing
def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]  
        message.pretty_print()




# Pass messages using HumanMessage 
input_data = {"messages": [HumanMessage(content="add 55+77  also add 99+77 then multiply for final result  ,, tell two good jokes")]}


print_stream(app.stream(input_data, stream_mode="values"))



#        [ START ]
#            │
#            ▼
#      ┌───────────┐
#      │ ouragent  │ ◄──────────┐
#      └─────┬─────┘            │
#            │                  │
#     [should_continue]         │
#      /           \            │
#  (tools called)  (no tools)   │
#    /               \          │
#   ▼                 ▼         │
# ┌───────┐        [ END ]      │
# │ tools │                     │
# └───┬───┘                     │
#     └─────────────────────────┘