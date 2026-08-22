import os
from pathlib import Path
from typing import Annotated, List, Union
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

load_dotenv()


# State definition with built-in message reducer
class AgentState(TypedDict):
    messages: Annotated[List[Union[HumanMessage, AIMessage]], add_messages]



# Initialize Groq LLM
llm_groq = ChatGroq(
    model="llama-3.3-70b-versatile",  
    temperature=0.3,
    api_key=os.getenv("GROQ_API_KEY"),
)



# Processing Node
def process(state: AgentState) -> dict:
    """Generates the AI response and returns it to be merged into state."""
    response = llm_groq.invoke(state["messages"])
    return {"messages": [response]}



# Build Graph
graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)

agent = graph.compile()

# Chat Loop
conversation_history: List[BaseMessage] = []


user_input = input("Enter your question (or 'exit' to quit): ").strip()
while user_input.lower() != "exit":
    if user_input:
        result = agent.invoke({"messages": [HumanMessage(content=user_input)]})
        
        # Extract the latest response
        latest_ai_message = result["messages"][-1]
        print(f"\nAI: {latest_ai_message.content}\n")
        
        # Maintain complete history
        conversation_history = result["messages"]

    user_input = input("Enter your question: ").strip()




# Save conversation to file
log_path = Path(__file__).with_name("logging.txt")
with log_path.open("w", encoding="utf-8") as file:
    file.write("Your conversation log:\n\n")
    for message in conversation_history:
        if isinstance(message, HumanMessage):
            file.write(f"You: {message.content}\n\n")
        elif isinstance(message, AIMessage):
            file.write(f"AI: {message.content}\n\n")
    file.write("End of conversation\n")

print(f"Conversation saved to {log_path.resolve()}")