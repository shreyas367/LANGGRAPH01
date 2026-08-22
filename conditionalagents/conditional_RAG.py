import os
from pathlib import Path
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage,SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()


# -----------------------------
# Embedding Model
# -----------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# -----------------------------
# Build Retriever
# -----------------------------


pdf_path = Path(__file__).resolve().parent.parent / "DBMS-Unit-2.pdf"


documents = PyPDFLoader(str(pdf_path)).load()

print(f"PDF has been loaded and has {len(documents)} pages ")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=150,
)

chunks = text_splitter.split_documents(documents)

vectorstore = FAISS.from_documents(chunks, embeddings)

index_directory = (
    Path(__file__).resolve().parent.parent
    / "faiss_indexes"
    / pdf_path.stem
)

index_directory.mkdir(parents=True, exist_ok=True)
vectorstore.save_local(str(index_directory))
print(f"Created FAISS vector store at: {index_directory}")



# -----------------------------
# Load PDFs
# -----------------------------

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
)





# -----------------------------
# LLM
# -----------------------------

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.3,
    api_key=os.getenv("GROQ_API_KEY")
)


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


@tool
def retriever_tool(query: str) -> str:
    """Search the PDF and return the most relevant document sections."""
    documents = retriever.invoke(query)

    if not documents:
        return "No relevant documents were found."

    print("\nRetrieved chunks:")
    for index, document in enumerate(documents, start=1):
        print(f"\n--- Chunk {index} ---")
        print(document.page_content)

    results = [
        f"Document {index}: {document.page_content}"
        for index, document in enumerate(documents, start=1)
    ]
    return "\n\n".join(results)


tools = [retriever_tool]
llm_with_tools = llm.bind_tools(tools)



def agent_node(state: State) -> dict[str, list[BaseMessage]]:
    messages = [
SystemMessage(
    content=(
        "Use the PDF to answer the question. "
        "Call retriever_tool once for each user question. "
        "After receiving the tool result, answer directly and do not call the tool again."
    )
),
        *state["messages"],
    ]

    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}



def should_continue(state: State) -> str:
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return END




graph = StateGraph(State)
graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(tools))

graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")

app = graph.compile()




conversation: list[BaseMessage] = []
user_input = input("Enter your question (or 'exit' to quit): ").strip()
while user_input.lower() != "exit":
    conversation.append(HumanMessage(content=user_input))

    result = app.invoke({"messages": conversation})
    conversation = result["messages"]

    print(f"\nAnswer:\n{conversation[-1].content}\n")
    user_input = input("Enter your question (or 'exit' to quit): ").strip()



# START
#   |
#   v
# Agent Node
#   |----------------------> END
#   |                         Final answer
#   v
# Retriever Tool
#   |
#   | PDF chunks
#   v
# Agent Node
