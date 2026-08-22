import os
from pathlib import Path
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode


from langchain_classic.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain_community.document_compressors import FlashrankRerank
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


# Step 3 Import: LangGraph Checkpointer
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

# -----------------------------
# 1. Embedding Model
# -----------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)



# -----------------------------
# 2. Vector Store Caching (Step 1)
# -----------------------------
pdf_path = Path(__file__).resolve().parent.parent / "DBMS-Unit-2.pdf"
index_directory = (
    Path(__file__).resolve().parent.parent
    / "faiss_indexes"
    / pdf_path.stem
)

index_file = index_directory / "index.faiss"
pkl_file = index_directory / "index.pkl"

if index_file.exists() and pkl_file.exists():
    print(f"Loading existing FAISS index from: {index_directory}")
    vectorstore = FAISS.load_local(
        folder_path=str(index_directory),
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )
    chunks = list(vectorstore.docstore._dict.values())
else:
    print(f"Index not found. Ingesting PDF: {pdf_path}")
    documents = PyPDFLoader(str(pdf_path)).load()
    print(f"PDF loaded: {len(documents)} pages.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=150,
    )
    chunks = text_splitter.split_documents(documents)

    print("Generating embeddings and building FAISS vector store...")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    index_directory.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(index_directory))
    print(f"Created and saved FAISS vector store at: {index_directory}")


# -----------------------------
# 3. Hybrid Search & Reranker (Step 2)
# -----------------------------
bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 5

dense_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5},
)

ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, dense_retriever],
    weights=[0.4, 0.6],
)

compressor = FlashrankRerank(top_n=3)
hybrid_rerank_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=ensemble_retriever,
)

# -----------------------------
# 4. Tool & LLM Setup
# -----------------------------
@tool
def retriever_tool(query: str) -> str:
    """Search the PDF with hybrid retrieval and return reranked sections."""
    documents = hybrid_rerank_retriever.invoke(query)

    if not documents:
        return "No relevant documents were found."

    print("\n--- Reranked chunks ---")
    for index, document in enumerate(documents, start=1):
        score = document.metadata.get("relevance_score", "N/A")
        print(f"\nChunk {index} (Score: {score}):")
        print(document.page_content)

    results = [
        f"Document {index}: {document.page_content}"
        for index, document in enumerate(documents, start=1)
    ]
    return "\n\n".join(results)

tools = [retriever_tool]

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.3,
    api_key=os.getenv("GROQ_API_KEY")
)
llm_with_tools = llm.bind_tools(tools)




# -----------------------------
# 5. StateGraph & Checkpointer (Step 3)
# -----------------------------
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

SYSTEM_PROMPT = SystemMessage(
    content=(
        "You are an expert tutor in Database Management Systems. "
        "Use the PDF via retriever_tool to answer the question. "
        "Call retriever_tool once for each user question. "
        "After receiving the tool result, answer directly and do not call the tool again."
    )
)

def agent_node(state: State) -> dict[str, list[BaseMessage]]:
    messages = state["messages"]
    # Ensure system prompt is applied once without bloating state
    if not isinstance(messages[0], SystemMessage):
        messages = [SYSTEM_PROMPT, *messages]
    
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

# Compile graph with MemorySaver checkpointer
checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)




# -----------------------------
# 6. Execution with Thread Handling
# -----------------------------
if __name__ == "__main__":
    # Every conversation session has a unique thread_id
    config = {"configurable": {"thread_id": "dbms_study_session_1"}}

    print("DBMS Agent Ready. Type 'exit' to quit.")
    while True:
        user_input = input("\nEnter your question: ").strip()
        if not user_input or user_input.lower() == "exit":
            print("Session ended.")
            break

        # Pass only the NEW message; the checkpointer loads previous history automatically
        result = app.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=config
        )

        final_response = result["messages"][-1].content
        print(f"\nAnswer:\n{final_response}\n")