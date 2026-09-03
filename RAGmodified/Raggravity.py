import os
import pickle
from pathlib import Path
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_classic.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain_community.document_compressors import FlashrankRerank
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

load_dotenv()

# -----------------------------
# 1. Models & Embeddings Setup
# -----------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

llm = ChatGroq(
    model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
    temperature=0.2,
    api_key=os.getenv("GROQ_API_KEY"),
)




# -----------------------------
# 2. Document Ingestion & Caching
# -----------------------------
def get_or_create_index(pdf_path: Path, base_index_dir: Path = Path("faiss_indexes")):
    """
    Checks if an index exists for the given PDF.
    If yes, loads FAISS and saved chunks.
    If no, parses the PDF, chunks it, creates FAISS index, and saves chunks.
    """
    doc_id = pdf_path.stem
    index_directory = base_index_dir / doc_id
    index_file = index_directory / "index.faiss"
    pkl_file = index_directory / "index.pkl"
    chunks_file = index_directory / "chunks.pkl"

    if index_file.exists() and pkl_file.exists() and chunks_file.exists():
        print(f"\n[Cache] Loading existing index from: {index_directory}")
        vectorstore = FAISS.load_local(
            folder_path=str(index_directory),
            embeddings=embeddings,
            allow_dangerous_deserialization=True,
        )
        with open(chunks_file, "rb") as f:
            chunks = pickle.load(f)
    else:
        print(f"\n[Ingest] Parsing PDF: {pdf_path.name}")
        loader = PyPDFLoader(str(pdf_path))
        documents = loader.load()
        print(f"[Ingest] Loaded {len(documents)} pages.")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=700,
            chunk_overlap=150,
        )
        chunks = text_splitter.split_documents(documents)
        print(f"[Ingest] Split into {len(chunks)} text chunks.")

        print("[Ingest] Generating embeddings and building FAISS vector store...")
        vectorstore = FAISS.from_documents(chunks, embeddings)

        # Persist index and chunks to disk
        index_directory.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(str(index_directory))
        with open(chunks_file, "wb") as f:
            pickle.dump(chunks, f)
        print(f"[Ingest] Successfully cached index to: {index_directory}")

    return vectorstore, chunks




# -----------------------------
# 3. Hybrid Retriever Builder
# -----------------------------
def create_hybrid_retriever(vectorstore: FAISS, chunks: list):
    """Combines BM25 (keyword) + FAISS (dense) + FlashRank (reranker)."""
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
    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=ensemble_retriever,
    )

# -----------------------------
# 4. LangGraph App Factory
# -----------------------------
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def build_rag_app(pdf_path: Path):
    """Builds a compiled LangGraph workflow tailored for the given PDF."""
    vectorstore, chunks = get_or_create_index(pdf_path)
    retriever = create_hybrid_retriever(vectorstore, chunks)
    doc_title = pdf_path.stem.replace("-", " ").replace("_", " ")

    @tool
    def document_search_tool(query: str) -> str:
        """Search the active document for relevant passages using hybrid search."""
        documents = retriever.invoke(query)
        if not documents:
            return "No relevant information found in the document."

        results = [
            f"[Passage {idx}]:\n{doc.page_content}"
            for idx, doc in enumerate(documents, start=1)
        ]
        return "\n\n".join(results)

    tools = [document_search_tool]
    bound_llm = llm.bind_tools(tools)

    def agent_node(state: State) -> dict[str, list[BaseMessage]]:
        system_message = SystemMessage(
            content=(
                f"You are an expert research assistant with access to the document: '{doc_title}'. "
                "Use the document_search_tool whenever you need specific facts, definitions, or details from the text. "
                "Answer the user accurately based on the retrieved context. "
                "If the information is not contained in the document, state that clearly."
            )
        )
        messages = state["messages"]
        # Prepend system message if not already present
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [system_message, *messages]

        response = bound_llm.invoke(messages)
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

    checkpointer = MemorySaver()
    app = graph.compile(checkpointer=checkpointer)
    return app, doc_title



# -----------------------------
# 5. Interactive CLI
# -----------------------------
def main():
    print("=" * 60)
    print("  General-Purpose Hybrid Agentic RAG (FlashRank + BM25 + FAISS)")
    print("=" * 60)

    while True:
        raw_input_path = input("\nEnter the path to your PDF file (or 'exit' to quit): ").strip()
        if raw_input_path.lower() == "exit":
            return

        # Strip surrounding quotes (common when dragging files into Windows terminal)
        cleaned_path = raw_input_path.strip('\'"')
        pdf_path = Path(cleaned_path).expanduser().resolve()

        if not pdf_path.exists():
            print(f"❌ File not found at: {pdf_path}. Please try again.")
            continue
        if pdf_path.suffix.lower() != ".pdf":
            print(f"❌ '{pdf_path.name}' is not a PDF file. Please provide a .pdf file.")
            continue

        break

    # Build the application for this specific PDF
    app, doc_title = build_rag_app(pdf_path)

    print(f"\n✅ Ready! Ask questions about '{doc_title}'.")
    print("Type 'exit' to quit or 'switch' to load another PDF.\n")

    config = {"configurable": {"thread_id": "cli_session"}}

    while True:
        user_query = input("\nYou: ").strip()
        if not user_query:
            continue
        if user_query.lower() == "exit":
            print("Goodbye!")
            break
        if user_query.lower() == "switch":
            # Recurse to switch document
            main()
            return

        result = app.invoke(
            {"messages": [HumanMessage(content=user_query)]},
            config=config,
        )
        final_answer = result["messages"][-1].content
        print(f"\nAssistant:\n{final_answer}")

if __name__ == "__main__":
    main()
    