import os
import sys

# ==========================================
# CONFIGURATION: Choose your provider here!
# Options: "OLLAMA" or "GEMINI"
# ==========================================
PROVIDER = "OLLAMA"

# Model selection configurations
OLLAMA_CHAT_MODEL = "batiai/gemma4-e4b:q4"       # Local model to use for chat
OLLAMA_EMBED_MODEL = "batiai/gemma4-e4b:q4"     # Local model to use for embeddings (or e.g. "nomic-embed-text")

GEMINI_CHAT_MODEL = "gemini-1.5-flash"
GEMINI_EMBED_MODEL = "models/text-embedding-004"

# ==========================================
# 1. Dependency Checks & Setup
# ==========================================
print(f"--- Initializing LangChain RAG Demo using {PROVIDER} ---")

# Verify API Keys if Gemini is selected
if PROVIDER == "GEMINI":
    if not os.environ.get("GOOGLE_API_KEY"):
        print("[ERROR] GOOGLE_API_KEY environment variable is not set.")
        print("Please set it in your terminal before running this script, e.g.:")
        print("  Windows CMD:  set GOOGLE_API_KEY=your_key")
        print("  PowerShell:   $env:GOOGLE_API_KEY='your_key'")
        sys.exit(1)

# Ensure sample data exists
sample_filename = "sample_knowledge.txt"
if not os.path.exists(sample_filename):
    print(f"Creating sample knowledge file: {sample_filename}...")
    sample_text = """
LangChain is a framework for developing applications powered by large language models (LLMs).
It simplifies the process of building LLM-powered applications by providing modular abstractions.
Key components of LangChain include:
1. Model I/O: Formatting prompts, calling models, and parsing output.
2. Retrieval: Loading, transforming, and indexing documents, and retrieving relevant context.
3. Chains: Combining components to solve specific tasks (e.g., Q&A, summarization).
4. Agents: Allowing LLMs to decide which tools to call dynamically based on input.

Ollama is a tool that allows you to run large language models locally on your machine.
It packages model weights, configuration, and datasets into a single Modelfile.
With Ollama, developers can run models like Llama, Gemma, and Mistral locally, ensuring privacy and offline functionality.
It is easy to integrate Ollama with LangChain using the langchain-ollama integration package.

Retrieval-Augmented Generation (RAG) is a technique that enhances LLM responses by fetching relevant context from a external knowledge base before generating an answer.
This helps prevent model hallucinations and ensures answers are grounded in up-to-date, factual information.
"""
    with open(sample_filename, "w", encoding="utf-8") as f:
        f.write(sample_text.strip())

# ==========================================
# 2. Step-by-Step RAG Pipeline
# ==========================================

# Step 2a: Load the document
print("Loading document...")
from langchain_community.document_loaders import TextLoader
loader = TextLoader(sample_filename, encoding="utf-8")
docs = loader.load()

# Step 2b: Split the document into chunks
print("Splitting document into chunks...")
from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=80)
all_splits = text_splitter.split_documents(docs)
print(f"Generated {len(all_splits)} chunks from source document.")

# Step 2c: Setup Embeddings and Vector Store
print("Initializing Embeddings and Vector Store...")
from langchain_community.vectorstores import FAISS

if PROVIDER == "OLLAMA":
    from langchain_huggingface import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
elif PROVIDER == "GEMINI":
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    embeddings = GoogleGenerativeAIEmbeddings(model=GEMINI_EMBED_MODEL)
else:
    raise ValueError(f"Unknown provider: {PROVIDER}")

# Index the chunks in FAISS (in-memory vector database)
vector_store = FAISS.from_documents(documents=all_splits, embedding=embeddings)
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 2})
print("Vector Store index completed.")

# Step 2d: Setup the LLM Chat Model
print(f"Initializing Chat Model ({PROVIDER})...")
if PROVIDER == "OLLAMA":
    from langchain_ollama import ChatOllama
    llm = ChatOllama(model=OLLAMA_CHAT_MODEL, temperature=0)
elif PROVIDER == "GEMINI":
    from langchain_google_genai import ChatGoogleGenerativeAI
    llm = ChatGoogleGenerativeAI(model=GEMINI_CHAT_MODEL, temperature=0)

# Step 2e: Define the Prompt Template
from langchain_core.prompts import ChatPromptTemplate

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, say that you don't know. "
    "Use three sentences maximum and keep the answer concise.\n\n"
    "Context:\n{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Step 2f: Create the RAG Chain
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)
print("RAG Chain compiled successfully.")

# ==========================================
# 3. Interactive Testing Interface
# ==========================================
print("\n--- RAG Bot Ready ---")
print("Ask a question about LangChain, Ollama, or RAG (type 'exit' or 'quit' to stop):")

while True:
    try:
        user_query = input("\nUser: ")
        if user_query.lower() in ["exit", "quit", "q"]:
            print("Goodbye!")
            break
        if not user_query.strip():
            continue

        print("Thinking...")
        response = rag_chain.invoke({"input": user_query})
        
        print(f"\nBot ({PROVIDER}): {response['answer']}")
    except KeyboardInterrupt:
        print("\nGoodbye!")
        break
    except Exception as e:
        print(f"\n[ERROR] An error occurred: {e}")
