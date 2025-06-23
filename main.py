import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from langchain.vectorstores import Qdrant as QdrantVectorStore
from openai import OpenAI
from qdrant_client.http.models import VectorParams, Distance
from dotenv import load_dotenv
import time
from langchain.document_loaders import PyPDFLoader
from langchain.schema import Document
import tempfile
from streamlit_option_menu import option_menu

load_dotenv()

# Load secrets
QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]
QDRANT_URL = st.secrets["QDRANT_URL"]

# Initialize clients globally
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
openai_client = OpenAI()

COLLECTION_NAME = "MyApp-Streamlit"
EMBEDDING_MODEL = "text-embedding-3-small"

# --- Helper functions ---
def reset_state():
    st.session_state.docs = []
    st.session_state.chat_history = []
    st.session_state.embeddings_pushed = False

def collection_exists_and_has_data(client, collection_name):
    try:
        collections = client.get_collections().collections
        if any(c.name == collection_name for c in collections):
            stats = client.get_collection(collection_name=collection_name)
            if stats.vectors_count > 0:
                return True
        return False
    except Exception as e:
        return False

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

def process_pdf(file) -> list[Document]:
    # Save the uploaded file to a temporary path for PyPDFLoader
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file.read())
        pdf_path = tmp_file.name

    # Load and split the PDF using LangChain tools
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    print(f"📄 Number of pages loaded: {len(documents)}")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(documents)

    return split_docs

def push_to_qdrant(docs):
    # Recreate the Qdrant collection
    qdrant_client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    )

    vectorstore = QdrantVectorStore(
        client=qdrant_client,
        collection_name=COLLECTION_NAME,
        embeddings=embeddings
    )

    batch_size = 5
    total = len(docs)

    progress_bar = st.progress(0)
    status_text = st.empty()
    warning_text = st.empty()

    for i in range(0, total, batch_size):
        batch_docs = docs[i:i + batch_size]
        success = False

        for attempt in range(5):
            try:
                vectorstore.add_documents(batch_docs)
                success = True
                warning_text.empty()
                break
            except Exception as e:
                wait = 2 ** attempt
                warning_text.warning(
                    f"Retry {attempt + 1}/5 failed for batch {i}-{i + len(batch_docs)}. "
                    f"Waiting {wait}s...\n\nError: {e}"
                )
                time.sleep(wait)

        if not success:
            status_text.error(f"❌ Failed to upload batch {i}-{i + len(batch_docs)} after 5 attempts. Aborting.")
            progress_bar.empty()
            return

        progress = min((i + len(batch_docs)) / total, 1.0)
        progress_bar.progress(progress)
        status_text.info(f"✅ Indexed {i + len(batch_docs)} of {total} chunks...")

    warning_text.empty()
    status_text.success("🎉 Indexing complete!")
    progress_bar.empty()


def similarity_search(query, k=5):
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = QdrantVectorStore(client=qdrant_client, collection_name=COLLECTION_NAME, embeddings=embeddings)
    return vectorstore.similarity_search(query, k=k)

def build_system_prompt(results):
    context = "\n\n\n".join(
        f"Page Content: {doc.page_content}\nPage Number: {doc.metadata.get('page', 'unknown')}"
        for doc in results
    )
    prompt = f"""
You are a helpful and expert assistant that answers technical questions strictly based on the provided context, which has been extracted from a PDF document.
This document may be a programming reference (like C++ or Node.js), a technical book (like *Designing Data-Intensive Applications*), or API documentation.

Context:
{context}

Instructions:
- Use only the provided context to answer the user's question.
- Clearly explain the concept, code behavior, or architecture in practical terms.
- If relevant, include accurate code snippets or technical examples from the context to illustrate your explanation.
- Use analogies or visual metaphors **only if they are hinted at or supported** in the context.
- If the question is not fully answerable using the context, say so clearly and avoid making assumptions.
- End your response with a reference to the page number(s) from the context for further reading.

Now answer the following question using only the above context.
"""
    return prompt

def openai_chat_completion(system_prompt, query):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query}
    ]
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )
    return response.choices[0].message.content

# --- Custom CSS for chat ---
st.markdown(
    """
    <style>
    .chat-container {
        max-height: 500px;
        overflow-y: auto;
        padding: 10px;
        background-color: #1e1e1e;
        border-radius: 10px;
        border: 1px solid #444;
        margin-top: 20px;
    }
    .user-msg {
        background-color: #2e7d32;
        color: #ffffff;
        padding: 12px 18px;
        border-radius: 20px 20px 0 20px;
        max-width: 70%;
        margin-left: auto;
        margin-bottom: 10px;
        font-family: 'Segoe UI';
    }
    .ai-msg {
        background-color: #333333;
        color: #ffffff;
        padding: 12px 18px;
        border-radius: 20px 20px 20px 0;
        max-width: 70%;
        margin-right: auto;
        margin-bottom: 10px;
        font-family: 'Segoe UI';
    }
    .user-name, .ai-name {
        font-size: 12px;
        margin-bottom: 2px;
        color: #bbbbbb;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Streamlit UI ---
st.title("📚 AskMyDoc")
selected_mode = option_menu(
    menu_title=None,
    options=["upload", "chat"],
    icons=["cloud-upload", "chat-left-text"],
    default_index=0,
    orientation="horizontal"
)

st.session_state.mode = selected_mode
mode = selected_mode


# Init state variables
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "embeddings_ready" not in st.session_state:
    st.session_state.embeddings_ready = False

if "pdf_uploaded" not in st.session_state:
    st.session_state.pdf_uploaded = False

# === Mode: Upload New PDF ===
if mode == "upload":
    uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"], key="pdf_uploader")

    if uploaded_file and not st.session_state.pdf_uploaded:
        with st.spinner("Processing PDF and pushing embeddings..."):
            docs = process_pdf(uploaded_file)
            push_to_qdrant(docs)
            st.session_state.chat_history = []
            st.session_state.embeddings_ready = True
            st.session_state.pdf_uploaded = True  # 🟢 Lock upload to avoid re-processing
        st.success("PDF processed and indexed successfully!")

# === Mode: Continue Chat ===
elif mode == "chat":
    st.session_state.embeddings_ready = True  # 🟢 Assume embeddings already exist

# === Show Chat UI if embeddings are ready ===
if st.session_state.embeddings_ready:

    def submit_query():
        query = st.session_state.user_input.strip()
        if not query:
            return
        st.session_state.user_input = ""

        st.session_state.chat_history.append({"role": "user", "content": query})
        results = similarity_search(query, k=5)
        system_prompt = build_system_prompt(results)
        answer = openai_chat_completion(system_prompt, query)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})

    st.text_input("Ask a question about your document:", key="user_input", on_change=submit_query)

    # --- Render chat
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for chat in st.session_state.chat_history:
        if chat["role"] == "user":
            st.markdown('<div class="user-name">User</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="user-msg">{chat["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="ai-name">AI</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="ai-msg">{chat["content"]}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
