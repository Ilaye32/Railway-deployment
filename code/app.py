import streamlit as st
import os
import glob
from typing import List
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

# Fix PyTorch meta tensor warning
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*meta tensor.*")

# Set environment variable to fix ChromaDB torch issue
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import LangChain components
try:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_openai import ChatOpenAI
    from langchain_groq import ChatGroq
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError as e:
    st.error(f"Import Error: {e}")
    st.error("Please install compatible versions. Run: pip install -r requirements.txt")
    st.stop()

# Import local modules
try:
    from vectordb import VectorDB
    from paths import PUBLICATION_DIR
except ImportError as e:
    st.error(f"Missing local module: {e}")
    st.error("Make sure vectordb.py and paths.py are in the same directory")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Ilaye - AI Engineer Portfolio and assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #6B7280;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        color: #1F2937;
    }
    .user-message {
        background-color: #DBEAFE;
        border-left: 4px solid #2563EB;
    }
    .assistant-message {
        background-color: #D1FAE5;
        border-left: 4px solid #059669;
    }
    .chat-message strong {
        color: #111827;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)


def load_documents() -> List[dict]:
    """Load documents from the publication directory."""
    results = []
    
    if not os.path.exists(PUBLICATION_DIR):
        return None
    
    # Find all markdown and text files
    markdown_files = glob.glob(os.path.join(PUBLICATION_DIR, "**", "*.md"), recursive=True)
    txt_files = glob.glob(os.path.join(PUBLICATION_DIR, "**", "*.txt"), recursive=True)
    all_files = markdown_files + txt_files
    
    if not all_files:
        return None
    
    # Read each file
    for file_path in all_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if content.strip():
                    results.append({
                        'content': content,
                        'metadata': {
                            'source': file_path,
                            'filename': os.path.basename(file_path)
                        }
                    })
        except Exception as e:
            st.warning(f"Could not read {file_path}: {e}")
    
    return results if results else None


class RAGAssistant:
    """RAG-based AI assistant."""

    def __init__(self):
        """Initialize the RAG assistant."""
        self.llm = self._initialize_llm()
        if not self.llm:
            raise ValueError("No valid API key found in .env file")

        self.vector_db = VectorDB()

        template = """Your name is Ilaye and you are an AI engineer freelancer that wants to sell yourself to get a contract. Use the following context about yourself to answer the question.

Rules:
1. Only answer questions based on the documents below
2. Answer clearly and politely
3. Be persuasive - you want to sell yourself
4. Greet users warmly
5. If asked about unrelated topics, say "I'm all about Ilaye"
6. Keep responses under 80 words
7. Never hallucinate
8. You are interacting with strangers and not ilaye nor Timibofa

Context:
{context}

Question: {question}

Answer:"""
        
        self.prompt_template = ChatPromptTemplate.from_template(template)
        self.chain = self.prompt_template | self.llm | StrOutputParser()

    def _initialize_llm(self):
        """Initialize the LLM with available API key."""
        try:
            if os.getenv("OPENAI_API_KEY"):
                model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
                return ChatOpenAI(
                    api_key=os.getenv("OPENAI_API_KEY"),
                    model=model_name,
                    temperature=0.0
                )
            elif os.getenv("GROQ_API_KEY"):
                model_name = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
                return ChatGroq(
                    api_key=os.getenv("GROQ_API_KEY"),
                    model=model_name,
                    temperature=0.0
                )
            elif os.getenv("GOOGLE_API_KEY"):
                model_name = os.getenv("GOOGLE_MODEL", "gemini-2.0-flash")
                return ChatGoogleGenerativeAI(
                    google_api_key=os.getenv("GOOGLE_API_KEY"),
                    model=model_name,
                    temperature=0.0
                )
        except Exception as e:
            st.error(f"Error initializing LLM: {e}")
        return None

    def add_documents(self, documents: List) -> None:
        """Add documents to the knowledge base."""
        if documents:
            self.vector_db.add_documents(documents)

    def query(self, question: str, n_results: int = 3) -> str:
        """Query the RAG assistant."""
        try:
            search_results = self.vector_db.search(question, n_results=n_results)
            retrieved_docs = search_results.get('documents', [])
            
            if not retrieved_docs:
                return "I couldn't find relevant information to answer your question."
            
            context = "\n\n---\n\n".join(retrieved_docs)
            answer = self.chain.invoke({"context": context, "question": question})
            return answer
        except Exception as e:
            return f"Error: {e}"


# Session state initialization
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'assistant' not in st.session_state:
    st.session_state.assistant = None
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
if 'doc_count' not in st.session_state:
    st.session_state.doc_count = 0
if 'auto_init_done' not in st.session_state:
    st.session_state.auto_init_done = False


def initialize_system():
    """Initialize the RAG system."""
    try:
        with st.spinner("🔄 Initializing RAG System..."):
            # Load documents
            docs = load_documents()
            if not docs:
                st.error("❌ No documents found in publications directory")
                return False
            
            # Initialize assistant
            assistant = RAGAssistant()
            assistant.add_documents(docs)
            
            # Save to session
            st.session_state.assistant = assistant
            st.session_state.initialized = True
            st.session_state.doc_count = len(docs)
            return True
    except Exception as e:
        st.error(f"❌ Initialization failed: {e}")
        return False


def main():
    # Auto-initialize on first load
    if not st.session_state.auto_init_done and not st.session_state.initialized:
        with st.spinner("🚀 Initializing AI Assistant... Please wait a few seconds..."):
            if initialize_system():
                st.session_state.auto_init_done = True
                st.success("✅ System Ready! You can start chatting now.")
                st.rerun()
            else:
                st.session_state.auto_init_done = True
                st.error("⚠️ Auto-initialization failed. Please check your setup.")
    
    # Header
    st.markdown('<h1 class="main-header">🤖 Ilaye - AI Engineer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Professional Portfolio & AI-Powered Q&A Assistant</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://api.dicebear.com/7.x/avataaars/svg?seed=Ilaye", width=150)
        
        st.markdown("### 👨‍💻 About Me")
        st.markdown("**AI Engineer & ML Specialist**")
        st.markdown("Building intelligent systems with cutting-edge AI.")
        
        st.markdown("---")
        
        st.markdown("### 🎯 Expertise")
        st.markdown("""
        - 🧠 Machine Learning
        - 💬 NLP & LLMs
        - 🔍 RAG Systems
        - 🚀 AI Applications
        - ☁️ MLOps
        """)
        
        st.markdown("---")
        
        st.markdown("### 📊 Stats")
        col1, col2 = st.columns(2)
        col1.metric("Projects", "15+")
        col1.metric("Tech Stack", "20+")
        col2.metric("Experience", "3+ yrs")
        col2.metric("Clients", "10+")
        
        st.markdown("---")
        
        # System Status
        st.markdown("### ⚙️ System Status")
        
        api_key_present = bool(
            os.getenv("OPENAI_API_KEY") or 
            os.getenv("GROQ_API_KEY") or 
            os.getenv("GOOGLE_API_KEY")
        )
        
        if api_key_present:
            st.success("🔑 API Key: ✓")
        else:
            st.error("🔑 API Key: ✗")
            st.info("Add API key to .env file")
        
        # Initialize button (manual re-initialization if needed)
        if st.button("🔄 Re-Initialize System", use_container_width=True):
            st.session_state.initialized = False
            st.session_state.auto_init_done = False
            if initialize_system():
                st.success(f"✅ Ready! {st.session_state.doc_count} docs loaded")
                st.rerun()
        
        # Status indicator
        if st.session_state.initialized:
            st.success("🟢 Online")
            st.info(f"📚 Documents: {st.session_state.doc_count}")
        else:
            st.warning("🟡 Not Initialized")
        
        st.markdown("---")
        
        # Clear chat
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    # Main content
    tab1, tab2 = st.tabs(["💬 Chat", "📈 Skills"])
    
    with tab1:
        st.markdown("### Ask Me Anything")
        st.markdown("Ask about my experience, skills, projects, or background.")
        
        # Sample questions
        st.markdown("**💡 Try these questions:**")
        samples = [
            "What is your experience with RAG systems?",
            "What projects have you worked on?",
            "What technologies do you know?",
            "Tell me about your background",
            "How do you approach ML problems?"
        ]
        
        cols = st.columns(2)
        for idx, question in enumerate(samples):
            with cols[idx % 2]:
                if st.button(question, key=f"q{idx}", use_container_width=True):
                    if not st.session_state.initialized:
                        st.warning("⚠️ Initialize system first")
                    else:
                        st.session_state.messages.append({"role": "user", "content": question})
                        with st.spinner("Thinking..."):
                            response = st.session_state.assistant.query(question)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        st.rerun()
        
        st.markdown("---")
        
        # Chat history
        if st.session_state.messages:
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    st.markdown(f'''
                        <div class="chat-message user-message">
                            <strong>👤 You:</strong><br>
                            <span style="color: #1F2937;">{msg["content"]}</span>
                        </div>
                    ''', unsafe_allow_html=True)
                else:
                    st.markdown(f'''
                        <div class="chat-message assistant-message">
                            <strong>🤖 Ilaye:</strong><br>
                            <span style="color: #1F2937;">{msg["content"]}</span>
                        </div>
                    ''', unsafe_allow_html=True)
        else:
            st.info("👋 Initialize the system and start chatting!")
        
        # Input
        st.markdown("---")
        col1, col2 = st.columns([5, 1])
        user_input = col1.text_input("Your question:", placeholder="Ask me anything...", label_visibility="collapsed")
        send = col2.button("Send 📤", use_container_width=True)
        
        if send and user_input:
            if not st.session_state.initialized:
                st.warning("⚠️ Please initialize the system first")
            else:
                st.session_state.messages.append({"role": "user", "content": user_input})
                with st.spinner("Generating response..."):
                    response = st.session_state.assistant.query(user_input)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()
    
    with tab2:
        st.markdown("### Skills Overview")
        skills = {
            "Python": 95,
            "TensorFlow/PyTorch": 90,
            "LangChain": 85,
            "Vector DBs": 88,
            "Cloud": 80
        }
        for skill, level in skills.items():
            st.write(f"**{skill}**")
            st.progress(level / 100)
            st.write(f"{level}%")
            st.write("")
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #6B7280; padding: 1rem;'>
            <p><strong>Built with Streamlit & RAG Technology</strong></p>
            <p>© 2024 Ilaye - AI Engineer</p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()