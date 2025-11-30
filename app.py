# app.py - DEFINITIVE FINAL STABLE CODE (Gyaan)

import os
import streamlit as st
from dotenv import load_dotenv
import pandas as pd

# Core LangChain Imports 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, CSVLoader

# --- CONFIGURATION (Stable) ---
load_dotenv()
VECTOR_DB_PATH = "faiss_index"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TEMP_DIR = "temp_uploads"
TEMP_PROCESSING_DIR = "temp_processing"

# --- AGENT CORE FUNCTIONS (Stable) ---

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def load_document_for_upload(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == ".pdf":
        return PyPDFLoader(file_path)
    elif ext == ".docx":
        return UnstructuredWordDocumentLoader(file_path)
    elif ext == ".csv":
        return CSVLoader(file_path)
    elif ext == ".xlsx":
        os.makedirs(TEMP_PROCESSING_DIR, exist_ok=True)
        csv_filepath = os.path.join(TEMP_PROCESSING_DIR, os.path.basename(file_path).replace(".xlsx", ".csv"))
        pd.read_excel(file_path, engine='openpyxl').to_csv(csv_filepath, index=False)
        return CSVLoader(csv_filepath)
    else:
        st.error(f"Unsupported file type for upload: {ext}")
        return None

def add_to_vector_store(uploaded_file, user_role, embeddings):
    if uploaded_file:
        os.makedirs(TEMP_DIR, exist_ok=True)
        temp_filepath = os.path.join(TEMP_DIR, uploaded_file.name)
        
        # --- FIX: Use .read() for universal Streamlit compatibility ---
        file_bytes = uploaded_file.read()
        with open(temp_filepath, "wb") as f:
            f.write(file_bytes) 
        # -----------------------------------------------------------

        try:
            loader = load_document_for_upload(temp_filepath)
            if loader is None: return 

            new_docs = loader.load()
            for doc in new_docs:
                doc.metadata['role_access'] = user_role 
                doc.metadata['filename'] = uploaded_file.name
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
            new_chunks = text_splitter.split_documents(new_docs)

            existing_store = FAISS.load_local(VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)
            new_store = FAISS.from_documents(new_chunks, embeddings)
            existing_store.merge_from(new_store) 

            existing_store.save_local(VECTOR_DB_PATH)
            st.toast(f"✅ KB Updated: '{uploaded_file.name}' added with '{user_role}' access.", icon='🚀')
            st.cache_resource.clear()
        
        except Exception as e:
            st.error(f"Error processing uploaded file: {e}")
        finally:
            if os.path.exists(temp_filepath): os.remove(temp_filepath)
            if uploaded_file.name.endswith(".xlsx"):
                csv_filepath = os.path.join(TEMP_PROCESSING_DIR, uploaded_file.name.replace(".xlsx", ".csv"))
                if os.path.exists(csv_filepath): os.remove(csv_filepath)
            
            
def load_rag_chain(user_role: str, embeddings):
    if not os.path.exists(VECTOR_DB_PATH):
        st.error("Vector store not found. Please run ingestion.py first.")
        st.stop()
    
    vector_store = FAISS.load_local(VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)
    access_filter = {"role_access": "Employee"} if user_role == 'Employee' else None
    
    retriever = vector_store.as_retriever(search_kwargs={"k": 3, "filter": access_filter})
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GEMINI_API_KEY, temperature=0.1)

    template = """You are an expert, helpful Knowledge Base Assistant for corporate policies.
    Answer the user's question ONLY based on the context provided below.

    * **LANGUAGE:** Respond fluently in the same language as the user's question.
    * **GROUNDING:** If you find the answer, cite the source document name and page number using superscript notation (e.g., [1], [2], etc.) immediately after the relevant sentence.
    * **CONFLICT:** If retrieved documents provide conflicting answers, state the conflict clearly and list all conflicting sources.
    * **NO ANSWER:** If the answer is not in the context, state that you cannot find the answer in the provided documents.

    Context: {context}
    Question: {question}
    Answer:"""
    custom_prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": custom_prompt}
    )

# --- SIMULATED LOGIN FUNCTION (Stable) ---

def login_placeholder():
    st.header("Welcome to Gyaan Portal") 
    st.caption("Internal KnowledgeBase Access")

    with st.form("login_form"):
        username = st.text_input("Username (Try: employee / manager)")
        password = st.text_input("Password (Placeholder)", type="password") 
        submitted = st.form_submit_button("Sign In")

        if submitted:
            if username.lower() == "manager":
                st.session_state['logged_in'] = True
                st.session_state['user_role'] = 'Manager'
                st.success("Manager Login Successful! Redirecting...")
                st.rerun()
            elif username.lower() == "employee":
                st.session_state['logged_in'] = True
                st.session_state['user_role'] = 'Employee'
                st.success("Employee Login Successful! Redirecting...")
                st.rerun()
            else:
                st.error("Invalid credentials. Please use 'employee' or 'manager'.")
        
        if 'logged_in' not in st.session_state:
            st.session_state['logged_in'] = False
            st.session_state['user_role'] = 'None'

# --- STREAMLIT UI ---
def main():
    # --- LOGIN GATE ---
    if 'logged_in' not in st.session_state or not st.session_state['logged_in']:
        login_placeholder()
        return 

    user_role = st.session_state['user_role']
    
    # --- FIX 1: INITIALIZE EMBEDDINGS GLOBALLY ---
    embeddings = get_embeddings() 
    
    # --- 1. LANGUAGE AND HEADER SETUP ---
    LANGUAGE_OPTIONS = {
        "English": {"title": "Gyaan: Your AI Agent for Company Docs!", "input": "Message Gyaan... Ask about policies, benefits, or performance data.", "initial": "Hello! I am ready to answer questions about the company policies."},
        "Hindi (हिंदी)": {"title": "ज्ञान: कंपनी दस्तावेज़ों के लिए आपका एआई एजेंट!", "input": "ज्ञान को संदेश भेजें... नीतियों, लाभों या प्रदर्शन डेटा के बारे में पूछें।", "initial": "नमस्ते! मैं कंपनी की नीतियों के बारे में सवालों का जवाब देने के लिए तैयार हूँ।"},
        "Kannada (ಕನ್ನಡ)": {"title": "ಜ್ಞಾನ: ಕಂಪನಿ ದಾಖಲೆಗಳಿಗಾಗಿ ನಿಮ್ಮ AI ಏಜೆಂಟ್!", "input": "ಜ್ಞಾನಕ್ಕೆ ಸಂದೇಶ ಕಳುಹಿಸಿ... ನೀತಿಗಳು, ಪ್ರಯೋಜನಗಳು ಅಥವಾ ಕಾರ್ಯಕ್ಷಮತೆಯ ಡೇಟಾದ ಬಗ್ಗೆ ಕೇಳಿ.", "initial": "ನಮಸ್ಕಾರ! ನಾನು ಕಂಪನಿ ನೀತಿಗಳ ಬಗ್ಗೆ ಪ್ರಶ್ನೆಗಳಿಗೆ ಉತ್ತರಿಸಲು ಸಿದ್ಧವಾಗಿದ್ದೇನೆ."}
    }
    
    # Use sidebar for non-essential controls
    with st.sidebar:
        selected_language = st.selectbox(
            label="Select Language",
            options=list(LANGUAGE_OPTIONS.keys()),
            index=0,
            key="lang_select_sidebar"
        )
    lang_config = LANGUAGE_OPTIONS[selected_language]
    
    # Main Content Area Header
    st.title(lang_config['title'])
    st.markdown(f"**Logged in as:** `{user_role}`", unsafe_allow_html=True)
    st.markdown("---") 

    # --- 2. UPLOAD WIDGET (Stabilized in Sidebar) ---
    with st.sidebar:
        if user_role == 'Manager':
            st.subheader("Document Ingestion & Update")
            st.caption("Supported: PDF, DOCX, CSV, XLSX")
            
            uploaded_file = st.file_uploader(
                "Upload Document to Update Knowledge Base",
                type=["pdf", "docx", "csv", "xlsx"],
                accept_multiple_files=False,
                key="manager_upload_input_sidebar",
                help="Select a file to merge into the live vector index."
            )
            
            if uploaded_file:
                if st.button("🚀 Process & Merge KB Update"):
                    with st.spinner(f"Updating index with {uploaded_file.name}..."):
                        # embeddings is now defined and accessible here (FIXED)
                        add_to_vector_store(uploaded_file, user_role, embeddings)
                        st.session_state.messages = [] 
                        st.rerun() 

    # 3. RAG Initialization
    rag_chain = load_rag_chain(user_role, embeddings)
    if not rag_chain: st.stop()
        
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({"role": "assistant", "content": f"{lang_config['initial']} You are currently accessing the Knowledge Base."})

    # 4. CHAT HISTORY DISPLAY (Standard, bug-free flow)
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 5. CHAT INPUT AND PROCESSING
    user_query = st.chat_input(lang_config['input'])

    if user_query:
        # 1. Add the user query to history immediately
        st.session_state.messages.append({"role": "user", "content": user_query})
        st.rerun()

    # 6. GENERATE RESPONSE (Runs only after user submits a new query)
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        
        with st.chat_message("assistant"):
            with st.spinner(f"Searching available knowledge..."):
                try:
                    result = rag_chain.invoke({"query": st.session_state.messages[-1]["content"]})
                    answer = result['result']
                    
                    # Collect Sources for Footnote References
                    source_references = {}
                    for doc in result['source_documents']:
                        ref_key = f"{doc.metadata.get('filename', 'Unknown File')}, Page: {doc.metadata.get('page', 'N/A')}"
                        source_references[ref_key] = True

                    # Format References List for the footer
                    ref_list_markdown = ""
                    if source_references:
                        ref_list_markdown = "\n\n---\n**References:**\n"
                        for i, ref in enumerate(source_references.keys()):
                            ref_list_markdown += f"* **[{i+1}]** {ref}\n"
                        
                    final_response = answer + ref_list_markdown
                    st.session_state.messages.append({"role": "assistant", "content": final_response})
                    st.rerun() 

                except Exception as e:
                    error_message = f"An API Error occurred. Please try again or check the system status."
                    if "Quota exceeded" in str(e):
                        error_message = "The daily answer quota has been temporarily reached. Please try again later."
                    st.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})
                    st.rerun() 

if __name__ == "__main__":
    main()