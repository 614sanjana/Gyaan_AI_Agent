## 🚀 Gyaan: AI KnowledgeBase Agent with RBAC

**Gyaan** is a secure, multilingual **KnowledgeBase Agent** designed to automate Tier 1 internal support for large organizations. It implements a **Hybrid Retrieval-Augmented Generation (RAG)** architecture with built-in **Role-Based Access Control (RBAC)** to manage access to sensitive company documents.

-----

## ✨ Key Features:

### 1\. Security and Access Control

  * **Role-Based Access Control (RBAC):** Securely filters document retrieval based on the user's login role (`Employee` or `Manager`) using **metadata filtering** in the FAISS vector store.
  * **Simulated Authentication:** A placeholder login portal verifies the user's role before granting access to the filtered knowledge base.

### 2\. Live Scalability and Architecture

  * **Live Document Ingestion:** Managers can upload new files (`PDF, DOCX, CSV, XLSX`) directly through the UI. The system performs **Incremental Indexing** by merging the new data into the live FAISS index.
  * **Hybrid RAG Model Strategy:** Uses the **Local Hugging Face Embedding Model** for stable, cost-free indexing and the powerful **Gemini 2.5 Flash** for high-quality, professional answer generation.
  * **Multi-Format Support:** Processes both **unstructured policies** (PDF/DOCX) and **structured data** (CSV/XLSX) within a single knowledge base.

### 3\. User Experience

  * **Multilingual Support:** Supports conversation and accurate retrieval in **English, Hindi, and Kannada**, demonstrated through careful prompt engineering.
  * **Professional Citation:** Answers include **superscript citations** and a **References** list for traceability and trust.

-----

## 💻 Tech Stack and Dependencies

| Component | Technology | Role |
| :--- | :--- | :--- |
| **LLM (Generation)** | **Gemini 2.5 Flash** | Final answer synthesis, conflict resolution, and multilingual translation. |
| **Embedding Model** | **HuggingFace: `all-MiniLM-L6-v2`** | Local, CPU-based model for stable, fast indexing. |
| **Vector Database** | **FAISS** | Storage and rapid semantic retrieval of document vectors and associated metadata. |
| **Frameworks** | LangChain, Streamlit | RAG pipeline orchestration and dynamic web interface. |
| **File Handling** | `pandas`, `unstructured` | Processing multi-format files (DOCX, XLSX, CSV). |

-----

## 🛠️ Setup and Run Instructions

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/614sanjana/Gyaan_AI_Agent
    cd Gyaan_AI_Agent
    ```
2.  **Setup Environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate 
    pip install -r requirements.txt
    ```
3.  **API Key Configuration:**
    Create a file named **`.env`** in the project root and add your key:
    ```
    GEMINI_API_KEY="YOUR_KEY_HERE"
    ```
4.  **Build Knowledge Base:** Run the ingestion script. *(The `faiss_index/` is included for fast initial setup, but this step is required if the `docs/` content changes).*
    ```bash
    python ingestion.py
    ```
5.  **Launch Application:**
    ```bash
    streamlit run app.py
    ```
    (Access the app at `http://localhost:8501`.)

-----

## 💡 Limitations and Future Scope

  * **Authentication:** The current login is a placeholder; future work involves integrating a true **SSO/IdP system** (e.g., Okta or Azure AD) for production readiness.
  * **OCR Integration:** The system currently relies on digital text. **Future Scope** includes integrating the **OCR.space API** to handle scanned or image-based legacy documents.
  * **Chat Memory:** The current chat is stateless (each query is independent). Implementing **session memory** (using Redis or similar) would allow for multi-turn conversations.
