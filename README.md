# 🚀 Gyaan: Your AI Agent for Company Docs!

**Gyaan** is a secure, multilingual **KnowledgeBase Agent** designed to automate Tier 1 internal support for large organizations. It uses a **Hybrid Retrieval-Augmented Generation (RAG)** architecture with built-in **Role-Based Access Control (RBAC)** to ensure employees only access public policies while managers can securely analyze sensitive data and update the knowledge base live.

---

## ✨ Key Features & Standout Differentiators

This project demonstrates several enterprise-ready features:

* **Role-Based Access Control (RBAC):** Securely filters search results based on the user's role (`Employee` or `Manager`) using metadata tagging in the Vector Database (FAISS).
* **Live Document Ingestion:** Managers can upload new files (`PDF, DOCX, CSV, XLSX`) directly through the UI, instantly merging the data into the live FAISS index without requiring IT intervention.
* **Hybrid RAG Architecture:** Uses the **Local Hugging Face Embedding Model** for stable, cost-free indexing and the powerful **Gemini 2.5 Flash** for superior answer generation.
* **Multilingual Support:** Supports conversation and accurate retrieval in **English, Hindi, and Kannada**.
* **Data Versatility:** Handles both **unstructured policy documents** (PDF, DOCX) and **structured data analysis** (CSV/XLSX).

---

## 💻 Tech Stack & Dependencies

| Component | Technology | Role |
| :--- | :--- | :--- |
| **LLM (Generation)** | **Gemini 2.5 Flash** | Final answer synthesis and multilingual translation. |
| **Embedding Model** | **HuggingFace: `all-MiniLM-L6-v2`** | Local, CPU-based model for stable, fast, and free indexing. |
| **Vector DB** | **FAISS** | Storage and rapid semantic retrieval of document vectors. |
| **Frameworks** | LangChain, Streamlit | RAG pipeline orchestration and web interface. |
| **Data Handlers** | `pandas`, `unstructured` | Processing complex multi-format files (DOCX, XLSX). |

---

## 🛠️ Setup & Run Instructions

**Pre-requisites:** Python 3.8+, Git, and a Gemini API Key.

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/614sanjana/Gyaan_AI_Agent](https://github.com/614sanjana/Gyaan_AI_Agent)
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
4.  **Launch Application:** The Vector DB (`faiss_index/`) is included in the repository for fast setup.
    ```bash
    streamlit run app.py
    ```
    *(If you modify the documents in the `docs/` folder, run `python ingestion.py` before launching the app.)*

---

## 🖼️ Architecture Diagram (Mandatory Item #3)

Your project uses a dual-path architecture to ensure both security and live updates. 

### 1. **Query Path (Security Enforced):**
* User Query $\rightarrow$ Streamlit UI $\rightarrow$ **Role Check** (Employee/Manager) $\rightarrow$ **Metadata Filter** $\rightarrow$ FAISS Vector Store $\rightarrow$ Retrieval $\rightarrow$ Gemini LLM $\rightarrow$ Answer.

### 2. **Manager Ingestion Path:**
* Manager Upload (PDF/CSV/DOCX) $\rightarrow$ **Tagging (`role_access: Manager`)** $\rightarrow$ Local Embedding $\rightarrow$ **FAISS Index (Merge)** $\rightarrow$ Disk Save.

---

## ☁️ Working Demo Link (Mandatory Item #1)

After pushing the final code to GitHub, your final action is to deploy to Streamlit Community Cloud.

1.  Go to **Streamlit Community Cloud**.
2.  Select **"New App"** and choose your **`614sanjana/Gyaan_AI_Agent`** repository.
3.  In **Advanced Settings**, add your secret: `GEMINI_API_KEY = YOUR_KEY`.
4.  Click **"Deploy!"**

This process will generate your public **Working Demo Link** (Item #1 complete).

---

Once you push the final `README.md` to GitHub and complete the deployment, your submission is ready!
