# ✨ Knowledge Lens AI — Interactive Streamlit Multi-PDF RAG Application

**Knowledge Lens AI** is an interactive, locally hosted Retrieval-Augmented Generation (RAG) web application built with **Streamlit**, **LangChain**, **ChromaDB**, **Sentence Transformers**, and **Ollama (`gemma`)**. It enables users to upload multiple PDF documents, switch dynamically between active knowledge sources, ask single-document or cross-document questions, inspect retrieved source chunks, and automatically log all interactions to CSV.

---

## Project Folder Structure

```text
STREAMLIT LLM APPLICATION/
├── .streamlit/
│   └── config.toml
├── assets/
│   ├── logo.png
│   └── sparkle.png
├── chroma_db/
├── dataset/
│   ├── uploaded_files/
│   ├── answers.csv
│   ├── queries.csv
│   └── sample.pdf
├── notebook/
│   └── Task14_Streamlit_LLM_App.ipynb
├── screenshots/
├── app.py
├── pdf_qa.py
├── README.md
└── requirements.txt
```

---

## Key Features

- **Multi-PDF Upload & Strict Validation:** Upload one or multiple `.pdf` files with extension and MIME-type (`application/pdf`) validation and automatic ChromaDB indexing.
- **Active Document Switcher & Page Counter:** Select any combination of uploaded PDFs in the sidebar and view combined page counts in real time.
- **Hybrid Semantic + Keyword Retrieval (`pdf_qa.py`):** Combines `all-MiniLM-L6-v2` cosine similarity with sub-query decomposition, IDF keyword weighting, named-entity/phrase boosting, and Table-of-Contents/References filtering.
- **Adaptive Response Length Modes:** Switch between `Auto`, `Concise`, and `Detailed` profiles to dynamically adjust prompt instructions and token generation limits.
- **Transparent Source Inspection & Metrics:** View the exact retrieved source chunks below every answer alongside response generation time (`Generated in X sec`).
- **Multi-Session Chat History & CSV Logging:** Create, switch, or clear multiple chat sessions (`New Chat`, `Clear All`) while automatically saving timestamps, active PDFs, questions, answers, and response times to `dataset/queries.csv` and `dataset/answers.csv`.

---

## Installation & Setup

1. **Install Ollama and pull the local Gemma model:**
   ```bash
   ollama pull gemma
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the Streamlit application:**
   ```bash
   streamlit run app.py
   ```

4. **Open in Browser:**
   Navigate to `http://localhost:8501` in your web browser.

---

## Usage Workflow

1. Upload one or more `.pdf` files using the sidebar uploader.
2. Select the active documents to query in **Active PDFs in Knowledge Lens**.
3. Choose a **Response Length** (`Auto`, `Concise`, or `Detailed`).
4. Enter your question in the chat input bar to receive a streamed answer grounded in your PDFs.
5. Expand **View Retrieved Source Chunks** below any response to inspect exact source passages and page numbers.