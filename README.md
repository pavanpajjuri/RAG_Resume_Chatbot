# Resume Q&A Chatbot with LangChain + LangGraph + Gradio

A production-ready Resume Question Answering (Q&A) chatbot built with **LangChain**, **LangGraph**, **Gradio**, and **Chroma vector store**. This app allows recruiters, hiring managers, or job seekers to **ask natural language questions** about uploaded resume PDFs and get **context-grounded, reliable answers** using Retrieval-Augmented Generation (RAG). The system combines vector-based semantic search with a powerful LLM (Open AI's GPT) layer for clean, meaningful language generation, ensuring responses are both accurate and professionally phrased.

- Supports multiple PDF resumes
- Uses LangGraph’s tool-using agent with memory
- Dynamic section detection and chunking
- Stateless Gradio interface, shareable via Hugging Face Spaces
- Clean, HR-friendly output formatting


---
## Demo

Try it live: [Hugging Face Space – `pavanpaj/resume_qa_bot_v2`](https://huggingface.co/spaces/pavanpaj/resume_qa_bot_v2)

---
## How It Works

### Architecture Overview

1.  **PDF Upload**: Upload one or more resume PDFs.
2.  **PDF Parsing & Section Splitting**: Uses `PyPDFLoader` and custom regex to detect section headers like “Education”, “Experience”, etc.
3.  **Recursive Chunking**: Splits each section into overlapping chunks using `RecursiveCharacterTextSplitter`.
4.  **Vector Store Creation**: Chunks are embedded using `text-embedding-3-large` and stored in an in-memory `Chroma` vector store.
5.  **LangGraph Agent**:
    - **Decision node**: Chooses between LLM response or tool (retrieval).
    - **Tool node**: Performs similarity search from vector store.
    - **Generate node**: Forms final answer using a context-aware system prompt.
6.  **Gradio ChatInterface**: Simple UI to upload PDFs and ask questions in natural language.


---
## Features

-   Upload **multiple resumes** (`.pdf`)
-   **Dynamic section detection** using regex
-   Uses **OpenAI GPT-3.5** for question answering
-   **LangGraph** agent handles tool use, memory, and transitions
-   Clean and **formal HR assistant tone**
-   Follows strict **no hallucination policy** — fallback if info not in resume
-   Supports **persistent memory** using `MemorySaver`

---
## Requirements

Create a `requirements.txt` file as follows:

```txt
langchain==0.3.26
langchain_community==0.3.27
langchain_openai==0.3.27
chromadb==1.0.15
openai==1.93.3
tiktoken==0.9.0
langgraph==0.5.2
pypdf==5.7.0
gradio==4.30.0
```


---
## Sample Questions

Try questions like:

-   “What are this candidate’s technical skills?”
-   “List the projects mentioned in the resume.”
-   “Where did the candidate work before?”
-   “What degree and university does the candidate have?”
-   “Summarize the professional experience section.”


---
## Components

| Component                                | Description                                                 |
| :--------------------------------------- | :---------------------------------------------------------- |
| `build_rag_pipeline()`                   | Loads PDFs, splits into chunks, builds Chroma vector store  |
| `split_resume_by_detected_sections()`    | Uses regex to detect section headers dynamically            |
| `retrieve()`                             | ToolNode tool to fetch relevant resume chunks               |
| `query_or_respond()`                     | LangGraph decision node — calls retrieve if needed          |
| `generate()`                             | Final answer generation using context and LLM               |
| `resume_chatbot()`                       | Gradio callback to handle uploads, pipeline, and chat       |


---
## Deployment

To deploy on Hugging Face Spaces:

1.  Create a new Space → Gradio → `resume_qa_bot`
2.  Upload:
    -   `app.py`
    -   `requirements.txt`
3.  Set your environment variables via “Settings” → “Secrets”
4.  Deploy

---
## Acknowledgements

-   LangChain
-   LangGraph
-   Gradio
-   OpenAI
-   ChromaDB
