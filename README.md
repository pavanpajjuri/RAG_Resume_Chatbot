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

## Architecture Overview

### 1. PDF Upload
- Users can upload one or more resume PDFs via the Gradio interface.

### 2. PDF Parsing & Section Splitting
- Each PDF is parsed using `PyPDFLoader` from LangChain.
- Custom regex is used to detect natural section headers like `Education`, `Experience`, `Projects`, etc.
- If headers aren't detected, the full document is treated as a single section.

### 3. Recursive Chunking
- Each section is split into overlapping chunks using `RecursiveCharacterTextSplitter`.
- Each chunk is a `Document` with metadata (e.g., section name, source).
- Overlapping (chunk_overlap=200) ensures context continuity for downstream answers.

### 4. Vector Store Creation
- Each chunk is embedded using OpenAI’s `text-embedding-3-large` model.
- The embedded chunks are stored in an in-memory `Chroma` vector store.
- This store enables fast, similarity-based retrieval.

### 5. LangGraph Agent (Tool-Using • Agentic • Stateful)

The RAG pipeline is powered by a **LangGraph agent**, designed as a stateful and tool-using reasoning graph.

#### Agentic Behavior
- The LLM is wrapped with `bind_tools([retrieve])`, turning it into an **agent** that can:
  - **Interpret user intent**,  
  - **Decide autonomously** whether it needs external tools (like retrieval), and  
  - **Invoke those tools on its own** to complete its reasoning.
- This creates **agent-like behavior**, where the LLM isn't just completing text, but actively **deciding actions** during the conversation.

#### Stateful Memory
- LangGraph uses `MemorySaver` to **preserve conversation history across steps**.
- This allows multi-turn conversations without context loss — the graph always has access to prior messages.

#### Graph Nodes & Flow
- The agent is modeled as a **directed graph of nodes**, each representing a stage in the reasoning workflow:

  - **`query_or_respond` node**
    - The LLM inspects the current message history and decides:
      - **Respond directly**, or
      - **Call a tool** like `retrieve()` to fetch context chunks.

  - **`tools` node**
    - Executes the requested tool(s) — in this case, it performs a **vector similarity search** over embedded resume chunks.
    - The retrieved content is then passed forward as a special `tool` message.

  - **`generate` node**
    - Receives the retrieved documents and builds a **context-aware system prompt** that includes:
      - Grounded resume snippets,
      - Instructions for truthful, non-fabricated answers.
    - The LLM then generates the final output based on this structured input.

#### Flow Logic
- The LangGraph wiring determines which node comes next:
  - If the LLM decides **not** to use a tool:
    ```
    query_or_respond --> END
    ```
  - If the LLM **uses** the retrieval tool:
    ```
    query_or_respond --> tools --> generate --> END
    ```

#### Why This Matters
This structure provides:
- **Tool-use autonomy** (agentic LLM),
- **Memory persistence** (stateful graph),
- **Modular reasoning flow** (node-based control),
- And ensures the **LLM never hallucinates**, as all answers are grounded in retrieved resume chunks.

The agent only answers when it has enough information — otherwise, it calls tools like a human assistant would.

### 6. Gradio ChatInterface
- A clean, user-friendly chat UI built with `gr.ChatInterface`.
- Supports file upload + chat in a single window.
- Runs seamlessly on Hugging Face Spaces with public shareability.

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
