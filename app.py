
import os
import getpass
import re
import gradio as gr

from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langgraph.graph import MessagesState, StateGraph
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage

from langgraph.prebuilt import ToolNode
from langgraph.graph import END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

# Langchain and Langsmith APIs
# Langchain: A framework to build LLM-powered applications by composing prompts, models, tools, memory, and chains.
# Langsmith: A debugging, observability, and evaluation platform for LangChain workflows. Helps inspect inputs, outputs, traces, and improve LLM app quality.
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ["LANGSMITH_API_KEY"] = os.environ.get("LANGSMITH_API_KEY", "")
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "")


# Using GPT 3.5 for chat llm and text-embedding-3-large for embeddings
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
# llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
embeddings = OpenAIEmbeddings(model = 'text-embedding-3-large')

def split_resume_by_detected_sections(text):
    # Find positions of all likely section headers
    pattern = r"(?m)^(?:[A-Z][A-Z\s]{2,}|[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)(?=\s*$)"
    matches = list(re.finditer(pattern, text))

    if not matches:
        # fallback — whole doc as one section
        return [Document(page_content=text, metadata={"section": "full"})]

    chunks = []
    for i, match in enumerate(matches):
        section_name = match.group().strip()
        start = match.end()
        end = matches[i+1].start() if i + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()
        full_text = f"{section_name}\n{section_text}"
        chunks.append(Document(page_content=full_text, metadata={"section": section_name.lower()}))

    return chunks

def build_rag_pipeline(resume_filenames):
    all_docs = []
    for filename in resume_filenames:
        if os.path.exists(filename):
            print(f"- Loading {filename}...")
            loader = PyPDFLoader(filename)
            docs_from_pdf = loader.load()
            # 3. Add the source filename to each page's metadata. This is important!
            for doc in docs_from_pdf:
                doc.metadata['source'] = filename
            all_docs.extend(docs_from_pdf)
        else:
            # If a file is not found, print a warning and continue.
            print(f"- WARNING: The file '{filename}' was not found. Skipping.")

        

    section_splits = []
    for doc in all_docs:
        text = doc.page_content if isinstance(doc, Document) else str(doc)
        section_splits.extend(split_resume_by_detected_sections(text))
    # Initialize a recursive text splitter to divide the document into chunks of max 1000 characters,
    # with a 200-character overlap between consecutive chunks. This overlap helps preserve context continuity.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " "],
        add_start_index=True
    )
    all_splits = text_splitter.split_documents(section_splits)
    print(f"Split resumes into {len(all_splits)} chunks")

    vector_store = Chroma.from_documents(documents=all_splits,
                                        embedding=OpenAIEmbeddings())
    return vector_store


# Define a tool named 'retrieve' using the @tool decorator.
# This tool will return both a formatted string and the actual documents as output.
@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Searches the resume chunks for the most relevant information to answer a question."""
    retrieved_docs = vector_store.similarity_search(query, k=7)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    # Return both the serialized string (for LLM consumption) and the raw documents (for structured use).
    return serialized, retrieved_docs


def query_or_respond(state: MessagesState):
    # Bind the 'retrieve' tool to the LLM for dynamic tool-calling.
    llm_with_tools = llm.bind_tools([retrieve])
    # Invoke the LLM with the current conversation messages (state["messages"]).
    # The LLM will decide whether to call the tool or return a direct answer.
    response = llm_with_tools.invoke(state['messages'])
    return {"messages":[response]}

# Define the tool execution node.
# ToolNode handles execution of tool calls that were requested by the LLM in the previous step.
# It automatically routes tool-calls to the correct function (e.g., `retrieve`) and returns the tool output as a ToolMessage.
tools = ToolNode([retrieve])
#	•	query_or_respond() = Decision node: LLM decides whether to use a tool or not.
#	•	ToolNode([retrieve]) = Execution node: Actually runs the tool when called.
def generate(state: MessagesState):
    # Extract the most recent tool messages in reverse order (latest first)
    recent_tool_messages = []
    for message in reversed(state["messages"]):
      if message.type == "tool":
        recent_tool_messages.append(message)
      else:
        break # Stop when we hit a non-tool message to isolate the latest tool-call output
    # Reverse the order to restore chronological flow (oldest to latest)
    tool_messages = recent_tool_messages[::-1]
    docs_content = "\n\n".join(doc.content for doc in tool_messages)
    # This is our new, much more detailed system prompt.
    system_message_content = (
      "You are a professional HR assistant. Your role is to answer questions strictly based on the candidate's resume information provided below.\n\n"
      "The context block may contain excerpts from one or more resumes. Please follow these instructions:\n"
      "1. Base your answer only on the provided '---CONTEXT---' block. You may use any retrieved context to answer, even if not explicitly labeled."
      "2. Scan all chunks in the context, especially sections like EDUCATION, EXPERIENCE, and PROJECTS, before answering."
      "3. If the source is specified (e.g., 'Source Page: ...'), use it to determine which candidate's resume the content belongs to.\n"
      "4. If the information needed is not found in the context, respond with: 'This information is not available in the provided resume.' Do not guess or fabricate answers.\n"
      "5. Keep your response concise, formal, and relevant to the query.\n\n"
      "---CONTEXT---\n"
      f"{docs_content}"
      "\n---END CONTEXT---"
    )
    conversation_messages = []
    for message in state["messages"]:
      if message.type in ("human","system") or (message.type == "ai" and not message.tool_calls):
        conversation_messages.append(message)
    # Prepend the system instruction and retrieved context to the conversation
    prompt = [SystemMessage(content = system_message_content)] + conversation_messages
    response = llm.invoke(prompt)
    return {"messages":[response]}


graph_builder = StateGraph(MessagesState)
# Add a node that decides whether to respond directly or make a tool call
graph_builder.add_node("query_or_respond", query_or_respond)
# Add a predefined tool execution node to handle any tool calls (e.g., retrieval)
graph_builder.add_node("tools", tools)
# Add a node that generates the final answer using retrieved content (from tool messages)
graph_builder.add_node("generate", generate)
graph_builder.set_entry_point("query_or_respond")
# If the LLM response includes a tool call, route to "tools"; otherwise, end the graph.
graph_builder.add_conditional_edges(
    "query_or_respond",     # Source node
    tools_condition,        # Function that inspects response for tool calls
    {END: END, "tools": "tools"}  # Destination mapping based on condition
)
# Define that after tool execution, go to the generate node
graph_builder.add_edge("tools", "generate")
graph_builder.add_edge("generate", END)
# Initialize an in-memory checkpointer to persist chat state between steps
memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)


resume_path = None
last_uploaded_file = None
vector_store = None  # Cache


def resume_chatbot(messages, history, resume_files):
    global resume_path, last_uploaded_file, vector_store

    if resume_files is not None and resume_files != last_uploaded_file:
        try:
            resume_paths = []
            for file_obj in resume_files:
                file_path = str(file_obj)
                print(f"Saving {file_path} -> uploaded_{os.path.basename(file_path)}")
                saved_path = f"uploaded_{os.path.basename(file_path)}"
                with open(saved_path, "wb") as f_out, open(file_path, "rb") as f_in:
                    f_out.write(f_in.read())
                resume_paths.append(saved_path)

            last_uploaded_file = resume_files  # memoize
            vector_store = build_rag_pipeline(resume_paths)

        except Exception as e:
            print("Error handling uploaded files:", str(e))
            return "Failed to process uploaded resumes."

    # Only continue if vector store is ready
    if vector_store is None:
        return "Please upload a resume to begin."

    config = {"configurable": {"thread_id": "gradio_thread"}}
    result = graph.invoke(
        {"messages": messages},
        config=config,
    )
    return result["messages"][-1].content


# Gradio Interface with file upload
demo = gr.ChatInterface(
    fn=resume_chatbot,
    # additional_inputs=[gr.File(label="Upload Resume", file_types=[".pdf"])],
    additional_inputs=[gr.File(label="Upload Resume(s)", file_types=[".pdf"], file_count="multiple")],
    title="Resume Q&A Chatbot",
    type="messages"
)

demo.launch(debug = True)