import os
import getpass
import re
import uuid
import gradio as gr
import chromadb 


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

from gradio.themes.utils import fonts, colors, sizes


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
        # Your loading logic here... (no changes needed)
        if os.path.exists(filename):
            print(f"- Loading {filename}...")
            loader = PyPDFLoader(filename)
            docs_from_pdf = loader.load()
            for doc in docs_from_pdf:
                doc.metadata['source'] = os.path.basename(filename)
            all_docs.extend(docs_from_pdf)
        else:
            print(f"- WARNING: The file '{filename}' was not found. Skipping.")

    # Your splitting logic here... (no changes needed)
    section_splits = []
    for doc in all_docs:
        text = doc.page_content if isinstance(doc, Document) else str(doc)
        section_splits.extend(split_resume_by_detected_sections(text))
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " "],
        add_start_index=True
    )
    all_splits = text_splitter.split_documents(section_splits)
    # print(f"Split resumes into {len(all_splits)} chunks")

    # 1. Manually create a new, truly ephemeral Chroma client.
    #    This guarantees the database is 100% fresh and in-memory.
    ephemeral_client = chromadb.EphemeralClient()

    # 2. Pass this new client to Chroma and give the collection a unique name.
    vector_store = Chroma.from_documents(
        documents=all_splits,
        embedding=OpenAIEmbeddings(),
        client=ephemeral_client, # This is the crucial change
        collection_name=f"resume_collection_{uuid.uuid4()}" # Ensures no name clashes
    )
    
    # print(f"--- Created a new, isolated vector store ---")
    return vector_store

def make_retrieve_tool(vector_store):
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
    return retrieve



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
      # This is the string you would use in your `generate` function
      "You are a professional HR assistant. Your role is to answer questions strictly based on the candidate's resume information provided in the context below."
      "\n\n"
      "Please follow these instructions precisely:\n"
      "1. **Identify the candidate's name** from the context. The name is often at the top of the resume or in the 'source' metadata. In all your responses, you MUST refer to the person by their identified name (e.g., 'Pavan's experience includes...') instead of using generic terms like 'the candidate'.\n"
      "2. Base your answer **exclusively** on the information within the '---CONTEXT---' block. Do not use any external knowledge.\n"
      "3. Scan all provided context chunks, especially sections like 'EXPERIENCE' and 'EDUCATION', before forming your answer to ensure it is comprehensive.\n"
      "4. If the information needed to answer the question is not found in the context, you MUST state: 'That information is not available in the provided resume.' If you have identified the candidate's name, personalize the message (e.g., 'That information is not available in Pavan's resume.').\n"
      "5. Keep your response concise and professional."
      "\n\n---CONTEXT---\n"
      f"{docs_content}"
      "\n---END CONTEXT---"
    )
    conversation_messages = []
    for message in state["messages"]:
      # print(message)
      if message.type in ("human","system") or (message.type == "ai" and not message.tool_calls):
        conversation_messages.append(message)
    # Prepend the system instruction and retrieved context to the conversation
    prompt = [SystemMessage(content = system_message_content)] + conversation_messages
    response = llm.invoke(prompt)
    return {"messages":[response]}


def create_graph_with_fresh_memory(retrieve_tool_fn): 
    graph_builder = StateGraph(MessagesState)
    # Add a node that decides whether to respond directly or make a tool call

    def query_or_respond(state: MessagesState):
        # Bind the 'retrieve' tool to the LLM for dynamic tool-calling.
        llm_with_tools = llm.bind_tools([retrieve_tool_fn])
        # Invoke the LLM with the current conversation messages (state["messages"]).
        # The LLM will decide whether to call the tool or return a direct answer.
        response = llm_with_tools.invoke(state['messages'])
        return {"messages":[response]}

    # Define the tool execution node.
    # ToolNode handles execution of tool calls that were requested by the LLM in the previous step.
    # It automatically routes tool-calls to the correct function (e.g., `retrieve`) and returns the tool output as a ToolMessage.
    tools = ToolNode([retrieve_tool_fn])
    #	•	query_or_respond() = Decision node: LLM decides whether to use a tool or not.
    #	•	ToolNode([retrieve]) = Execution node: Actually runs the tool when called.
    
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
    return graph_builder.compile(checkpointer=MemorySaver())


# ==============================================================================
# Gradio Script for UI
# ==============================================================================

def handle_resume_upload(resume_files, state):
    """Handles the file upload, creates the RAG pipeline, and resets the state."""
    if not resume_files:
        return state, None, gr.update(interactive=False, placeholder="Upload a resume to begin...")

    try:
        resume_paths = [file_obj.name for file_obj in resume_files]
        vector_store = build_rag_pipeline(resume_paths)
        retrieve_tool = make_retrieve_tool(vector_store)
        thread_id = str(uuid.uuid4())
        graph = create_graph_with_fresh_memory(retrieve_tool)
        
        new_state = {"graph": graph, "thread_id": thread_id}
        # print(f"--- NEW SESSION CREATED. Thread ID: {thread_id} ---")

        # 1. Create a clean list of filenames for showing the uploaded resumes message
        filenames_str = ", ".join([os.path.basename(p) for p in resume_paths])
        welcome_message = f"Successfully received and processed: **{filenames_str}**. You can now ask any questions."
        initial_message_for_display = [{"role": "assistant", "content": welcome_message}]
        
        
        # Clear chatbot and enable/update the input box
        return new_state, initial_message_for_display, gr.update(interactive=True, placeholder="Ask a question about the uploaded resume(s)...")

    except Exception as e:
        print(f"Error handling uploaded files: {e}")
        return state, None, gr.update(interactive=False, value=f"Error: {e}")


def resume_chat_turn(user_input, history, state):
    """Handles a single turn of the chat, using and returning the 'messages' format."""
    if not user_input:
        yield gr.update(interactive=True), history
        return history

    if not state.get("graph"):
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": "Please upload resume(s) first before asking questions."})
        yield "", history
        return history

    # The 'history' is already a list of dictionaries. Just add the new user message.
    history.append({"role": "user", "content": user_input})
    yield "", history

    # 2. Add a placeholder "typing" message and yield another update
    history.append({"role": "assistant", "content": "..."})
    yield None, history

    config = {"configurable": {"thread_id": state["thread_id"]}}
    result = state["graph"].invoke({"messages": history[:-1]}, config=config)
    assistant_message = result["messages"][-1].content
    
    # Append the assistant's response to the history
    history.pop()
    history.append({"role": "assistant", "content": assistant_message})
    
    # Return the entire updated history list
    yield None, history


final_theme = gr.themes.Soft(
    font=fonts.GoogleFont("Roboto")
).set(
    body_background_fill_dark="#111111",
    block_background_fill_dark="#1C1C1C",
    block_border_width_dark="1px",
    block_border_color_dark="#2A2A2A",
)

# --- Final Gradio Interface using gr.Blocks and the correct 'messages' format ---
with gr.Blocks(theme=final_theme, title="Resume Q&A Chatbot") as demo:
    session_state = gr.State({})
    
    gr.Markdown("# Resume Q&A Chatbot")
    
    upload_button = gr.UploadButton(
        "📁 Click to Upload Resume(s)",
        file_types=[".pdf"],
        file_count="multiple"
    )

    chatbot_display = gr.Chatbot(
        label="Conversation", 
        height=500, 
        type='messages'  # This is crucial
    )
    
    user_input_box = gr.Textbox(
        label="Your Question",
        placeholder="Upload a resume above to begin...",
        interactive=False,
        autofocus=True
    )
    
    user_input_box.submit(
        fn=resume_chat_turn,
        inputs=[user_input_box, chatbot_display, session_state],
        outputs=[user_input_box, chatbot_display]
    )
    
    upload_button.upload(
        fn=handle_resume_upload,
        inputs=[upload_button, session_state],
        outputs=[session_state, chatbot_display, user_input_box]
    )
    gr.Markdown(
    "<div style='text-align: center; color: #949494; font-size: small; padding-top: 20px;'>"
    "Built by Pavan Pajjuri"
    "</div>"
)

demo.launch(debug=True)
