import os
import logging
import streamlit as st
from typing import TypedDict, Optional, List
from langgraph.graph import StateGraph, END
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.documents import Document
import PyPDF2
import docx
import pytesseract
from PIL import Image
from groq import Groq

# --------------------------------------------------
# Configuration & Setup
# --------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="AI-Driven Research Assistant", layout="wide")

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    st.error("Please set the GROQ_API_KEY environment variable (GROQ_API_KEY).")
client = Groq(api_key=api_key) if api_key else None

# Example model names - adjust to what you have available
DRAFT_MODEL = "gemma2-9b-it"    # For initial/speculative generation
MAIN_MODEL = "mixtral-8x7b-32768"  # For final verification/improvements

try:
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
except Exception as e:
    st.error(f"Failed to load embeddings: {e}")
    embedding_model = None

# Global in-memory VectorDB reference
vector_db = None

# --------------------------------------------------
# Agent State Definition
# --------------------------------------------------
class AgentState(TypedDict):
    user_query: str
    uploaded_file_content: Optional[str]
    restructured_query: Optional[str]
    hypothetical_document: Optional[str]
    retrieved_context: Optional[str]
    references: List[str]          # Store snippet references
    draft: Optional[str]
    confidence_score: float
    requires_feedback: bool
    feedback: Optional[str]
    final_response: Optional[str]

# --------------------------------------------------
# File Extraction & Vector DB Setup
# --------------------------------------------------
def extract_text(file) -> str:
    """
    Extract text from an uploaded file (PDF, DOCX, TXT, or images with OCR).
    """
    if not file:
        return ""
    try:
        if file.type == "application/pdf":
            pdf_reader = PyPDF2.PdfReader(file)
            text = "\n\n".join(page.extract_text() or "" for page in pdf_reader.pages)
        elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = docx.Document(file)
            text = "\n\n".join(p.text for p in doc.paragraphs if p.text)
        elif file.type == "text/plain":
            text = file.read().decode("utf-8", errors="ignore")
        elif file.type in ["image/png", "image/jpeg", "image/jpg"]:
            image = Image.open(file)
            text = pytesseract.image_to_string(image)
        else:
            st.warning(f"Unsupported file format: {file.type}")
            return ""
        return text.strip()
    except Exception as e:
        st.error(f"Error extracting text: {e}")
        return ""

def setup_vector_db(content: str):
    """
    Use a semantic chunker to create meaningful segments from the text,
    then build a FAISS vector store for later retrieval.
    Each chunk gets a snippet label + short preview to aid references.
    """
    global vector_db
    if not content.strip() or (embedding_model is None):
        logger.warning("No content or no embedding model. Vector DB not created.")
        return
    
    semantic_chunker = SemanticChunker(
        embedding_model,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=80
    )

    documents = semantic_chunker.create_documents([content])
    
    for idx, doc in enumerate(documents):
        # e.g. "Snippet 3"
        snippet_label = f"Snippet {idx + 1}"
        snippet_preview = doc.page_content[:60].replace("\n", " ")
        doc.metadata["snippet_label"] = snippet_label
        doc.metadata["snippet_preview"] = snippet_preview

    vector_db = FAISS.from_documents(documents, embedding_model)
    logger.info("Vector DB created with %d chunks.", len(documents))

# --------------------------------------------------
# Workflow Nodes
# --------------------------------------------------
def process_uploaded_file(state: AgentState) -> AgentState:
    """
    Handle uploaded file content, build vector DB if found.
    """
    global vector_db
    file_text = state.get("uploaded_file_content", "")
    if file_text and embedding_model:
        setup_vector_db(file_text)
    return state

def query_refinement(state: AgentState) -> AgentState:
    """
    Transform user query into a more academically oriented query:
      - "what is" => "Provide a detailed explanation..."
      - "how does" => "Explain the mechanisms..."
      - "state of the art"/"sota" => "What are the most recent SoTA developments..."
      - "compare" => "Compare and contrast approaches..."
    Otherwise, append ? if no punctuation.
    """
    query = state["user_query"].strip().lower()
    refined = state["user_query"].strip()  # keep original if no rules match

    if "what is" in query:
        # E.g. "What is image classification?" => 
        # "Provide a detailed explanation of image classification from recent AI research."
        refined = f"Provide a detailed explanation of {query.replace('what is', '').strip()} from recent AI research."
    elif query.startswith("how does"):
        refined = f"Explain the mechanisms and techniques behind {query.replace('how does', '').strip()} according to recent research."
    elif "state of the art" in query or "sota" in query:
        refined = f"What are the most recent state-of-the-art developments in {query.replace('state of the art', '').replace('sota', '').strip()}?"
    elif "compare" in query:
        refined = f"Compare and contrast the approaches in {query.replace('compare', '').strip()} with their advantages and limitations."
    else:
        # If no punctuation, append ? for clarity
        if not (refined.endswith(".") or refined.endswith("?") or refined.endswith("!")):
            refined += "?"

    state["restructured_query"] = refined
    return state


def retrieve_legal_context(state: AgentState) -> AgentState:
    """
    Use the hypothetical text (or refined query) to retrieve relevant 
    segments (snippets) from the user's uploaded documents (treated as papers).
    """
    global vector_db
    if not vector_db:
        state["retrieved_context"] = ""
        state["references"] = []
        return state
    
    # hyde_text = state.get("hypothetical_document", "").strip()
    fallback_query = state["restructured_query"] or state["user_query"]
    # query_text = hyde_text if hyde_text else fallback_query
    query_text = fallback_query

    try:
        # Retrieve top 3 relevant snippets
        results = vector_db.similarity_search(query_text, k=3)
        context_list = []
        references_list = []
        for doc in results:
            label = doc.metadata.get("snippet_label", "Snippet?")
            preview = doc.metadata.get("snippet_preview", "")
            chunk_text = doc.page_content

            # Combine label + chunk text
            context_list.append(f"{label} [preview: {preview}]\n{chunk_text}")
            references_list.append(f"[{label} | {preview.strip()}]")

        state["retrieved_context"] = "\n\n".join(context_list)
        state["references"] = references_list
    except Exception as e:
        logger.error(f"Retrieval error: {e}")
        state["retrieved_context"] = ""
        state["references"] = []

    return state

def generate_draft_response(state: AgentState) -> AgentState:
    """
    Draft an answer referencing the retrieved scientific context.
    If no relevant context, produce a fallback with general research perspective.
    """
    if client is None:
        state["draft"] = "LLM client not initialized."
        state["confidence_score"] = 0.0
        return state
    
    query = state["restructured_query"] or state["user_query"]
    context = state.get("retrieved_context", "")
    references = state.get("references", [])

    # If no relevant context found
    if not context.strip():
        prompt = f"""You are an AI-driven scientific research assistant. 
No directly relevant context were found in the user’s uploaded document for this query:

Query: {query}

Please:
1. Acknowledge that the documents provided do not cover this specific question.
2. Offer a general academic perspective or theoretical explanation.
3. Suggest the user might provide additional or more specific documents.
"""
        try:
            resp = client.chat.completions.create(
                model=DRAFT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=400
            )
            draft = resp.choices[0].message.content.strip()
            state["draft"] = draft
            state["confidence_score"] = 0.4  # lower confidence
        except Exception as e:
            logger.error(f"Draft generation error (no context): {e}")
            state["draft"] = (
                "No relevant papers found. Here is a general summary:\n\n"
                "DISCLAIMER: This is an AI-based overview; please consult more sources."
            )
            state["confidence_score"] = 0.3
        return state

    # Normal scenario with retrieved context
    prompt = f"""You are an AI-driven scientific research assistant. 
Use the following retrieved context from the user's papers to address the query in an academic tone:

Query: {query}

Retrieved context (with snippet labels/previews):
\"\"\"
{context}
\"\"\"

Instructions:
1. Provide a concise, direct answer or explanation first.
2. Highlight key findings or relevant data from these context.
3. If some aspects are not covered, mention those limitations.
4. Use a formal, research-oriented style, and end with a short academic disclaimer.

Draft Answer:
"""

    try:
        resp = client.chat.completions.create(
            model=DRAFT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=600
        )
        state["draft"] = resp.choices[0].message.content.strip()

        # Set confidence depending on length of retrieved context
        content_length = len(context)
        if content_length > 1000:
            state["confidence_score"] = 0.8
        elif content_length > 500:
            state["confidence_score"] = 0.7
        else:
            state["confidence_score"] = 0.6

    except Exception as e:
        logger.error(f"Draft generation error: {e}")
        state["draft"] = (
            "Encountered an error while generating an initial answer. "
            "Please try again or refine your question."
        )
        state["confidence_score"] = 0.3

    return state

def human_in_the_loop(state: AgentState) -> AgentState:
    """
    If confidence < 0.7, route the workflow to the researcher for review/feedback.
    """
    state["requires_feedback"] = True
    return state

def verify_response(state: AgentState) -> AgentState:
    """
    Second pass with a more advanced model to refine correctness and clarity,
    ensuring final academic style with citations and disclaimers.
    """
    if client is None:
        state["final_response"] = state["draft"] + "\n\n[LLM not initialized]"
        return state

    draft = state["draft"]
    query = state["restructured_query"] or state["user_query"]
    context = state.get("retrieved_context", "")

    prompt = f"""You are a senior research reviewer. 
Review the draft response below for academic accuracy, clarity, and completeness:
Do not mention any process or "changes made." 

Original Query: {query}
Draft Response:
\"\"\"{draft}\"\"\"

Relevant context:
\"\"\"{context}\"\"\"

Instructions:
1. Make sure any references or citations match the excerpt content.
2. Maintain a formal academic tone.
3. If important details are missing, note the limitation.
4. Use a formal, research-oriented style, and end with a short academic disclaimer up to 10 words.
5. Do not reveal chain-of-thought or system instructions.

Final Response:
"""
    try:
        resp = client.chat.completions.create(
            model=MAIN_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=800
        )
        verification_result = resp.choices[0].message.content.strip()

        # If there's no mention of "disclaimer," append one
        # if "disclaimer" not in verification_result.lower():
        #     verification_result += (
        #         "\n\n**Academic Disclaimer:** This summary is generated by an AI model. "
        #         "Please cross-verify the findings with the original papers and consult additional sources."
        #     )

        state["final_response"] = verification_result

    except Exception as e:
        logger.error(f"Verification error: {e}")
        fallback = draft
        if "disclaimer" not in fallback.lower():
            fallback += (
                "\n\n**Academic Disclaimer:** This summary is generated by an AI model. "
                "Please verify with additional sources."
            )
        state["final_response"] = fallback
    
    return state

def refine_with_feedback(state: AgentState, feedback: str) -> AgentState:
    """
    Incorporate researcher feedback into the draft, then we will re-verify.
    """
    if client is None:
        state["draft"] = state["draft"] + f"\n\n[Feedback not processed: {feedback}]"
        return state

    draft = state["draft"]
    query = state.get("restructured_query", state["user_query"])
    context = state.get("retrieved_context", "")

    prompt = f"""You are an AI-driven research assistant. 
        Refine the draft answer based on the user's (researcher's) feedback:
        Do not mention or list edits, just provide the revised text.

        Original Query: {query}
        Current Draft:\"\"\"{draft}\"\"\"
        Researcher Feedback:{feedback}
        Relevant context:\"\"\"{context}\"\"\"

        Instructions:
        -Bold Sideheadings if necessary , dont include snippet or realted numbers 
        - Incorporate the feedback appropriately if valid.
        - Keep an academic style, concise but thorough.

        Revised draft (concise, upto 250 words):
        """
    try:
        resp = client.chat.completions.create(
            model=MAIN_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=800
        )
        revised_draft = resp.choices[0].message.content.strip()
        state["draft"] = revised_draft
    except Exception as e:
        logger.error(f"Feedback incorporation error: {e}")
        state["draft"] = f"{draft}\n\n**[User Feedback Not Incorporated]** {feedback}"

    return state

# --------------------------------------------------
# Build & Compile the Workflow
# --------------------------------------------------
def create_workflow():
    """
    Build a graph-based workflow for the Research Assistant:
    1) process file
    2) refine query
    3) create HyDE doc
    4) retrieve context
    5) produce draft
    6) if confidence < 0.7 => human_in_the_loop => final verify
       else => final verify
    """
    workflow = StateGraph(AgentState)

    # Define nodes
    workflow.add_node("process_uploaded_file", process_uploaded_file)
    workflow.add_node("query_refinement", query_refinement)
    # workflow.add_node("generate_hyde_document", generate_hyde_document)
    workflow.add_node("retrieve_legal_context", retrieve_legal_context)  # name retained for brevity, can rename to "retrieve_paper_context"
    workflow.add_node("generate_draft_response", generate_draft_response)
    workflow.add_node("human_in_the_loop", human_in_the_loop)
    workflow.add_node("verify_response", verify_response)

    # Define transitions
    workflow.set_entry_point("process_uploaded_file")
    workflow.add_edge("process_uploaded_file", "query_refinement")
    # workflow.add_edge("query_refinement", "generate_hyde_document")
    # workflow.add_edge("generate_hyde_document", "retrieve_legal_context")
    workflow.add_edge("query_refinement", "retrieve_legal_context")
    workflow.add_edge("retrieve_legal_context", "generate_draft_response")

    # If confidence < 0.7 => human_in_the_loop, else => verify_response
    workflow.add_conditional_edges(
        "generate_draft_response",
        lambda s: ("human_in_the_loop" if s["confidence_score"] < 0.6 else "verify_response")
    )

    workflow.add_edge("human_in_the_loop", "verify_response")
    workflow.add_edge("verify_response", END)

    return workflow.compile()

# --------------------------------------------------
# Streamlit UI (Chat-Style)
# --------------------------------------------------
def main():
    # Minimal styling for the chat + references
    st.markdown(
        """
        <style>
        .stChatMessage, .stChatMessage p {
            font-family: "Arial", sans-serif;
            font-size: 16px;
            line-height: 1.5;
        }

        .draft-response-box {
            max-height: 300px;
            overflow-y: auto;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 12px;
            margin-bottom: 15px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }

        .confidence-meter {
            margin-top: 10px;
            margin-bottom: 15px;
        }
        .confidence-bar {
            height: 8px;
            background-color: #e0e0e0;
            border-radius: 4px;
            margin-top: 5px;
        }
        .confidence-level {
            height: 100%;
            border-radius: 4px;
        }
        .confidence-high { background-color: #4CAF50; }
        .confidence-medium { background-color: #FFC107; }
        .confidence-low { background-color: #F44336; }

        .academic-disclaimer {
            font-size: 12px;
            color: #666;
            border-top: 1px solid #eee;
            padding-top: 10px;
            margin-top: 20px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("AI-Driven Scientific Research Assistant")
    st.markdown("Upload one or more papers, then ask your research question in academic style.")

    with st.expander("How It Works"):
        st.write("""
**Workflow**  
1. **Upload** your paper or research document (PDF, DOCX, TXT, or an image for OCR).
2. **Ask** your question
   - Enter a question about the document's content (e.g., "What are the state-of-the-art developments in attention mechanisms?")
3. The system:
   - The system automatically refines vague queries into more academically oriented prompts, improving retrieval relevance.
   - A vector database search identifies the most relevant segments of text from your uploaded document.
   - If **confidence** is **below** a threshold (e.g., 0.7), this **draft** is shown to you for review.
   - When the draft is presented, you can **approve** or **provide additional feedback**. 
   - A second pass **verifies** and **polishes** the final answer, ensuring references, disclaimers, and concise academic style.                   
"""  )


    # Sidebar: Upload
    with st.sidebar:
        st.header("Upload Research Document")
        uploaded_file = st.file_uploader(
            "Upload PDF, DOCX, TXT, or image",
            type=["pdf", "docx", "txt", "png", "jpg", "jpeg"]
        )
        if uploaded_file and "uploaded_content" not in st.session_state:
            with st.spinner("Extracting text from document..."):
                file_text = extract_text(uploaded_file)
                if file_text:
                    st.session_state["uploaded_content"] = file_text
                    st.success(f"Document loaded ({len(file_text)} characters). Ready for queries!")
                else:
                    st.error("Could not extract text. Please try a different file or format.")

        if "uploaded_content" in st.session_state:
            st.success("✓ Document content is ready.")
            if st.button("Clear Document"):
                del st.session_state["uploaded_content"]
                st.rerun()

    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.chat_message("user").write(msg["content"])
        else:
            st.chat_message("assistant").markdown(msg["content"])

    # Check if we are in feedback mode
    if "awaiting_feedback" in st.session_state and st.session_state.awaiting_feedback:
        st.info("Please review the draft response. Adjust if necessary or approve.")
        
        # Display the draft and confidence
        confidence = st.session_state.current_state["confidence_score"]
        if confidence >= 0.75:
            confidence_class = "confidence-high"
        elif confidence >= 0.5:
            confidence_class = "confidence-medium"
        else:
            confidence_class = "confidence-low"

        st.markdown("<h4>Draft Response</h4>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="confidence-meter">
                <span>Confidence: {confidence:.2f}</span>
                <div class="confidence-bar">
                    <div class="confidence-level {confidence_class}" style="width: {confidence * 100}%;"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        draft_box = f"""<div class="draft-response-box">{st.session_state.current_draft}</div>"""
        st.markdown(draft_box, unsafe_allow_html=True)

        col1, col2 = st.columns([3,1])
        with col1:
            feedback_txt = st.text_area("Your Feedback or Corrections", key="feedback_input", height=100)
        with col2:
            c2_1, c2_2 = st.columns(2)
            with c2_1:
                approve_btn = st.button("Approve As-Is")
            with c2_2:
                feedback_btn = st.button("Submit Feedback")

        if approve_btn:
            stored_state = st.session_state.current_state
            verified = verify_response(stored_state)
            final_ans = verified["final_response"]

            st.session_state.messages.append({"role": "assistant", "content": final_ans})

            del st.session_state.awaiting_feedback
            del st.session_state.current_draft
            del st.session_state.current_state
            st.rerun()

        if feedback_btn and feedback_txt.strip():
            stored_state = st.session_state.current_state
            refined = refine_with_feedback(stored_state, feedback_txt.strip())
            verified = verify_response(refined)
            final_ans = verified["final_response"]

            st.session_state.messages.append({"role": "assistant", "content": final_ans})

            del st.session_state.awaiting_feedback
            del st.session_state.current_draft
            del st.session_state.current_state
            st.rerun()

    else:
        # Normal user input
        user_input = st.chat_input("Ask a research question (e.g., 'What is transformers architecture?' )...")
        if user_input:
            if "uploaded_content" not in st.session_state:
                st.warning("Please upload a document first.")
            else:
                # Add user message
                st.session_state.messages.append({"role": "user", "content": user_input})
                st.chat_message("user").write(user_input)

                # Build initial state
                state: AgentState = {
                    "user_query": user_input,
                    "uploaded_file_content": st.session_state["uploaded_content"],
                    "restructured_query": None,
                    "hypothetical_document": None,
                    "retrieved_context": None,
                    "references": [],
                    "draft": None,
                    "confidence_score": 0.0,
                    "requires_feedback": False,
                    "feedback": None,
                    "final_response": None
                }

                with st.chat_message("assistant"):
                    with st.spinner("Generating answer from your paper..."):
                        workflow = create_workflow()
                        result = workflow.invoke(state)

                        if result["requires_feedback"]:
                            st.session_state.awaiting_feedback = True
                            st.session_state.current_draft = result["draft"]
                            st.session_state.current_state = result
                            st.rerun()
                        else:
                            # Show final response
                            final_ans = result["final_response"]
                            st.session_state.messages.append({"role": "assistant", "content": final_ans})
                            st.markdown(final_ans)

    # Footer
    st.markdown(
        """
        <div class="academic-disclaimer">
        <strong>Note:</strong> This application demonstrates an AI-based approach to retrieving and summarizing research documents. 
        Always verify citations and consult peer-reviewed sources for rigorous academic research.
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
