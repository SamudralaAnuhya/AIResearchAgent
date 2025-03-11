# nodes.py
import streamlit as st
import logging
from typing import TypedDict, Optional, List
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_experimental.text_splitter import SemanticChunker
import PyPDF2
import docx
import pytesseract
from PIL import Image
from groq import Groq
from langchain_core.documents import Document

from config import logger, DRAFT_MODEL, MAIN_MODEL, GROQ_API_KEY

"""
All Node Functions for the Workflow
"""

# Global references
embedding_model = None
vector_db = None
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

class AgentState(TypedDict):
    user_query: str
    uploaded_file_content: Optional[str]
    restructured_query: Optional[str]
    hypothetical_document: Optional[str]  # not used if we removed HyDE
    retrieved_context: Optional[str]
    references: List[str]
    draft: Optional[str]
    confidence_score: float
    requires_feedback: bool
    feedback: Optional[str]
    final_response: Optional[str]

def load_embeddings():
    global embedding_model
    if embedding_model is None:
        try:
            embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        except Exception as e:
            st.error(f"Failed to load embeddings: {e}")
            embedding_model = None

def extract_text(file) -> str:
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
    global vector_db, embedding_model
    if not content.strip() or embedding_model is None:
        logger.warning("No content or no embedding model. Vector DB not created.")
        return
    
    semantic_chunker = SemanticChunker(
        embedding_model,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=90
    )

    documents = semantic_chunker.create_documents([content])
    for idx, doc in enumerate(documents):
        label = f"Snippet {idx+1}"
        preview = doc.page_content[:60].replace("\n"," ")
        doc.metadata["snippet_label"] = label
        doc.metadata["snippet_preview"] = preview

    vector_db = FAISS.from_documents(documents, embedding_model)
    logger.info("Vector DB created with %d chunks.", len(documents))

# --------------------
# Workflow Node Functions
# --------------------
def process_uploaded_file(state: AgentState) -> AgentState:
    load_embeddings()
    txt = state.get("uploaded_file_content", "")
    if txt and embedding_model:
        setup_vector_db(txt)
    return state

def query_refinement(state: AgentState) -> AgentState:
    query = state["user_query"].strip().lower()
    refined = state["user_query"].strip()

    if "what is" in query:
        refined = f"Provide a detailed explanation of {query.replace('what is', '').strip()} from recent AI research."
    elif query.startswith("how does"):
        refined = f"Explain the mechanisms and techniques behind {query.replace('how does', '').strip()} according to recent research."
    elif "state of the art" in query or "sota" in query:
        refined = f"What are the most recent state-of-the-art developments in {query.replace('state of the art','').replace('sota','').strip()}?"
    elif "compare" in query:
        refined = f"Compare and contrast the approaches in {query.replace('compare','').strip()} with their advantages and limitations."
    else:
        if not (refined.endswith(".") or refined.endswith("?") or refined.endswith("!")):
            refined += "?"

    state["restructured_query"] = refined
    return state

def retrieve_research_context(state: AgentState) -> AgentState:
    global vector_db
    if not vector_db:
        state["retrieved_context"] = ""
        state["references"] = []
        return state

    query_text = state["restructured_query"] or state["user_query"]
    try:
        results = vector_db.similarity_search(query_text, k=3)
        context_list = []
        refs = []
        for doc in results:
            lbl = doc.metadata.get("snippet_label","Snippet?")
            prv = doc.metadata.get("snippet_preview","")
            chunk_text = doc.page_content

            context_list.append(f"{lbl} [preview: {prv}]\n{chunk_text}")
            refs.append(f"[{lbl} | {prv}]")

        state["retrieved_context"] = "\n\n".join(context_list)
        state["references"] = refs
    except Exception as e:
        logger.error(f"Retrieval error: {e}")
        state["retrieved_context"] = ""
        state["references"] = []

    return state

def generate_draft_response(state: AgentState) -> AgentState:
    if client is None:
        state["draft"] = "LLM client not initialized."
        state["confidence_score"] = 0.0
        return state
    
    query = state["restructured_query"] or state["user_query"]
    context = state.get("retrieved_context", "")

    if not context.strip():
        prompt = f"""You are an AI-driven scientific research assistant. 
No directly relevant context was found in the document for this query:

Query: {query}

Give a concise overview, mention the limitation, and encourage further docs.
"""
        try:
            resp = client.chat.completions.create(
                model=DRAFT_MODEL,
                messages=[{"role":"user","content":prompt}],
                temperature=0.3,
                max_tokens=300
            )
            state["draft"] = resp.choices[0].message.content.strip()
            state["confidence_score"] = 0.4
        except Exception as e:
            logger.error(f"Draft generation error: {e}")
            state["draft"] = "No relevant context found. Basic fallback answer."
            state["confidence_score"] = 0.3
        return state

    prompt = f"""You are an AI research assistant. Use the retrieved context to answer:

Query: {query}

Retrieved Context:
\"\"\"{context}\"\"\"

Refer to relevant sections as needed, but do not explicitly list snippet numbers.
End with a short academic disclaimer in the final sentence.
"""
    try:
        resp = client.chat.completions.create(
            model=DRAFT_MODEL,
            messages=[{"role":"user","content":prompt}],
            temperature=0.3,
            max_tokens=600
        )
        draft = resp.choices[0].message.content.strip()
        state["draft"] = draft

        length_context = len(context)
        if length_context > 1000: 
            state["confidence_score"] = 0.8
        elif length_context > 500:
            state["confidence_score"] = 0.7
        else:
            state["confidence_score"] = 0.6
    except Exception as e:
        logger.error(f"Draft generation error: {e}")
        state["draft"] = "Error generating draft."
        state["confidence_score"] = 0.3
    return state

def human_in_the_loop(state: AgentState) -> AgentState:
    state["requires_feedback"] = True
    return state

def verify_response(state: AgentState) -> AgentState:
    if client is None:
        state["final_response"] = state["draft"]
        return state

    draft = state["draft"]
    query = state["restructured_query"] or state["user_query"]
    context = state.get("retrieved_context","")

    prompt = f"""You are a senior research reviewer. 
Refine the draft for correctness and conciseness, no meta commentary. 
End with one-sentence disclaimer if missing.

Query: {query}
Draft:
\"\"\"{draft}\"\"\"
Context:
\"\"\"{context}\"\"\"

Final response (concise):
"""
    try:
        resp = client.chat.completions.create(
            model=MAIN_MODEL,
            messages=[{"role":"user","content":prompt}],
            temperature=0.2,
            max_tokens=600
        )
        final_resp = resp.choices[0].message.content.strip()
        state["final_response"] = final_resp
    except Exception as e:
        logger.error(f"Verification error: {e}")
        state["final_response"] = draft

    return state

def refine_with_feedback(state: AgentState, feedback: str) -> AgentState:
    if client is None:
        state["draft"] += f"\n\n[Feedback not processed: {feedback}]"
        return state

    draft = state["draft"]
    query = state.get("restructured_query", state["user_query"])
    context = state.get("retrieved_context","")

    prompt = f"""You are an AI assistant. 
Revise the draft based on the feedback below, no mention of process or changes:

Original Query: {query}
Draft:
\"\"\"{draft}\"\"\"
Feedback:
{feedback}
Context:
\"\"\"{context}\"\"\"

Revised draft (concise):
"""
    try:
        resp = client.chat.completions.create(
            model=MAIN_MODEL,
            messages=[{"role":"user","content":prompt}],
            temperature=0.2,
            max_tokens=600
        )
        state["draft"] = resp.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Feedback incorporation error: {e}")
        state["draft"] += f"\n\n[Feedback Not Incorporated: {feedback}]"
    return state
