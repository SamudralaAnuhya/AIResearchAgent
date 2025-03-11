# nodes.py

import logging
import streamlit as st
from groq import Groq
from config import GROQ_API_KEY, DRAFT_MODEL, MAIN_MODEL
from agent_state import AgentState
from file_processing import extract_text, setup_vector_db, get_vector_db, load_embeddings_if_needed

logger = logging.getLogger(__name__)

client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

def process_uploaded_file(state: AgentState) -> AgentState:
    """ Load embeddings, set up vector DB with the uploaded text. """
    load_embeddings_if_needed()
    txt = state.get("uploaded_file_content","")
    if txt:
        setup_vector_db(txt)
    return state

def query_refinement(state: AgentState) -> AgentState:
    q = state["user_query"].strip().lower()
    refined = state["user_query"].strip()
    if "what is" in q:
        refined = f"Provide a detailed explanation of {q.replace('what is','').strip()} from recent AI research."
    elif q.startswith("how does"):
        refined = f"Explain the mechanisms behind {q.replace('how does','').strip()} based on recent studies."
    elif "state of the art" in q or "sota" in q:
        refined = f"What are the latest state-of-the-art developments in {q.replace('state of the art','').replace('sota','').strip()}?"
    elif "compare" in q:
        refined = f"Compare and contrast the approaches in {q.replace('compare','').strip()} with advantages and limitations."
    else:
        if not (refined.endswith(".") or refined.endswith("?") or refined.endswith("!")):
            refined += "?"
    state["restructured_query"] = refined
    return state

def retrieve_context(state: AgentState) -> AgentState:
    """ Retrieves relevant chunks from the vector DB using the refined query. """
    vdb = get_vector_db()
    if not vdb:
        state["retrieved_context"] = ""
        state["references"] = []
        return state

    query_text = state.get("restructured_query") or state["user_query"]
    try:
        results = vdb.similarity_search(query_text, k=3)
        combined_context = []
        refs = []
        for doc in results:
            lbl = doc.metadata.get("snippet_label","Snippet?")
            prv = doc.metadata.get("snippet_preview","")
            text = doc.page_content
            combined_context.append(f"{lbl} [preview: {prv}]\n{text}")
            refs.append(f"[{lbl} | {prv}]")
        state["retrieved_context"] = "\n\n".join(combined_context)
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
    
    query = state.get("restructured_query") or state["user_query"]
    context = state.get("retrieved_context","")
    if not context.strip():
        # No context scenario
        prompt = f"""You are an AI-driven research assistant. 
No relevant information was found in the document for:

Query: {query}

Provide a short overview, note the limitation, and encourage more docs."""
        state["confidence_score"] = 0.4
    else:
        prompt = f"""You are an AI research assistant. Use the following context to answer:

Query: {query}
Context:
\"\"\"{context}\"\"\"

Draft a concise summary referencing snippet labels if needed.
End with a short academic disclaimer in the final sentence.
"""
        # Confidence depends on how large the context is
        state["confidence_score"] = 0.7 if len(context) > 500 else 0.6

    try:
        resp = client.chat.completions.create(
            model=DRAFT_MODEL,
            messages=[{"role":"user","content":prompt}],
            temperature=0.3,
            max_tokens=500
        )
        draft = resp.choices[0].message.content.strip()
        state["draft"] = draft
    except Exception as e:
        logger.error(f"Draft gen error: {e}")
        state["draft"] = "Error generating draft."
        state["confidence_score"] = 0.3

    return state

def human_in_the_loop(state: AgentState) -> AgentState:
    state["requires_feedback"] = True
    return state

def verify_response(state: AgentState) -> AgentState:
    """ Final pass with a more advanced model, ensuring correctness. """
    if client is None:
        state["final_response"] = state["draft"]
        return state

    draft = state["draft"]
    query = state.get("restructured_query") or state["user_query"]
    context = state.get("retrieved_context","")

    prompt = f"""You are a senior research reviewer. 
Refine the draft for clarity and correctness, no mention of changes. 
End with one-sentence disclaimer if missing.

Query: {query}
Draft:
\"\"\"{draft}\"\"\"
Context:
\"\"\"{context}\"\"\"

Final Answer:
"""
    try:
        resp = client.chat.completions.create(
            model=MAIN_MODEL,
            messages=[{"role":"user","content":prompt}],
            temperature=0.2,
            max_tokens=500
        )
        final_answer = resp.choices[0].message.content.strip()
        state["final_response"] = final_answer
    except Exception as e:
        logger.error(f"Verification error: {e}")
        state["final_response"] = draft

    return state

def refine_with_feedback(state: AgentState, feedback: str) -> AgentState:
    """ Incorporate user feedback into the draft. """
    if client is None:
        state["draft"] += f"\n\n[Feedback not processed: {feedback}]"
        return state

    draft = state["draft"]
    query = state.get("restructured_query") or state["user_query"]
    context = state.get("retrieved_context","")

    prompt = f"""You are an AI assistant. 
Refine the draft based on this feedback, no mention of changes:

Query: {query}
Draft:
\"\"\"{draft}\"\"\"
Feedback:
{feedback}
Context:
\"\"\"{context}\"\"\"

Revised Draft:
"""
    try:
        resp = client.chat.completions.create(
            model=MAIN_MODEL,
            messages=[{"role":"user","content":prompt}],
            temperature=0.2,
            max_tokens=500
        )
        refined = resp.choices[0].message.content.strip()
        state["draft"] = refined
    except Exception as e:
        logger.error(f"Feedback incorporation error: {e}")
        state["draft"] += f"\n\n[Feedback Not Incorporated: {feedback}]"

    return state
