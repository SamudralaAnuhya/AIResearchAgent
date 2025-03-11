# app.py
import streamlit as st
from agent_state import AgentState
from nodes import refine_with_feedback
from file_processing import extract_text 
from workflow import create_workflow

def main():
    st.title("AI-Driven Scientific Research Assistant")
    st.markdown("This is a modularized Streamlit app for research-based RAG.")

    with st.sidebar:
        st.header("Upload Document")
        uploaded_file = st.file_uploader("Upload PDF, DOCX, TXT, or image", 
                                         type=["pdf","docx","txt","png","jpg","jpeg"])
        if uploaded_file and "uploaded_content" not in st.session_state:
            with st.spinner("Extracting..."):
                file_text = extract_text(uploaded_file)
                if file_text:
                    st.session_state["uploaded_content"] = file_text
                    st.success("Document loaded!")
                else:
                    st.error("Cannot extract any text.")
        
        if "uploaded_content" in st.session_state:
            st.success("Document ready!")
            if st.button("Clear Document"):
                del st.session_state["uploaded_content"]
                st.rerun()

    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.chat_message("user").write(msg["content"])
        else:
            st.chat_message("assistant").markdown(msg["content"])

    # Check feedback loop
    if "awaiting_feedback" in st.session_state and st.session_state.awaiting_feedback:
        st.info("Please review the draft and provide feedback or approve.")
        conf = st.session_state.current_state["confidence_score"]
        st.write(f"Confidence: {conf:.2f}")

        st.markdown(f"**Draft:**\n\n{st.session_state.current_draft}")

        fb = st.text_area("Feedback", key="feedback_input", height=100)
        colA, colB = st.columns(2)
        with colA:
            if st.button("Approve"):
                stored = st.session_state.current_state
                from nodes import verify_response
                verified = verify_response(stored)
                final_ans = verified["final_response"]
                st.session_state.messages.append({"role":"assistant","content":final_ans})
                del st.session_state.awaiting_feedback
                del st.session_state.current_draft
                del st.session_state.current_state
                st.rerun()
        with colB:
            if st.button("Submit Feedback"):
                if fb.strip():
                    stored = st.session_state.current_state
                    refined = refine_with_feedback(stored, fb.strip())
                    from nodes import verify_response
                    verified = verify_response(refined)
                    final_ans = verified["final_response"]
                    st.session_state.messages.append({"role":"assistant","content":final_ans})
                    del st.session_state.awaiting_feedback
                    del st.session_state.current_draft
                    del st.session_state.current_state
                    st.rerun()

    else:
        # Normal input
        user_query = st.chat_input("Ask your research question...")
        if user_query:
            if "uploaded_content" not in st.session_state:
                st.warning("Please upload a document first.")
            else:
                st.session_state.messages.append({"role":"user","content":user_query})
                st.chat_message("user").write(user_query)

                with st.chat_message("assistant"):
                    with st.spinner("Processing..."):
                        # Build initial state
                        state: AgentState = {
                            "user_query": user_query,
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

                        wf = create_workflow()
                        res = wf.invoke(state)
                        if res["requires_feedback"]:
                            st.session_state.awaiting_feedback = True
                            st.session_state.current_draft = res["draft"]
                            st.session_state.current_state = res
                            st.rerun()
                        else:
                            st.markdown(res["final_response"])
                            st.session_state.messages.append({"role":"assistant","content":res["final_response"]})

if __name__ == "__main__":
    main()
