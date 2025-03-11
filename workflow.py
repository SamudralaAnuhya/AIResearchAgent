# workflow.py
from typing import TypedDict
from langgraph.graph import StateGraph, END
from agent_state import AgentState
from nodes import (
    query_refinement , refine_with_feedback ,
    generate_draft_response,human_in_the_loop ,
    verify_response ,retrieve_context,
    process_uploaded_file)

def create_workflow():
    """
    Builds a graph-based workflow:
      1) process file
      2) refine query
      3) retrieve context
      4) draft
      5) if confidence < threshold => human_in_the_loop => final verify
         else => final verify
    """
    workflow = StateGraph(AgentState)

    workflow.add_node("process_uploaded_file", process_uploaded_file)
    workflow.add_node("query_refinement", query_refinement)
    workflow.add_node("retrieve_context", retrieve_context)
    workflow.add_node("generate_draft_response", generate_draft_response)
    workflow.add_node("human_in_the_loop", human_in_the_loop)
    workflow.add_node("verify_response", verify_response)

    workflow.set_entry_point("process_uploaded_file")
    workflow.add_edge("process_uploaded_file", "query_refinement")
    workflow.add_edge("query_refinement", "retrieve_context")
    workflow.add_edge("retrieve_context", "generate_draft_response")

    workflow.add_conditional_edges(
        "generate_draft_response",
        lambda s: "human_in_the_loop" if s["confidence_score"] < 0.9 else "verify_response"
    )
    workflow.add_edge("human_in_the_loop", "verify_response")
    workflow.add_edge("verify_response", END)

    return workflow.compile()
