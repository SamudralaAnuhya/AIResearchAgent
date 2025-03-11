# agent_state.py

from typing import TypedDict, Optional, List

class AgentState(TypedDict):
    user_query: str
    uploaded_file_content: Optional[str]
    restructured_query: Optional[str]
    hypothetical_document: Optional[str]
    retrieved_context: Optional[str]
    references: List[str]
    draft: Optional[str]
    confidence_score: float
    requires_feedback: bool
    feedback: Optional[str]
    final_response: Optional[str]
