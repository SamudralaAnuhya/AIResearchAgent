# prompts.py

"""
Prompt-building functions for the AI-Driven Research Assistant.
Each function returns a string that can be fed to the LLM.
"""

def no_context_prompt(query: str) -> str:
    return f"""You are an AI-driven scientific research assistant. 
No relevant content was found for this query:

Query: {query}

Give a concise overview, mention the limitation, and encourage further docs.
"""

def draft_prompt(query: str, context: str) -> str:
    return f"""You are an AI research assistant. Use the relevant text below to answer:

Query: {query}

Retrieved Text:
\"\"\"{context}\"\"\"

Draft a concise summary referencing snippet labels if needed.
End with a short academic disclaimer in the final sentence.
"""

def verify_prompt(query: str, draft: str, context: str) -> str:
    return f"""You are a senior research reviewer. 
Refine the draft for correctness and conciseness, no meta commentary. 
End with one-sentence disclaimer if missing.

Query: {query}
Draft:
\"\"\"{draft}\"\"\"
Context:
\"\"\"{context}\"\"\"

Final response (concise):
"""

def feedback_prompt(query: str, draft: str, feedback: str, context: str) -> str:
    return f"""You are an AI assistant. 
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

