# file_processing.py

import logging
import streamlit as st
import PyPDF2
import docx
import pytesseract
from PIL import Image
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

embedding_model = None  # We'll load this once
vector_db = None        # In-memory reference to the Vector DB

def load_embeddings_if_needed():
    global embedding_model
    if embedding_model is None:
        try:
            embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            logger.info("Embedding model loaded.")
        except Exception as e:
            st.error(f"Failed to load embeddings: {e}")
            embedding_model = None

def extract_text(file) -> str:
    """ Extract text from PDF, DOCX, TXT, or image. """
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
    Splits the content into semantic chunks, then creates a FAISS Vector DB.
    Stores snippet label and snippet preview in each chunk's metadata.
    """
    global embedding_model, vector_db
    load_embeddings_if_needed()
    if not content.strip() or embedding_model is None:
        logger.warning("No content or no embedding model. Vector DB not created.")
        return
    
    chunker = SemanticChunker(
        embedding_model,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=90
    )

    docs = chunker.create_documents([content])
    for idx, doc in enumerate(docs):
        lbl = f"Snippet {idx+1}"
        prv = doc.page_content[:60].replace("\n"," ")
        doc.metadata["snippet_label"] = lbl
        doc.metadata["snippet_preview"] = prv

    vector_db = FAISS.from_documents(docs, embedding_model)
    logger.info(f"Vector DB created with {len(docs)} chunks.")

def get_vector_db():
    """ Utility to return the global vector_db reference. """
    global vector_db
    return vector_db
