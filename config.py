# config.py
import os
import logging

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model names
DRAFT_MODEL = "gemma2-9b-it"
MAIN_MODEL = "llama3-70b-8192"

# Other constants or environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
