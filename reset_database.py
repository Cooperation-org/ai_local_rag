import os
from dotenv import load_dotenv
import populate_database

# Load Config Settings
load_dotenv()  # take environment variables from .env.
CHROMA_PATH = chroma_path = os.getenv("CHROMA_PATH")

populate_database.clear_database()
print(f"Removed all content from database {CHROMA_PATH}")

