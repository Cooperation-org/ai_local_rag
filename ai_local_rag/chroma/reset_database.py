import os
from dotenv import load_dotenv
import ai_local_rag.chroma.populate_database as populate_database

# Load Config Settings
load_dotenv()  # take environment variables from .env.
chroma_db_path_pdf = os.getenv("CHROMA_DB_PATH_PDF")

populate_database.clear_database()
print(f"Removed all content from database {chroma_db_path_pdf}")
