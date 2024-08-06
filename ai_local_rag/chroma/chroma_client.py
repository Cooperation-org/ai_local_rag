from chromadb.config import Settings
import chromadb
import logging
import os

import dotenv
from langchain.vectorstores import Chroma

# from ai_local_rag.utils.get_embedding_function import get_embedding_function


dotenv.load_dotenv()


logging.basicConfig()
logger = logging.getLogger("chroma_client")
logger.setLevel(logging.DEBUG)


chroma_path = os.getenv("CHROMA_PATH_SLACK")
collection_name = os.getenv("CHROMA_SLACK_COLLECTION")


client = chromadb.Client(Settings(
    persist_directory=chroma_path
))

logger.info(f"Number of collections:  {client.count_collections()}")

logger.info(client.list_collections())
"""
collection = client.get_collection(name=collection_name)
result = collection.query(n_results=5)
logging.info(result)
logging.info(len(client.list_collections()))
"""
