import argparse
import logging
import os

import chromadb
import numpy as np
from dotenv import load_dotenv

from ai_local_rag.utils.get_embedding_function import \
    get_embedding_function_for_slack

# Load Config Settings
load_dotenv()  # take environment variables from .env.


logging.basicConfig()
logger = logging.getLogger("slack_loader")
logger.setLevel(logging.DEBUG)

chroma_path = os.getenv("CHROMA_DB_PATH_SLACK")
chroma_collection = os.getenv("CHROMA_SLACK_COLLECTION")

verbose_str = os.getenv("VERBOSE").lower()
VERBOSE = False
if verbose_str == "true":
    VERBOSE = True

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)


def _get_collection():
    # Create or load a collection
    collection_name = chroma_collection
    collection = None
    chroma_client = chromadb.PersistentClient(path=chroma_path)
    try:
        collection = chroma_client.get_collection(collection_name)

        return collection
    except Exception as e:
        print(f"Error accessing collection: {e}")


def query_rag(query_text: str):
    # Load the collection
    collection = _get_collection()

    # QUERY WITH METADATA
    embedding_function = get_embedding_function_for_slack()
    query_embedding = embedding_function(query_text)
    query_include = ["metadatas", "documents", "distances", "embeddings"]

    # Ensure query_embedding is in the expected format
    if isinstance(query_embedding, np.ndarray):
        query_embedding = query_embedding.tolist()

    # Query the collection and retrieve results with the specified includes
    try:
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=5,  # Number of results to retrieve
            include=query_include
        )
        result_len = len(results.get('ids', []))
        logger.info(f"Query returned {result_len} item(s)")

        if VERBOSE:

            # Process and print results
            ids = results.get('ids', [])
            embeddings = results.get('embeddings', [])
            distances = results.get('distances', [])
            metadatas = results.get('metadatas', [])
            documents = results.get('documents', [])

            # Process and print results
            # for i in range(len(ids)):
            for i, result_id in enumerate(ids):
                logger.debug(f"Result {i + 1}:")
                logger.debug(f"ID: {result_id}")

                if distances:
                    logger.debug(
                        f"Distance (Similarity Score) {i + 1}: {distances[i]}\n")
                else:
                    logger.debug("Distance not available.")

                if documents:
                    logger.debug(f"Documents {i + 1}: {documents[i]}\n")
                else:
                    logger.debug("Distance not available.")

                if metadatas:
                    logger.debug(f"Metadata {i + 1}: {metadatas[i]}\n")
                else:
                    logger.debug("Metadata not available.")

                if embeddings:
                    # logger.debug(f"Embedding: {embeddings[i]}")
                    logger.debug(f"Retreived an embedding {i + 1}\n")
                else:
                    print("Embedding not available.")

            logger.debug("-" * 40)

        return results
    except Exception as e:
        print(f"Error querying the collection: {e}")


if __name__ == "__main__":
    main()
