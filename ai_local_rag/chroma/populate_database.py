import argparse
import os
import shutil

from dotenv import load_dotenv
from langchain.schema.document import Document
# from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader
# from langchain.vectorstores.chroma import Chroma
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ai_local_rag.utils.get_embedding_function import get_embedding_function_for_pdf

# Load Config Settings
load_dotenv()  # take environment variables from .env.


def main():

    # Check if the database should be cleared (using the --clear flag).
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true",
                        help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    # Create (or update) the data store with board game information.
    data_path = os.getenv("DATA_PATH_BG")
    documents = _load_documents(data_path)
    chunks = _split_documents(documents)
    _add_to_chroma(chunks)

    # Create (or update) the data store with machine learning lecture information
    # data_path = os.getenv("DATA_PATH_ML")
    # documents = _load_documents(data_path)
    # chunks = _split_documents(documents)
    # _add_to_chroma(chunks)


def _load_documents(data_path):
    document_loader = PyPDFDirectoryLoader(data_path)
    return document_loader.load()


def _split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


def _add_to_chroma(chunks: list[Document]):
    chroma_db_path_pdf = os.getenv("CHROMA_DB_PATH_PDF")
    # Load the existing database.
    db = Chroma(
        persist_directory=chroma_db_path_pdf, embedding_function=get_embedding_function_for_pdf()
    )

    # Calculate Page IDs.
    chunks_with_ids = _calculate_chunk_ids(chunks)

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        db.persist()
    else:
        print("✅ No new documents to add")


def _calculate_chunk_ids(chunks):

    # This will create IDs like "data/monopoly.pdf:6:2"
    # Page Source : Page Number : Chunk Index

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks


def clear_database():
    """ 
    Remove the database so that we can rebuild it
    """
    load_dotenv()  # take environment variables from .env.
    chroma_db_path = os.getenv("CHROMA_DB_PATH_PDF")

    if os.path.exists(chroma_db_path):
        shutil.rmtree(chroma_db_path)


if __name__ == "__main__":
    main()
