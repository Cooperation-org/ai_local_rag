"""
Loader for slack
"""
import os
import logging
import dotenv

import numpy as np
from dotenv import load_dotenv
import chromadb
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.schema.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.agent_toolkits import SlackToolkit
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import SlackDirectoryLoader
from langchain_community.vectorstores import Chroma
from ai_local_rag.utils.get_embedding_function import get_embedding_function_for_slack

dotenv.load_dotenv()
logging.basicConfig()
logger = logging.getLogger("slack_loader")
logger.setLevel(logging.DEBUG)

chroma_path = os.getenv("CHROMA_DB_PATH_SLACK")
chroma_collection = os.getenv("CHROMA_SLACK_COLLECTION")


def slack_toolkit():
    toolkit = SlackToolkit()
    my_tools = toolkit.get_tools()

    # llm = ChatOpenAI(temperature=0, model="gpt-4")
    llm = ChatOllama(model="mistral")

    prompt = hub.pull("hwchase17/openai-tools-agent")
    agent = create_openai_tools_agent(
        tools=my_tools,
        llm=llm,
        prompt=prompt,
    )
    agent_executor = AgentExecutor(agent=agent, tools=my_tools, verbose=True)
    agent_executor.invoke(
        {
            "input": "Send a greeting to my coworkers in the #general channel. Note use `channel` as key of channel id, and `message` as key of content to sent in the channel."
        }
    )
    agent_executor.invoke(
        {"input": "How many channels are in the workspace? Please list out their names."}
    )
    agent_executor.invoke(
        {
            "input": "Tell me the number of messages sent in the #introductions channel from the past month."
        }
    )


def _slack_loader():
    local_zip_file = os.getenv("SLACK_EXPORT_ZIP")
    slack_workspace_url = os.getenv("SLACK_WORKSPACE_URL")

    loader = SlackDirectoryLoader(local_zip_file, slack_workspace_url)
    docs = loader.load()
    logger.info(f"Slack export contains {len(docs)} docs")
    return docs


def _split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


def _print_chunks(chunks):
    i = 0
    for chunk in chunks:
        logger.info(f"chunk {i} contains: {chunk}\n")
        i += 1
        if i > 5:
            break


def _calculate_chunk_ids(chunks: list[Document]):

    # This will create IDs like "c-linkedtrust:U069UCY6WPL:1721902091.115329:2"
    # Channel : UserId: Timestamp: Chunk Index

    # Document format is:[Document(metadata={'soource': 'xxxx'})]

    last_page_id = None
    current_chunk_index = 0

    response = {}
    chunk_id_list = []
    metadata_list = []
    page_content = []

    for chunk in chunks:
        user = chunk.metadata.get("user")
        channel = chunk.metadata.get("channel")
        timestamp = chunk.metadata.get("timestamp")
        current_page_id = f"{channel}:{user}:{timestamp}"

        page_content.append(chunk.page_content)

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add the id to the page meta-data.
        chunk.metadata["id"] = chunk_id

        # Add the metadata to the metadata list
        metadata_list.append(chunk.metadata)

        # Add it to the list of ids
        chunk_id_list.append(chunk_id)

    response["texts"] = page_content
    response["ids"] = chunk_id_list
    response["metadatas"] = metadata_list

    # return chunks
    return response


def _add_to_chroma_with_langchain(chunks: list[Document]):

    logger.info(f"_add_to_chroma - collection name:  {chroma_collection}")
    # Load the existing database.
    db = Chroma(collection_name=chroma_collection,
                persist_directory=chroma_path, embedding_function=get_embedding_function_for_slack()
                )

    db.persist()


def _add_to_chroma(chunks_with_ids: list[Document]):

    chroma_client = chromadb.PersistentClient(path=chroma_path)
    # settings=Settings(chroma_db_impl="duckdb+parquet"))

    logger.info(f"_add_to_chroma - collection name:  {chroma_collection}")

    # Create or load a collection
    collection_name = chroma_collection
    collection = None
    try:
        collection = chroma_client.get_or_create_collection(collection_name)
    except Exception as e:
        print(f"Error accessing collection: {e}")

    texts = chunks_with_ids['texts']
    metadatas = chunks_with_ids['metadatas']
    ids = chunks_with_ids['ids']

    # Get the embeddings function
    embedding_function = get_embedding_function_for_slack()

    # Generate the embeddings
    embeddings = embedding_function(texts)
    # print("Embeddings:  ", embeddings)

    # Debugging: Print lengths of all inputs
    logger.debug(f"Texts length: {len(texts)}\n")
    logger.debug(f"Embeddings length: {len(embeddings)}\n")
    logger.debug(f"Metadata length: {len(metadatas)}\n")
    logger.debug(f"IDs length: {len(ids)}\n")

    # Ensure all lists have the same length
    assert len(texts) == len(embeddings) == len(metadatas) == len(
        ids), "Lengths of input lists do not match."

    # Ensure we have an embedding for each ID
    for i in range(len(ids)):
        print(f"ID: {ids[i]}")
        print(f"Embedding: {embeddings[i]}")

    # Add text and embeddings to the collection
    try:
        collection.add(documents=texts, embeddings=embeddings,
                       metadatas=metadatas, ids=ids)

    except Exception as e:
        print(f"Error adding to collection: {e}")

    logger.info("DONE LOADING and QUERING")


def _query():
    collection = []

    # Create or load a collection
    collection_name = chroma_collection
    collection = None
    chroma_client = chromadb.PersistentClient(path=chroma_path)
    try:
        collection = chroma_client.get_collection(collection_name)
    except Exception as e:
        print(f"Error accessing collection: {e}")

     # QUERY WITH METADATA
    query_text = 'Great i have tasks on the front and in back so i will contact with both'
    embedding_function = get_embedding_function_for_slack()
    query_embedding = embedding_function(query_text)
    query_include = ["metadatas", "distances", "embeddings"]

    # Ensure query_embedding is in the expected format
    if isinstance(query_embedding, np.ndarray):
        query_embedding = query_embedding.tolist()

    # Query the collection
    try:
        result = collection.query(
            query_embedding, n_results=1, include=query_include)
        print("Query Result:  ", result)
    except Exception as e:
        logger.info(f"Error querying the collection: {e}")

    # Example: Query the collection and retrieve results with metadata
    try:
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=5,  # Number of results to retrieve
            include=query_include
        )

        # Process and print results
        ids = results.get('ids', [])
        embeddings = results.get('embeddings', [])
        distances = results.get('distances', [])
        metadatas = results.get('metadatas', [])

        # Process and print results
        # for i in range(len(ids)):
        for i, result_id in enumerate(ids):
            print(f"Result {i + 1}:")
            print(f"ID: {result_id}")

            if distances:
                print(f"Distance: {distances[i]}")
            else:
                print("Distance not available.")

            if metadatas:
                print(f"Metadata: {metadatas[i]}")
            else:
                print("Metadata not available.")

            if embeddings:
                print(f"Embedding: {embeddings[i]}")
            else:
                print("Embedding not available.")

        print("-" * 40)
    except Exception as e:
        print(f"Error querying the collection: {e}")


def main():
    # Load Config Settings
    logger.info("STARTING")
    load_dotenv()  # take environment variables from .env.
    documents = _slack_loader()
    chunks = _split_documents(documents)
    _print_chunks(chunks)
    chunks = _calculate_chunk_ids(chunks)
    _print_chunks(chunks)
    _add_to_chroma(chunks)
    _query()
    logger.info("FINISHED")


if __name__ == "__main__":
    main()
