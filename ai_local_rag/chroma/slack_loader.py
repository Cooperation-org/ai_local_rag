"""
Loader for slack
"""
import os

import dotenv
import logging
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.schema.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.agent_toolkits import SlackToolkit
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import SlackDirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.ollama import OllamaEmbeddings
from ai_local_rag.utils.get_embedding_function import get_embedding_function, CustomOpenAIEmbeddings, CustomOllamaEmbeddings

dotenv.load_dotenv()
logging.basicConfig()
logger = logging.getLogger("slack_loader")
logger.setLevel(logging.DEBUG)


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

    response["chunks"] = page_content
    response["chunk_ids"] = chunk_id_list
    response["metadata"] = metadata_list

    # return chunks
    return response


def _add_to_chroma_with_langchain(chunks: list[Document]):
    chroma_path = os.getenv("CHROMA_PATH_SLACK")
    chroma_collection = os.getenv("CHROMA_SLACK_COLLECTION")
    logger.info(f"_add_to_chroma - collection name:  {chroma_collection}")
    # Load the existing database.
    db = Chroma(collection_name=chroma_collection,
                persist_directory=chroma_path, embedding_function=get_embedding_function()
                )

    db.persist()


def _add_to_chroma(chunks_with_ids: list[Document]):
    chroma_path = os.getenv("CHROMA_PATH_SLACK")
    chroma_collection = os.getenv("CHROMA_SLACK_COLLECTION")

    chroma_client = chromadb.PersistentClient(path=chroma_path)
    # settings=Settings(chroma_db_impl="duckdb+parquet"))

    logger.info(f"_add_to_chroma - collection name:  {chroma_collection}")

    chroma_client = chromadb.Client()
    embedding_function = get_embedding_function()
    collection = chroma_client.get_or_create_collection(
        name=chroma_collection, embedding_function=embedding_function
    )

    # Calculate Page IDs.
    # chunks_with_ids = _calculate_chunk_ids(chunks)

    # collection = chroma_client.create_collection(name=chroma_collection)

    collection.add(ids=chunks_with_ids["chunk_ids"],
                   metadatas=chunks_with_ids["metadata"],
                   documents=chunks_with_ids["chunks"])

    results = collection.query(
        # Chroma will embed this for you
        query_texts=["This is a query document about c-linked-trust"],
        n_results=2  # how many results to return
    )

    logger.info("QUERY RESULTS")
    logger.info(results)


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
    logger.info("FINISHED")


if __name__ == "__main__":
    main()
