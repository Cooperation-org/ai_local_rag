import argparse
import logging
import os

import chromadb
import numpy as np
from dotenv import load_dotenv

from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama

from ai_local_rag.utils.get_embedding_function import \
    get_embedding_function_for_slack

# Load Config Settings
load_dotenv()  # take environment variables from .env.


logging.basicConfig()
logger = logging.getLogger("query_slack_data")
logger.setLevel(logging.DEBUG)

chroma_path = os.getenv("CHROMA_DB_PATH_SLACK")
chroma_collection = os.getenv("CHROMA_SLACK_COLLECTION")

verbose_str = os.getenv("VERBOSE").lower()
VERBOSE = False
if verbose_str == "true":
    VERBOSE = True

PROMPT_TEMPLATE = """
The following is a relevant document based on your query about "{query}":

Document ID: {doc_id}
Similarity Score: {score}
Document Text:
{doc_text}

How can I assist you further with this information?
"""


def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_slack_rag(query_text)


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


def query_slack_rag(query_text: str):
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
            n_results=1,  # Number of results to retrieve
            include=query_include
        )
        result_len = len(results.get('ids', []))
        logger.info(f"Query returned {result_len} item(s)\n")

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

        ##### Query Complete ####

        # Extract the first result (most similar)
        # Extract the first result (most similar)
        result_id = results.get('ids', [None])[0]
        result_metadata = results.get('metadatas', [None])[0]
        result_text = results.get('documents', [None])[0]
        result_distance = results.get('distances', [None])[0]

        logger.debug(f"Most similar result ID: {result_id}")
        logger.debug(f"Distance (similarity score): {result_distance}\n")
        logger.debug(f"Document text: {result_text}")
        logger.debug(f"Metadata: {result_metadata}\n")

        ##### Build the Prompt ####
        chat_prompt_template = ChatPromptTemplate.from_template(
            PROMPT_TEMPLATE)
        # Fill the template with actual data from the query result
        filled_prompt = []
        filled_prompt.append(chat_prompt_template.format(
            query=query_text[0],
            doc_id=result_id[0],
            score=round(result_distance[0],
                        4) if result_distance is not None else "N/A",
            doc_text=result_text[0] or "No document found."
        ))

        logger.debug(f"Formatted Chat Prompt:  {filled_prompt}\n")

        # Assume you have a language model set up (like an OpenAI model)
        language_model = Ollama(model="mistral")
        llm_result = language_model.generate(filled_prompt)
        # max_tokens=50,
        # temperature=0.7,
        # num_return_sequences=3
        # logger.debug(f"\nLanguage Model's 'Generate' Response:")
        # logger.debug(llm_result)

        # The 'invoke' function returns a much more simple response than 'generate'
        # response2 = language_model.invoke(filled_prompt)
        # logger.debug(f"\nLanguage Model's 'Invoke' Response:")
        # logger.debug(response2)

        # Extracting elements from the LLMResult object

        # 1. Extract the generated texts
        generated_texts = [
            generation.text for generation_list in llm_result.generations for generation in generation_list]
        logger.debug(f"Generated Texts:")
        for text in generated_texts:
            logger.debug(f"- {text}")

        # 2. Extract token usage information (if available)
        if llm_result.llm_output and "token_usage" in llm_result.llm_output:
            token_usage = llm_result.llm_output["token_usage"]
            total_tokens = token_usage.get("total_tokens", 0)
            prompt_tokens = token_usage.get("prompt_tokens", 0)
            completion_tokens = token_usage.get("completion_tokens", 0)

            logger.debug(f"Token Usage:")
            logger.debug(f"- Total Tokens: {total_tokens}")
            logger.debug(f"- Prompt Tokens: {prompt_tokens}")
            logger.debug(f"- Completion Tokens: {completion_tokens}\n")

        # 3. Extract the model name (if available)
        # model_name = llm_result.llm_output.get("model_name", "Unknown Model")
        # print(f"\nModel Name: {model_name}")

        # 4. Optionally, handle run_info (if it's included in your version of LangChain)
        # This would typically be done if the LLMResult included any runtime info you want to log or analyze.
        # Example:
        run_info = llm_result.run
        logger.debug(f"Run Info: {run_info}\n")
        # if run_info:
        #    print("\nRun Info:")
        #    for key, value in run_info.items():
        #        print(f"{key}: {value}")

        # Format the response message
        response_message = {
            "query": query_text, "response": generated_texts, "sources": result_metadata[0]['source'], "channel": result_metadata[0]['channel']}

        logger.debug(f"Response Message:  {response_message}\n")
        return response_message
    except Exception as e:
        print(f"Error querying the collection: {e}")


if __name__ == "__main__":
    main()
