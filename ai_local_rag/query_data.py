import argparse
import os
from dotenv import load_dotenv
# from langchain.vectorstores.chroma import Chroma
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama

from ai_local_rag.utils.get_embedding_function import get_embedding_function_for_pdf

# Load Config Settings
load_dotenv()  # take environment variables from .env.
chroma_db_path_pdf = os.getenv("CHROMA_DB_PATH_PDF")

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


def query_rag(query_text: str):
    # Prepare the DB.
    embedding_function = get_embedding_function_for_pdf()
    db = Chroma(persist_directory=chroma_db_path_pdf,
                embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join(
        [doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    # print(prompt)

    model = Ollama(model="mistral")
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)

    # Format the response
    response_message = {"query": query_text,
                        "response": response_text, "sources": sources}
    # print(f"qyery_data:query_rag: Response Message:  {response_message}")
    return response_message


if __name__ == "__main__":
    main()
