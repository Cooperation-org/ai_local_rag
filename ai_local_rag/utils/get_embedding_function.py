from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from chromadb.utils import embedding_functions
# from langchain_community.embeddings.bedrock import BedrockEmbeddings
# from langchain_community.embeddings import FastEmbedEmbeddings


def get_embedding_function():
    # embeddings = BedrockEmbeddings(
    #    credentials_profile_name="default", region_name="us-east-1"
    # )

    # Make sure you run this first:  ollama pull nomic-embed-text
    # embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # embeddings = embedding_functions.DefaultEmbeddingFunction()
    embeddings = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2")
    # embeddings = FastEmbedEmbeddings()
    return embeddings


class CustomOpenAIEmbeddings(OpenAIEmbeddings):

    def __init__(self, openai_api_key, *args, **kwargs):
        super().__init__(openai_api_key=openai_api_key, *args, **kwargs)

    def _embed_documents(self, texts):
        embeddings = [
            self.client.create(
                input=text, model="text-embedding-ada-002").data[0].embedding
            for text in texts
        ]
        return embeddings

    def __call__(self, input):
        return self._embed_documents(input)


class CustomOllamaEmbeddings(OllamaEmbeddings):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _embed_documents(self, texts):
        embeddings = [
            self.
            self.client.create(
                input=text, model="nomic-embed-text").data[0].embedding
            for text in texts
        ]
        return embeddings

    def __call__(self, input):
        return self._embed_documents(input)
