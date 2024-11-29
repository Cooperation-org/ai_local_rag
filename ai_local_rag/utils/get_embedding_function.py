import os

import dotenv

import numpy as np

from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
import nomic
from nomic import embed
# from langchain_community.embeddings.bedrock import BedrockEmbeddings
# from langchain_community.embeddings import FastEmbedEmbeddings
dotenv.load_dotenv()


def get_embedding_function_for_slack():
    # embeddings = BedrockEmbeddings(
    #    credentials_profile_name="default", region_name="us-east-1"
    # )

    # Make sure you run this first:  ollama pull nomic-embed-text
    # ollama_model = OllamaModel("nomic-embed-text")
    # embeddings = CustomOllamaEmbedding("nomic-embed-text-v1")
    embeddings = CustomSentenceTransformerEmbedding('paraphrase-MiniLM-L6-v2')

    # embeddings = embedding_functions.DefaultEmbeddingFunction()

    # embeddings = embedding_functions.SentenceTransformerEmbeddingFunction(
    #    model_name="all-MiniLM-L6-v2")

    # embeddings = FastEmbedEmbeddings()
    return embeddings


def get_embedding_function_for_pdf():
    # embeddings = BedrockEmbeddings(
    #    credentials_profile_name="default", region_name="us-east-1"
    # )

    # Make sure you run this first:  ollama pull nomic-embed-text
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # embeddings = embedding_functions.DefaultEmbeddingFunction()
    # embeddings = embedding_functions.SentenceTransformerEmbeddingFunction(
    #    model_name="all-MiniLM-L6-v2")
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


class CustomOllamaEmbedding:
    def __init__(self, model_name):
        self.model_name = model_name

    def __call__(self, input):
        nomic_api_key = os.getenv("NOMIC_API_KEY")
        nomic.login(nomic_api_key)
        if isinstance(input, str):
            input = [input]
        # Assuming nomic library provides a function to embed text
        embeddings = embed.text(input, model=self.model_name)
        return embeddings


class CustomSentenceTransformerEmbedding:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)

    def __call__(self, input):
        if isinstance(input, str):
            input = [input]
        # Generate embeddings
        embeddings = self.model.encode(input)
        # Ensure embeddings are in list format if using NumPy arrays
        if isinstance(embeddings, np.ndarray):
            embeddings = embeddings.tolist()
        return embeddings
