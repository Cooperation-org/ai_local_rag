from langchain_community.embeddings.ollama import OllamaEmbeddings
# from langchain_community.embeddings.bedrock import BedrockEmbeddings
# from langchain_community.embeddings import FastEmbedEmbeddings


def get_embedding_function():
    # embeddings = BedrockEmbeddings(
    #    credentials_profile_name="default", region_name="us-east-1"
    # )

    # Make sure you run this first:  ollama pull nomic-embed-text
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    # embeddings = FastEmbedEmbeddings()
    return embeddings
