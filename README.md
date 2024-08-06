# Simple Local RAG

A basic local LLM RAG Chatbot With LangChain that exposes itself via REST endpoints.

## Setup

Clone this repository and create a clean python v3.10 virtual environment and activate it.

#### Dependencies

The following is a high-level list of components used to run this local RAG:

- langchain
- ollama
- pypdf
- chromadb
- fastembed

```
pip install -r requirements.txt
```

#### Setup up Ollama

This depends on the Ollama platform to run the LLM locally. The setup is straightforward. First, visit ollama.ai and download the app appropriate for your operating system.

Next open your terminal and execute the following command to pull the latest Mistral model.

```
ollama pull llama3
```

#### Configuration

Create a `.env` file in the root directory and add the following environment variables:

```.env

CHROMA_DB_PATH_PDF=chroma_boardgames
DATA_PATH_BG=data_boardgames
```

#### Build the Vector Store

The `populate_database.py` loads any PDF files it finds in the `DATA_PATH_BG` folder. The repository currently includes a couple of example board game instruction manuals to seed a Chroma Vector store. The module reads the folder and loads each of the PDF's it into vector storage in two steps: first, it splits the document into smaller chunks to accommodate the token limit of the LLM; second, it vectorizes these chunks using FastEmbeddings and stores them into Chroma. It will generate a chunk ID that will indicate which PDF file, page number and chunk number of the embedding. This allows us to analyze how the model is producing a response, but also allows us to incrementally add new data to the database without have to fully reload it. Run the database load:

` python -m populate_database.py`

If you need to clear the database for any reason, run:

`python -m reset_database.py`

The above command will remove the chroma database. If you need to recreate it, simply rerun `populate_database.py`

## Running the RAG from the commandline:

The instruction manuals for both Monopoly and Ticket To Ride have been loaded into the Chroma DB. Ask the RAG questions about these two board games and see how well it does answering your questions. The RAG can be invoked using the following command with the sample question:

```
python query_data.py  How do I build a hotel in monopoly?
```

Here are some additional questions you can try:

- How much total money does a player start with in Monopoly? (Answer with the number only)
- How many points does the longest continuous train get in Ticket to Ride? (Answer with the number only)

You can also browse the instruction manuals that are in the `./data_boardgames` folder to come up with your own questions.

## Running the FASTAPI server to expose the RAG via API

Start the FASTPI server to expose api's that can be called from the ai_rag_ui or curl.

```
python query_data.py  How do I build a hotel in monopoly?
```

## Running the test cases

```
pytest test_rag.py
```
