import logging
from query_data import query_rag
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from models.rag_query_model import QueryInput, QueryOutput
from utils.async_utils import async_retry


class Message(BaseModel):
    """ Message class defined in Pydantic """
    channel: str
    author: str
    text: str


app = FastAPI(
    title="PDF Document Chatbot",
    description="Endpoints for various PDF documents",
)

channel_list = ["general", "dev", "marketing"]
message_map = {}


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    exc_str = f'{exc}'.replace('\n', ' ').replace('   ', ' ')
    logging.error(f"{request}: {exc_str}")
    content = {'status_code': 10422, 'message': exc_str, 'data': None}
    return JSONResponse(content=content, status_code=status.HTTP_422_UNPROCESSABLE_ENTITY)


@async_retry(max_retries=10, delay=1)
async def invoke_agent_with_retry(query: str):
    """
    Retry the agent if a tool fails to run. This can help when there
    are intermittent connection issues to external APIs.
    """

    return await query_rag({"input": query})


@app.get("/")
async def get_status():
    return {"status": "running"}


@app.post("/post_message", status_code=status.HTTP_201_CREATED)
def post_message(message: Message):
    """Post a new message to the specified channel."""
    channel = message.channel
    if channel in channel_list:
        # message_map[channel].append(message)
        return message
    else:
        raise HTTPException(status_code=404, detail="channel not found")


@app.post("/rag-query")
async def query_rag_api(query: QueryInput):
    print(f"api.py - API Request Data: {query}")
    query_response = query_rag({"input": query})
    print(query_response)

    # query: str
    # response: str
    # sources: list[str]
    query_response2 = {
        "query": query, "response": query_response["response"], "sources": query_response["sources"]}
    print(f"Query Response2:  {query_response2}")

    # query_response["intermediate_steps"] = [
    #    str(s) for s in query_response["intermediate_steps"]
    # ]

    return query_response2


@app.post("/rag-query2")
async def query_rag_api2(
    query: QueryInput,
) -> QueryOutput:
    query_response = query_rag({"input": query})
    print(query_response)

    # query: str
    # response: str
    # sources: list[str]
    query_text = query["query"]
    query_response2 = {
        "query": query_text, "response": query_response["response"], "sources": query_response["sources"]}
    print(f"Query Response2:  {query_response2}")

    # query_response["intermediate_steps"] = [
    #    str(s) for s in query_response["intermediate_steps"]
    # ]

    return query_response2
