from pydantic import BaseModel


class QueryInput(BaseModel):
    query: str


class QueryOutput(BaseModel):
    query: str
    response: str
    sources: list[str]
