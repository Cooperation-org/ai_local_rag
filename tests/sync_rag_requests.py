import time

import requests

CHATBOT_API = "http://localhost:8000/rag-query"
POST_TEST = "http://localhost:8000/post_message"

questions = [
    "How do I get out of jail in Monopoly"
]

request_bodies = [{"query": q} for q in questions]

start_time = time.perf_counter()
# outputs = [requests.post(CHATBOT_API, json=data) for data in request_bodies]
# print(f"RAG Client Response:  {outputs}")
end_time = time.perf_counter()

print(f"Run time: {end_time - start_time} seconds")

outputs = [requests.post(CHATBOT_API, json=data) for data in request_bodies]
