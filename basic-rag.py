from llama_index import SimpleDirectoryReader
from llama_index import Document
from llama_index import VectorStoreIndex
from llama_index import ServiceContext
from llama_index.llms import OpenAI
import os
import openai



openai.api_key = ""

documents = SimpleDirectoryReader(
    input_files=["data/order-management.pdf"]
).load_data()

document = Document(text="\n\n".join([doc.text for doc in documents]))

llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
service_context = ServiceContext.from_defaults(
    llm=llm, embed_model="local:BAAI/bge-small-en-v1.5"
)
index = VectorStoreIndex.from_documents([document],service_context=service_context)

query_engine = index.as_query_engine()

response = query_engine.query(
    "how to create an order release by ship unit?"
)
print(str(response))