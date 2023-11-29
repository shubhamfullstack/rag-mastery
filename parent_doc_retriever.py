# trying the parent document retreiver from langchain

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.storage import InMemoryStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.retrievers import ParentDocumentRetriever
import os

os.environ["OPENAI_API_KEY"] = ""


loader = PyPDFLoader("data/order-management.pdf", extract_images=True)
docs = loader.load_and_split()

# This text splitter is used to create the child documents
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
# The vectorstore to use to index the child chunks
vectorstore = Chroma(
    collection_name="full_documents", embedding_function=OpenAIEmbeddings()
)
# The storage layer for the parent documents
store = InMemoryStore()
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
)

retriever.add_documents(docs, ids=None)

sub_docs = vectorstore.similarity_search("how to identify packaged item")

print(sub_docs[0].page_content)

retrieved_docs = retriever.get_relevant_documents("how to identify packaged item")
print("########")
print(retrieved_docs[0].page_content)