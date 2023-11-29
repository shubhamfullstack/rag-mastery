import warnings

import os
import openai
from llama_index import Document
from llama_index import SimpleDirectoryReader
from llama_index.node_parser import HierarchicalNodeParser
from llama_index.node_parser import get_leaf_nodes
from llama_index.llms import OpenAI
from llama_index import ServiceContext
from llama_index import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.indices.postprocessor import SentenceTransformerRerank
from llama_index.retrievers import AutoMergingRetriever
from llama_index.query_engine import RetrieverQueryEngine


openai.api_key = ""
warnings.filterwarnings('ignore')


documents = SimpleDirectoryReader(
    input_files=["data/order-management.pdf"]
).load_data()

document = Document(text="\n\n".join([doc.text for doc in documents]))

# create the hierarchical node parser w/ default settings
node_parser = HierarchicalNodeParser.from_defaults(
    chunk_sizes=[2048, 512, 128]
)

nodes = node_parser.get_nodes_from_documents([document])
leaf_nodes = get_leaf_nodes(nodes)
llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)


auto_merging_context = ServiceContext.from_defaults(
    llm=llm,
    embed_model="local:BAAI/bge-small-en-v1.5",
    node_parser=node_parser,
)


storage_context = StorageContext.from_defaults()
storage_context.docstore.add_documents(nodes)

automerging_index = VectorStoreIndex(
    leaf_nodes, storage_context=storage_context, service_context=auto_merging_context
)

if not os.path.exists("./merging_index"):
    storage_context = StorageContext.from_defaults()
    storage_context.docstore.add_documents(nodes)

    automerging_index = VectorStoreIndex(
            leaf_nodes,
            storage_context=storage_context,
            service_context=auto_merging_context
        )

    automerging_index.storage_context.persist(persist_dir="./merging_index")
else:
    automerging_index = load_index_from_storage(
        StorageContext.from_defaults(persist_dir="./merging_index"),
        service_context=auto_merging_context
    )


automerging_retriever = automerging_index.as_retriever(
    similarity_top_k=12
)

retriever = AutoMergingRetriever(
    automerging_retriever, 
    automerging_index.storage_context, 
    verbose=True
)

rerank = SentenceTransformerRerank(top_n=6, model="BAAI/bge-reranker-base")

auto_merging_engine = RetrieverQueryEngine.from_args(
    automerging_retriever, node_postprocessors=[rerank]
)

auto_merging_response = auto_merging_engine.query(
    "how to create an order release by ship unit?"
)

print(auto_merging_response)