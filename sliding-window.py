import warnings

import os
import openai
from llama_index import Document
from llama_index import SimpleDirectoryReader
from llama_index.node_parser import SentenceWindowNodeParser
from llama_index.llms import OpenAI
from llama_index import ServiceContext
from llama_index import VectorStoreIndex
from llama_index import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index import load_index_from_storage
from llama_index.indices.postprocessor import MetadataReplacementPostProcessor
from llama_index.indices.postprocessor import SentenceTransformerRerank

openai.api_key = ""
warnings.filterwarnings('ignore')


documents = SimpleDirectoryReader(
    input_files=["data/order-management.pdf"]
).load_data()

document = Document(text="\n\n".join([doc.text for doc in documents]))

# create the sentence window node parser w/ default settings
node_parser = SentenceWindowNodeParser.from_defaults(
    window_size=3,
    window_metadata_key="window",
    original_text_metadata_key="original_text",
)


llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)

sentence_context = ServiceContext.from_defaults(
    llm=llm,
    embed_model="local:BAAI/bge-small-en-v1.5",
    node_parser=node_parser,
)

sentence_index = VectorStoreIndex.from_documents(
    [document], service_context=sentence_context
)

if not os.path.exists("./sentence_index"):
    sentence_index = VectorStoreIndex.from_documents(
        [document], service_context=sentence_context
    )

    sentence_index.storage_context.persist(persist_dir="./sentence_index")
else:
    sentence_index = load_index_from_storage(
        StorageContext.from_defaults(persist_dir="./sentence_index"),
        service_context=sentence_context
    )


def get_sentence_window_query_engine(
    sentence_index, similarity_top_k=6, rerank_top_n=2
):
    # define postprocessors
    postproc = MetadataReplacementPostProcessor(target_metadata_key="window")
    rerank = SentenceTransformerRerank(
        top_n=rerank_top_n, model="BAAI/bge-reranker-base"
    )

    sentence_window_engine = sentence_index.as_query_engine(
        similarity_top_k=similarity_top_k, node_postprocessors=[postproc, rerank]
    )
    return sentence_window_engine

query_engine = get_sentence_window_query_engine(sentence_index, similarity_top_k=6)

response = query_engine.query(
    "how to create an order release by ship unit?"
)
print(str(response))