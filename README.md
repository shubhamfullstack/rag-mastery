## Version

python - 3.10.5


## Getting started

1. Create the virtual env using the below commands

```
 python3 -m venv env
 source env/bin/activate
```

if there is any issue upgrade the pip using `python3 -m pip install --upgrade pip` command

2. Install langchain, chromadb, openai and pypdf packages.

```
pip install langchain openai pypdf
pip install rapidocr-onnxruntime 
pip install tiktoken
pip install chromadb==0.4.5
pip install spacy
pip install gensim
pip install scikit-learn

pip install llama-index transformers
python3 -m pip install torch torchvision
pip install sentence-transformers
```

Note: if there is a issue in downloading spacy model

```
sudo vim /etc/hosts
199.232.68.133 raw.githubusercontent.com
python3 -m spacy download en_core_web_sm
```

## Techniques Tried

1. [ParentDocumentRetreiver by Langchain](https://python.langchain.com/docs/modules/data_connection/retrievers/parent_document_retriever)
    - [Code](./parent_doc_retriever.py)
2. [Sentence Boundary Detection](https://subscription.packtpub.com/book/data/9781800208421/1/ch01lvl1sec07/sentence-boundary-detection)
3. [Topic Modeling and Semantic Clustering with spaCy](https://fouadroumieh.medium.com/topic-modeling-and-semantic-clustering-with-spacy-960dd4ac3c9a)

[chatgpt code for LDA and clustering](https://chat.openai.com/share/032d4617-a520-428c-a936-1e549b89e471)

4. [comparative analysis](https://towardsdatascience.com/how-to-chunk-text-data-a-comparative-analysis-3858c4a0997a)
5. [Sentence window retrievel and auto merging retrievel](./DeepLearningCourseRAG.md)