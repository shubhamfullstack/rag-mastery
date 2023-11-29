import numpy as np
import spacy
from langchain.document_loaders import PyPDFLoader

# Load the Spacy model
nlp = spacy.load('en_core_web_sm')

loader = PyPDFLoader("data/order-management.pdf", extract_images=True)
docs = loader.load_and_split()

documents = []
for doc in docs:
    documents.append(doc.page_content)   
    
text = "".join(documents) 

def process(text):
    doc = nlp(text)
    sents = list(doc.sents)
    vecs = np.stack([sent.vector / sent.vector_norm for sent in sents])

    return sents, vecs

def cluster_text(sents, vecs, threshold):
    clusters = [[0]]
    for i in range(1, len(sents)):
        if np.dot(vecs[i], vecs[i-1]) < threshold:
            clusters.append([])
        clusters[-1].append(i)
    
    return clusters

def clean_text(text):
    # Add your text cleaning process here
    return text

# Initialize the clusters lengths list and final texts list
clusters_lens = []
final_texts = []

# Process the chunk
threshold = 0.3
sents, vecs = process(text)

# Cluster the sentences
clusters = cluster_text(sents, vecs, threshold)

for cluster in clusters:
    cluster_txt = clean_text(' '.join([sents[i].text for i in cluster]))
    cluster_len = len(cluster_txt)
    
    # Check if the cluster is too short
    if cluster_len < 60:
        continue
    
    # Check if the cluster is too long
    elif cluster_len > 3000:
        threshold = 0.6
        sents_div, vecs_div = process(cluster_txt)
        reclusters = cluster_text(sents_div, vecs_div, threshold)
        
        for subcluster in reclusters:
            div_txt = clean_text(' '.join([sents_div[i].text for i in subcluster]))
            div_len = len(div_txt)
            
            if div_len < 60 or div_len > 3000:
                continue
            
            clusters_lens.append(div_len)
            final_texts.append(div_txt)
            
    else:
        clusters_lens.append(cluster_len)
        final_texts.append(cluster_txt)

print(final_texts[50])
#print(cluster_len)