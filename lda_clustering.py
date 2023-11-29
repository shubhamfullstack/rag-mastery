from langchain.document_loaders import PyPDFLoader
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

loader = PyPDFLoader("data/order-management.pdf", extract_images=True)
docs = loader.load_and_split()

documents = []
for doc in docs:
    documents.append(doc.page_content)    
# Preprocess and lemmatize documents using spaCy
processed_documents = []
for doc in documents:
    tokens = nlp(doc)
    lemmatized_text = " ".join([token.lemma_ for token in tokens if not token.is_stop and token.is_alpha])
    processed_documents.append(lemmatized_text)

# Vectorize the processed documents using CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(processed_documents)

# Apply Latent Dirichlet Allocation for topic modeling
num_topics = 8  # You can adjust this parameter based on your data
lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
lda.fit(X)

# Display the topics and associated words
feature_names = vectorizer.get_feature_names_out()
for topic_idx, topic in enumerate(lda.components_):
    top_keywords_idx = topic.argsort()[:-5 - 1:-1]
    top_keywords = [feature_names[i] for i in top_keywords_idx]
    print(f"Topic #{topic_idx + 1}: {', '.join(top_keywords)}")

# Assign topics to documents
topic_assignments = lda.transform(X)
for i, doc in enumerate(documents):
    assigned_topic = topic_assignments[i].argmax()
    print(f"Document #{i + 1} is assigned to Topic #{assigned_topic + 1}")