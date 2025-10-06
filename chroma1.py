from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')  # Fast & free

import chromadb
client = chromadb.Client()
collection = client.create_collection(name="my_text_data")

text = "The capital of France is Paris."
vector = model.encode(text)

collection.add(
    documents=[text],
    embeddings=[vector.tolist()],
    ids=["doc1"]
)

query = "Where is the Eiffel Tower?"
query_vector = model.encode(query)

results = collection.query(
    query_embeddings=[query_vector.tolist()],
    n_results=1
)

print(results)
