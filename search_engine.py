from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


def load_faiss_data(index_file_path, sentences_file_path):
    index = faiss.read_index(index_file_path)
    sentences = np.load(sentences_file_path, allow_pickle=True).tolist()
    return index, sentences


index_file_path = "kandc_index.index"
sentences_file_path = "kandc.npy"


index, sentences = load_faiss_data(index_file_path, sentences_file_path)

# Example: search for similar sentences
query = "epileptic seizure can be"
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
query_embedding = model.encode([query])

D, I = index.search(query_embedding, k=5)  # Search for the 5 nearest neighbors
print(f"Query: {query}")
print("Top 5 similar sentences:")
for idx in I[0]:
    print(sentences[idx], "\n")
