from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings


def load_embedding_model(model_path, normalize_embedding=True):
    return HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": normalize_embedding},
    )


embeddings = load_embedding_model(model_path="all-MiniLM-L6-v2")

# Load the FAISS vector store from the saved directory
vector_store = FAISS.load_local(
    "vectorstore", embeddings, allow_dangerous_deserialization=True
)


# Define your query
query = "Define HRM"

similar_chunks = vector_store.similarity_search(query)

print(similar_chunks)
