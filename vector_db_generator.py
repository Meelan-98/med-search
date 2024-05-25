import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


# Step 1: Read and extract text from the PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text


# Step 2: Split text into sentences
def split_into_sentences(text):
    # Using a simple sentence split. You may use more sophisticated methods if necessary
    sentences = text.split(". ")
    sentences = [sentence.strip() for sentence in sentences if sentence]
    return sentences


# Step 3: Encode sentences using SentenceTransformer
def encode_sentences(sentences, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(sentences)
    return embeddings


# Step 4: Create a FAISS vector database
def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index


# # Step 5: Save FAISS index to a file
# def save_faiss_index(index, file_path):
#     faiss.write_index(index, file_path)


def save_faiss_data(index, sentences, index_file_path, sentences_file_path):
    faiss.write_index(index, index_file_path)
    np.save(sentences_file_path, np.array(sentences))


# Main function to create the FAISS vector DB from a PDF
def create_faiss_db_from_pdf(pdf_path, index_file_path, sentence_file_path):
    text = extract_text_from_pdf(pdf_path)
    sentences = split_into_sentences(text)
    embeddings = encode_sentences(sentences)
    index = create_faiss_index(embeddings)
    save_faiss_data(index, sentences, index_file_path, sentence_file_path)


if __name__ == "__main__":
    pdf_path = "documents/kandc.pdf"
    index_file_path = "kandc_index.index"
    sentences_file_path = "kandc.npy"

    # Create and save the FAISS index
    create_faiss_db_from_pdf(pdf_path, index_file_path, sentences_file_path)
