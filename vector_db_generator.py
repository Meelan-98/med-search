from PyPDF2 import PdfReader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS


def load_embedding_model(model_path, normalize_embedding=True):
    return HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": normalize_embedding},
    )


embeddings = load_embedding_model(model_path="all-MiniLM-L6-v2")


def split_paragraphs(rawText):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )

    return text_splitter.split_text(rawText)


def load_pdfs(pdfs):
    text_chunks = []

    for pdf in pdfs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            raw = page.extract_text()
            chunks = split_paragraphs(raw)
            text_chunks += chunks
    return text_chunks


list_of_pdfs = [
    "documents/kandc.pdf",
]
text_chunks = load_pdfs(list_of_pdfs)

store = FAISS.from_texts(text_chunks, embeddings)

store.save_local("./vector_spaces/kandc_store")
