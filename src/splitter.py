from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def get_chunks():

    # Get project root
    BASE_DIR = Path(__file__).resolve().parent.parent

    # PDF path
    pdf_path = (
        BASE_DIR
        / "data"
        / "pdfs"
        / "NIPS-2017-attention-is-all-you-need-Paper.pdf"
    )

    # Load PDF
    loader = PyPDFLoader(str(pdf_path))
    documents = loader.load()

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(documents)

    # Give every chunk a unique ID
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i

    return chunks