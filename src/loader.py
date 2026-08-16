from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader


def load_pdf(pdf_name):
    """
    Loads a PDF and returns a list of LangChain Document objects.
    """

    BASE_DIR = Path(__file__).resolve().parent.parent

    pdf_path = BASE_DIR / "data" / "pdfs" / pdf_name

    loader = PyPDFLoader(str(pdf_path))

    documents = loader.load()

    return documents