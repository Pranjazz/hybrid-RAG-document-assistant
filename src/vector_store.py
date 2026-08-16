import faiss
import numpy as np

from splitter import get_chunks
from sentence_transformers import SentenceTransformer


# Load chunks once
chunks = get_chunks()


# Extract text from chunks
texts = [
    chunk.page_content
    for chunk in chunks
]


# Load embedding model once
model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# Create embeddings
embeddings = model.encode(texts)

# FAISS expects float32
embeddings = np.array(
    embeddings
).astype("float32")


# Create FAISS index
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(
    dimension
)


# Add vectors to FAISS
index.add(embeddings)


def retrieve_faiss(query, k=3):

    # Convert query into embedding
    query_embedding = model.encode(
        [query]
    )

    query_embedding = np.array(
        query_embedding
    ).astype("float32")


    # Search FAISS
    distances, indices = index.search(
        query_embedding,
        k
    )


    # Return original Document objects
    results = [
        chunks[index]
        for index in indices[0]
    ]

    return results