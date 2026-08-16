from rank_bm25 import BM25Okapi
from splitter import get_chunks


# Load chunks once
chunks = get_chunks()

# Prepare text for BM25
tokenized_documents = [
    chunk.page_content.lower().split()
    for chunk in chunks
]

# Build BM25 index once
bm25 = BM25Okapi(tokenized_documents)


def retrieve_bm25(query, k=3):

    # Convert query into tokens
    tokenized_query = query.lower().split()

    # Get BM25 score for every chunk
    scores = bm25.get_scores(tokenized_query)

    # Get indices of highest-scoring chunks
    top_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:k]

    # Return original Document objects
    results = [
        chunks[index]
        for index in top_indices
    ]

    return results