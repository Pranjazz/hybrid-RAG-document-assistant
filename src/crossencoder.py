from sentence_transformers import CrossEncoder


model = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


def rerank(query, documents, top_k=5):

    # Create query-document pairs
    pairs = [
        [query, doc.page_content]
        for doc in documents
    ]

    # Get relevance scores
    scores = model.predict(pairs)

    # Combine documents with scores
    scored_documents = list(
        zip(documents, scores)
    )

    # Sort highest score first
    scored_documents.sort(
        key=lambda item: item[1],
        reverse=True
    )

    # Return top documents
    return scored_documents[:top_k]

def is_relevant(reranked_results, threshold=0.0):

    if not reranked_results:
        return False

    best_score = reranked_results[0][1]

    return best_score >= threshold