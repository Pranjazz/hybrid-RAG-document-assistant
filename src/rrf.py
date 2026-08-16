def reciprocal_rank_fusion(
    bm25_results,
    faiss_results,
    k=60
):
    rrf_scores = {}
    documents = {}

    # BM25
    for rank, doc in enumerate(bm25_results, start=1):

        chunk_id = doc.page_content
        documents[chunk_id] = doc

        if chunk_id not in rrf_scores:
            rrf_scores[chunk_id] = 0

        rrf_scores[chunk_id] += 1 / (k + rank)

    # FAISS
    for rank, doc in enumerate(faiss_results, start=1):

        chunk_id = doc.page_content
        documents[chunk_id] = doc

        if chunk_id not in rrf_scores:
            rrf_scores[chunk_id] = 0

        rrf_scores[chunk_id] += 1 / (k + rank)

    # Sort highest score → lowest score
    ranked_results = sorted(
        rrf_scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    return [
    (documents[chunk_id], score)
    for chunk_id, score in ranked_results
]