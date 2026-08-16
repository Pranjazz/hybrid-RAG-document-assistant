from bm25_retriever import retrieve_bm25
from vector_store import retrieve_faiss
from rrf import reciprocal_rank_fusion
from crossencoder import rerank, is_relevant

from llm import (
    build_context,
    build_prompt,
    generate
)

from grounding import check_grounding


# ============================================================
# CONFIGURATION
# ============================================================

# Candidates retrieved from each first-stage retriever
RETRIEVAL_K = 10

# Candidates passed from RRF to Cross Encoder
RERANK_CANDIDATES = 8

# Final documents sent to the LLM
FINAL_TOP_K = 3


# ============================================================
# RUN RAG
# ============================================================

def run_rag(query):


    # ========================================================
    # 1. BM25 RETRIEVAL
    # ========================================================

    bm25_results = retrieve_bm25(
        query,
        RETRIEVAL_K
    )


    # ========================================================
    # 2. FAISS RETRIEVAL
    # ========================================================

    faiss_results = retrieve_faiss(
        query,
        RETRIEVAL_K
    )


    # ========================================================
    # 3. RECIPROCAL RANK FUSION
    # ========================================================

    rrf_results = reciprocal_rank_fusion(
        bm25_results,
        faiss_results
    )


    # ========================================================
    # 4. LIMIT RRF CANDIDATES
    # ========================================================

    candidate_documents = [
        doc
        for doc, score in rrf_results[
            :RERANK_CANDIDATES
        ]
    ]


    # ========================================================
    # 5. CROSS ENCODER RERANKING
    # ========================================================

    reranked_results = rerank(
        query,
        candidate_documents,
        top_k=FINAL_TOP_K
    )


    # ========================================================
    # 6. RELEVANCE GATE
    # ========================================================

    if not is_relevant(
        reranked_results
    ):

        return {
            "answer":
                "I don't have enough information "
                "in the provided documents.",

            "context": "",

            "prompt": "",

            "reranked_results":
                reranked_results,

            "grounding_score": 0.0
        }


    # ========================================================
    # 7. BUILD CONTEXT
    # ========================================================

    context = build_context(
        reranked_results
    )


    # ========================================================
    # 8. BUILD PROMPT
    # ========================================================

    prompt = build_prompt(
        query,
        context
    )


    # ========================================================
    # 9. GENERATE ANSWER
    # ========================================================

    answer = generate(
        prompt
    )


    # ========================================================
    # 10. GROUNDING CHECK
    # ========================================================

    grounding_score = check_grounding(
        answer,
        context
    )


    # ========================================================
    # 11. RETURN RESULT
    # ========================================================

    return {

        "answer":
            answer,

        "context":
            context,

        "prompt":
            prompt,

        "reranked_results":
            reranked_results,

        "grounding_score":
            grounding_score
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    query = input(
        "\nEnter your question: "
    )

    result = run_rag(
        query
    )


    print(
        "\n===== FINAL ANSWER =====\n"
    )

    print(
        result["answer"]
    )


    print(
        "\n===== SEMANTIC GROUNDING SCORE ====="
    )

    print(
        f"{result['grounding_score']:.4f}"
    )


    print(
        "\n===== FINAL RERANKED RESULTS =====\n"
    )


    for rank, (doc, score) in enumerate(
        result["reranked_results"],
        start=1
    ):

        print(
            "=" * 80
        )

        print(
            f"Rank: {rank}"
        )

        print(
            f"Cross Encoder Score: "
            f"{score:.4f}"
        )

        print(
            f"Page: "
            f"{doc.metadata.get('page_label')}"
        )

        print(
            doc.page_content[:500]
        )

        print(
            "=" * 80
        )