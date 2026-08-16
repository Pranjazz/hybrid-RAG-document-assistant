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

import time


# ============================================================
# CONFIGURATION
# ============================================================

RETRIEVAL_K = 10

RERANK_CANDIDATES = 8

FINAL_TOP_K = 3


# ============================================================
# RUN RAG
# ============================================================

def run_rag(query):

    total_start = time.perf_counter()


    # ========================================================
    # 1. BM25 RETRIEVAL
    # ========================================================

    start = time.perf_counter()

    bm25_results = retrieve_bm25(
        query,
        RETRIEVAL_K
    )

    bm25_time = (
        time.perf_counter() - start
    ) * 1000


    # ========================================================
    # 2. FAISS RETRIEVAL
    # ========================================================

    start = time.perf_counter()

    faiss_results = retrieve_faiss(
        query,
        RETRIEVAL_K
    )

    faiss_time = (
        time.perf_counter() - start
    ) * 1000


    # ========================================================
    # 3. RECIPROCAL RANK FUSION
    # ========================================================

    start = time.perf_counter()

    rrf_results = reciprocal_rank_fusion(
        bm25_results,
        faiss_results
    )

    rrf_time = (
        time.perf_counter() - start
    ) * 1000


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

    start = time.perf_counter()

    reranked_results = rerank(
        query,
        candidate_documents,
        top_k=FINAL_TOP_K
    )

    rerank_time = (
        time.perf_counter() - start
    ) * 1000


    # ========================================================
    # 6. RELEVANCE GATE
    # ========================================================

    start = time.perf_counter()

    relevant = is_relevant(
        reranked_results
    )

    relevance_time = (
        time.perf_counter() - start
    ) * 1000


    # ========================================================
    # 7. REFUSE IF NOT RELEVANT
    # ========================================================

    if not relevant:

        total_time = (
            time.perf_counter() - total_start
        ) * 1000

        print("\n===== RAG LATENCY =====")

        print(
            f"BM25:          {bm25_time:.2f} ms"
        )

        print(
            f"FAISS:         {faiss_time:.2f} ms"
        )

        print(
            f"RRF:           {rrf_time:.2f} ms"
        )

        print(
            f"Cross Encoder: {rerank_time:.2f} ms"
        )

        print(
            f"Relevance Gate:{relevance_time:.2f} ms"
        )

        print(
            f"Total:         {total_time:.2f} ms"
        )

        return {
            "answer":
                "I don't have enough information "
                "in the provided documents.",

            "context": "",

            "prompt": "",

            "reranked_results":
                reranked_results,

            "grounding_score": 0.0,

            "latency": {
                "bm25": bm25_time,
                "faiss": faiss_time,
                "rrf": rrf_time,
                "cross_encoder": rerank_time,
                "relevance_gate": relevance_time,
                "generation": 0.0,
                "grounding": 0.0,
                "total": total_time
            }
        }


    # ========================================================
    # 8. BUILD CONTEXT
    # ========================================================

    start = time.perf_counter()

    context = build_context(
        reranked_results
    )

    context_time = (
        time.perf_counter() - start
    ) * 1000


    # ========================================================
    # 9. BUILD PROMPT
    # ========================================================

    start = time.perf_counter()

    prompt = build_prompt(
        query,
        context
    )

    prompt_time = (
        time.perf_counter() - start
    ) * 1000


    # ========================================================
    # 10. GENERATE ANSWER
    # ========================================================

    print(
        "\nGenerating answer..."
    )

    start = time.perf_counter()

    answer = generate(
        prompt
    )

    generation_time = (
        time.perf_counter() - start
    ) * 1000


    # ========================================================
    # 11. GROUNDING CHECK
    # ========================================================

    start = time.perf_counter()

    grounding_score = check_grounding(
        answer,
        context
    )

    grounding_time = (
        time.perf_counter() - start
    ) * 1000


    # ========================================================
    # 12. TOTAL LATENCY
    # ========================================================

    total_time = (
        time.perf_counter() - total_start
    ) * 1000


    # ========================================================
    # 13. PRINT LATENCY
    # ========================================================

    print("\n===== RAG LATENCY =====")

    print(
        f"BM25:          {bm25_time:.2f} ms"
    )

    print(
        f"FAISS:         {faiss_time:.2f} ms"
    )

    print(
        f"RRF:           {rrf_time:.2f} ms"
    )

    print(
        f"Cross Encoder: {rerank_time:.2f} ms"
    )

    print(
        f"Relevance Gate:{relevance_time:.2f} ms"
    )

    print(
        f"Context:       {context_time:.2f} ms"
    )

    print(
        f"Prompt:        {prompt_time:.2f} ms"
    )

    print(
        f"LLM:           {generation_time:.2f} ms"
    )

    print(
        f"Grounding:     {grounding_time:.2f} ms"
    )

    print(
        f"Total:         {total_time:.2f} ms"
    )


    # ========================================================
    # 14. RETURN RESULT
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
            grounding_score,

        "latency": {

            "bm25":
                bm25_time,

            "faiss":
                faiss_time,

            "rrf":
                rrf_time,

            "cross_encoder":
                rerank_time,

            "relevance_gate":
                relevance_time,

            "context":
                context_time,

            "prompt":
                prompt_time,

            "generation":
                generation_time,

            "grounding":
                grounding_time,

            "total":
                total_time
        }
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