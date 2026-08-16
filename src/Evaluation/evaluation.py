import sys
from pathlib import Path
import time


# ============================================================
# PATH SETUP
# ============================================================

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent

sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(CURRENT_DIR))


# ============================================================
# IMPORTS
# ============================================================

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

from questions import questions

from rag_metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy
)


# ============================================================
# CONFIGURATION
# ============================================================

K = 10
TOP_K = 3


# ============================================================
# HIT@K
# ============================================================

def hit_at_k(documents, expected_pages):

    if not expected_pages:
        return None

    retrieved_pages = []

    for doc in documents:

        page = doc.metadata.get(
            "page_label"
        )

        if page is not None:
            retrieved_pages.append(
                str(page)
            )

    for expected_page in expected_pages:

        if str(expected_page) in retrieved_pages:
            return 1

    return 0


# ============================================================
# EVALUATE ONE QUESTION
# ============================================================

def evaluate_question(item):

    query = item["question"]

    expected_pages = item.get(
        "expected_pages",
        []
    )

    print("\n" + "=" * 80)
    print("QUESTION:")
    print(query)
    print("=" * 80)


    # ========================================================
    # 1. BM25 RETRIEVAL
    # ========================================================

    start = time.perf_counter()

    bm25_results = retrieve_bm25(
        query,
        K
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
        K
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

    hybrid_documents = [
        doc
        for doc, score in rrf_results
    ]


    # ========================================================
    # 4. CROSS ENCODER RERANKING
    # ========================================================

    start = time.perf_counter()

    reranked_results = rerank(
        query,
        hybrid_documents,
        top_k=TOP_K
    )

    rerank_time = (
        time.perf_counter() - start
    ) * 1000

    reranked_documents = [
        doc
        for doc, score in reranked_results
    ]


    # ========================================================
    # 5. HIT@K
    # ========================================================

    bm25_hit = hit_at_k(
        bm25_results,
        expected_pages
    )

    faiss_hit = hit_at_k(
        faiss_results,
        expected_pages
    )

    hybrid_hit = hit_at_k(
        hybrid_documents[:TOP_K],
        expected_pages
    )

    reranked_hit = hit_at_k(
        reranked_documents,
        expected_pages
    )


    # ========================================================
    # 6. RETRIEVAL METRICS
    # ========================================================

    print("\n----- RETRIEVAL HIT@K -----")

    if bm25_hit is None:

        print(
            f"BM25 Hit@{K}:          N/A"
        )

        print(
            f"FAISS Hit@{K}:         N/A"
        )

        print(
            f"Hybrid Hit@{TOP_K}:    N/A"
        )

        print(
            f"Reranked Hit@{TOP_K}:  N/A"
        )

    else:

        print(
            f"BM25 Hit@{K}:          {bm25_hit}"
        )

        print(
            f"FAISS Hit@{K}:         {faiss_hit}"
        )

        print(
            f"Hybrid Hit@{TOP_K}:    {hybrid_hit}"
        )

        print(
            f"Reranked Hit@{TOP_K}:  {reranked_hit}"
        )


    # ========================================================
    # 7. LATENCY
    # ========================================================

    print("\n----- LATENCY -----")

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


    # ========================================================
    # 8. RELEVANCE GATE
    # ========================================================

    relevant = is_relevant(
        reranked_results
    )


    # ========================================================
    # 9. GENERATE ANSWER
    # ========================================================

    if not relevant:

        answer = (
            "I don't have enough information "
            "in the provided documents."
        )

        context = ""

        grounding_score = 0.0

        precision = 0.0
        recall = 0.0
        faithfulness_score = 0.0
        relevancy_score = 0.0

    else:

        context = build_context(
            reranked_results
        )

        prompt = build_prompt(
            query,
            context
        )

        start = time.perf_counter()

        answer = generate(
            prompt
        )

        generation_time = (
            time.perf_counter() - start
        ) * 1000

        grounding_score = check_grounding(
            answer,
            context
        )

        precision = context_precision(
            reranked_documents,
            expected_pages
        )

        recall = context_recall(
            reranked_documents,
            expected_pages
        )

        faithfulness_score = faithfulness(
            answer,
            context
        )

        relevancy_score = answer_relevancy(
            query,
            answer
        )


    # ========================================================
    # 10. ANSWER
    # ========================================================

    print("\n----- ANSWER -----")

    print(answer)


    # ========================================================
    # 11. RAG METRICS
    # ========================================================

    print("\n----- RAG METRICS -----")

    print(
        f"Context Precision: "
        f"{precision:.4f}"
    )

    print(
        f"Context Recall:    "
        f"{recall:.4f}"
    )

    print(
        f"Faithfulness:      "
        f"{faithfulness_score:.4f}"
    )

    print(
        f"Answer Relevancy:  "
        f"{relevancy_score:.4f}"
    )

    print(
        f"Grounding Score:   "
        f"{grounding_score:.4f}"
    )


    # ========================================================
    # 12. TOP RERANKED DOCUMENTS
    # ========================================================

    print("\n----- TOP RERANKED DOCUMENTS -----")

    for rank, (doc, score) in enumerate(
        reranked_results,
        start=1
    ):

        print(
            f"Rank {rank} | "
            f"Score: {score:.4f} | "
            f"Page: {doc.metadata.get('page_label')}"
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    for item in questions:

        evaluate_question(item)