import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC_DIR))

from bm25_retriever import retrieve_bm25
from vector_store import retrieve_faiss
from rrf import reciprocal_rank_fusion
from crossencoder import rerank


queries = [
    "What is multi-head attention?",
    "What is the capital of France?"
]


for query in queries:

    print("\n" + "=" * 80)
    print("QUERY:", query)
    print("=" * 80)

    bm25_results = retrieve_bm25(query, 10)

    faiss_results = retrieve_faiss(query, 10)

    rrf_results = reciprocal_rank_fusion(
        bm25_results,
        faiss_results
    )

    documents = [
        doc
        for doc, score in rrf_results
    ]

    results = rerank(
        query,
        documents,
        top_k=3
    )

    for rank, (doc, score) in enumerate(
        results,
        start=1
    ):

        print(
            f"Rank {rank} | "
            f"Score: {score:.4f} | "
            f"Page: {doc.metadata.get('page_label')}"
        )