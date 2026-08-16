import sys
from pathlib import Path

import streamlit as st


# ============================================================
# PATH SETUP
# ============================================================

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR / "src"

sys.path.insert(0, str(SRC_DIR))


# ============================================================
# IMPORT RAG PIPELINE
# ============================================================

from hybrid_retrieval import run_rag


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Hybrid RAG Document Assistant",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# HEADER
# ============================================================

st.title("🤖 Hybrid RAG Document Assistant")

st.markdown(
    """
    Ask questions about the indexed document using:

    **BM25 + FAISS + RRF + Cross-Encoder +  Qwen3 0.6B via Ollama**
    """
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ RAG Pipeline")

    st.markdown(
        """
        **Retrieval**
        - BM25
        - FAISS
        - Reciprocal Rank Fusion

        **Reranking**
        - Cross-Encoder

        **Generation**
        - Qwen3 0.6B via Ollama

        **Evaluation**
        - Grounding Score
        - Source Retrieval
        """
    )

    st.divider()

    st.info(
        "The assistant answers only from information "
        "retrieved from the indexed document."
    )


# ============================================================
# QUESTION INPUT
# ============================================================

st.subheader("Ask a question")

query = st.text_input(
    "Question",
    placeholder="e.g. What is multi-head attention?"
)


# ============================================================
# ASK
# ============================================================

if st.button(
    "🔍 Ask Question",
    type="primary"
):

    if not query.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        # ====================================================
        # RUN RAG
        # ====================================================

        with st.spinner(
            "Running hybrid retrieval..."
        ):

            result = run_rag(
                query
            )


        # ====================================================
        # EXTRACT RESULTS
        # ====================================================

        answer = result.get(
            "answer",
            ""
        )

        grounding_score = result.get(
            "grounding_score",
            0.0
        )

        reranked_results = result.get(
            "reranked_results",
            []
        )

        latency = result.get(
            "latency",
            {}
        )


        # ====================================================
        # ANSWER
        # ====================================================

        st.subheader("💬 Answer")

        st.info(
            answer
        )


        # ====================================================
        # MAIN METRICS
        # ====================================================

        st.subheader("📊 RAG Performance")

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "Grounding",
                f"{grounding_score:.3f}"
            )

        with col2:

            st.metric(
                "Documents",
                len(reranked_results)
            )

        with col3:

            total = latency.get(
                "total",
                0
            )

            st.metric(
                "Total Latency",
                f"{total / 1000:.2f}s"
            )

        with col4:

            llm_time = latency.get(
                "generation",
                0
            )

            st.metric(
                "LLM",
                f"{llm_time / 1000:.2f}s"
            )


        # ====================================================
        # LATENCY BREAKDOWN
        # ====================================================

        st.subheader(
            "⏱️ Latency Breakdown"
        )

        latency_col1, latency_col2 = st.columns(2)

        with latency_col1:

            st.write(
                f"**BM25:** "
                f"{latency.get('bm25', 0):.2f} ms"
            )

            st.write(
                f"**FAISS:** "
                f"{latency.get('faiss', 0):.2f} ms"
            )

            st.write(
                f"**RRF:** "
                f"{latency.get('rrf', 0):.2f} ms"
            )

            st.write(
                f"**Cross-Encoder:** "
                f"{latency.get('cross_encoder', 0):.2f} ms"
            )

        with latency_col2:

            st.write(
                f"**Context:** "
                f"{latency.get('context', 0):.2f} ms"
            )

            st.write(
                f"**Prompt:** "
                f"{latency.get('prompt', 0):.2f} ms"
            )

            st.write(
                f"**LLM:** "
                f"{latency.get('generation', 0):.2f} ms"
            )

            st.write(
                f"**Grounding:** "
                f"{latency.get('grounding', 0):.2f} ms"
            )


        # ====================================================
        # SOURCES
        # ====================================================

        st.subheader(
            "📚 Retrieved Sources"
        )

        if reranked_results:

            for rank, (
                doc,
                score
            ) in enumerate(
                reranked_results,
                start=1
            ):

                page = doc.metadata.get(
                    "page_label",
                    "Unknown"
                )

                with st.expander(
                    f"Rank {rank} | "
                    f"Page {page} | "
                    f"Cross-Encoder Score: {score:.4f}"
                ):

                    st.write(
                        doc.page_content
                    )

        else:

            st.info(
                "No relevant documents were retrieved."
            )


        # ====================================================
        # CONTEXT
        # ====================================================

        if result.get("context"):

            with st.expander(
                "🔎 View Combined Context"
            ):

                st.text(
                    result["context"]
                )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Hybrid RAG Document Assistant | "
    "BM25 + FAISS + RRF + Cross-Encoder +  Qwen3 0.6B via Ollama"
)