from sentence_transformers import SentenceTransformer
import numpy as np
import re


# ============================================================
# MODEL
# ============================================================

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# ============================================================
# TEXT SPLITTING
# ============================================================

def split_into_sentences(text):

    sentences = re.split(
        r'(?<=[.!?])\s+|\n+',
        text
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


# ============================================================
# SEMANTIC SIMILARITY
# ============================================================

def semantic_similarity(
    text1,
    text2
):

    embeddings = model.encode(
        [text1, text2],
        normalize_embeddings=True
    )

    return float(
        np.dot(
            embeddings[0],
            embeddings[1]
        )
    )


# ============================================================
# GROUNDING CHECK
# ============================================================

def check_grounding(
    answer,
    context
):

    if not answer.strip():
        return 0.0

    if not context.strip():
        return 0.0


    # --------------------------------------------------------
    # Split context into smaller pieces
    # --------------------------------------------------------

    context_sentences = (
        split_into_sentences(
            context
        )
    )


    if not context_sentences:
        return 0.0


    # --------------------------------------------------------
    # Encode answer + context sentences
    # --------------------------------------------------------

    answer_embedding = model.encode(
        [answer],
        normalize_embeddings=True
    )[0]

    context_embeddings = model.encode(
        context_sentences,
        normalize_embeddings=True
    )


    # --------------------------------------------------------
    # Compare answer with every context sentence
    # --------------------------------------------------------

    similarities = np.dot(
        context_embeddings,
        answer_embedding
    )


    # Best supporting sentence
    best_score = float(
        np.max(similarities)
    )


    # --------------------------------------------------------
    # Print supporting evidence
    # --------------------------------------------------------

    best_index = int(
        np.argmax(similarities)
    )

    best_sentence = (
        context_sentences[best_index]
    )


    print(
        f"Semantic grounding score: "
        f"{best_score:.4f}"
    )

    print(
        "Best supporting context:"
    )

    print(
        best_sentence[:300]
    )


    return best_score