from sentence_transformers import SentenceTransformer
import numpy as np


model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


def split_into_claims(answer):

    claims = [
        sentence.strip()
        for sentence in answer.split(".")
        if sentence.strip()
    ]

    return claims


def calculate_similarity(claim, context):

    embeddings = model.encode(
        [claim, context],
        normalize_embeddings=True
    )

    similarity = np.dot(
        embeddings[0],
        embeddings[1]
    )

    return similarity


def verify_claims(answer, context):

    claims = split_into_claims(answer)

    results = []

    for claim in claims:

        score = calculate_similarity(
            claim,
            context
        )

        results.append({
            "claim": claim,
            "score": score
        })

    return results


if __name__ == "__main__":

    answer = """
    Multi-head attention uses multiple attention heads.
    The queries, keys and values are linearly projected multiple times.
    The outputs are concatenated and projected.
    """

    context = """
    We found it beneficial to linearly project the queries,
    keys and values h times with different, learned linear projections.
    On each of these projected versions we perform the attention
    function in parallel. These are concatenated and once again
    projected.
    """

    results = verify_claims(
        answer,
        context
    )

    print("\n===== CLAIM VERIFICATION =====")

    for result in results:

        print("=" * 60)
        print("Claim:", result["claim"])
        print(
            "Similarity:",
            round(result["score"], 4)
        )