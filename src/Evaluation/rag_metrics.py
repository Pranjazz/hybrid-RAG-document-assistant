import re


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):
    """
    Normalize text for lexical comparison.
    """

    text = str(text).lower()

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize(text):

    return set(
        normalize_text(text).split()
    )


# ============================================================
# CONTEXT PRECISION
# ============================================================

def context_precision(
    retrieved_documents,
    expected_pages
):
    """
    Project-level approximation of context precision.

    Measures the proportion of retrieved documents
    that come from expected relevant pages.
    """

    if not retrieved_documents:
        return 0.0

    if not expected_pages:
        return 0.0

    expected_pages = {
        str(page)
        for page in expected_pages
    }

    relevant = 0

    for doc in retrieved_documents:

        page = str(
            doc.metadata.get(
                "page_label"
            )
        )

        if page in expected_pages:

            relevant += 1

    return relevant / len(
        retrieved_documents
    )


# ============================================================
# CONTEXT RECALL
# ============================================================

def context_recall(
    retrieved_documents,
    expected_pages
):
    """
    Project-level approximation of context recall.

    Measures how many expected relevant pages
    were retrieved.
    """

    if not expected_pages:
        return 0.0

    expected_pages = {
        str(page)
        for page in expected_pages
    }

    retrieved_pages = set()

    for doc in retrieved_documents:

        page = doc.metadata.get(
            "page_label"
        )

        if page is not None:

            retrieved_pages.add(
                str(page)
            )

    found = (
        retrieved_pages
        .intersection(
            expected_pages
        )
    )

    return len(found) / len(
        expected_pages
    )


# ============================================================
# FAITHFULNESS
# ============================================================

def faithfulness(
    answer,
    context
):
    """
    Project-level lexical approximation
    of faithfulness.

    Measures the proportion of answer
    tokens that appear in the context.

    This is NOT the official RAGAS metric.
    """

    answer_tokens = tokenize(
        answer
    )

    context_tokens = tokenize(
        context
    )

    if not answer_tokens:

        return 0.0

    supported = (
        answer_tokens
        .intersection(
            context_tokens
        )
    )

    return (
        len(supported)
        /
        len(answer_tokens)
    )


# ============================================================
# ANSWER RELEVANCY
# ============================================================

def answer_relevancy(
    question,
    answer
):
    """
    Project-level approximation of answer
    relevancy.

    Uses token overlap but also handles
    short factual answers.

    This is NOT the official RAGAS metric.
    """

    question_tokens = tokenize(
        question
    )

    answer_tokens = tokenize(
        answer
    )

    if not question_tokens:
        return 0.0

    if not answer_tokens:
        return 0.0


    # --------------------------------------------------------
    # Direct token overlap
    # --------------------------------------------------------

    overlap = (
        question_tokens
        .intersection(
            answer_tokens
        )
    )

    overlap_score = (
        len(overlap)
        /
        len(question_tokens)
    )


    # --------------------------------------------------------
    # Short factual answer
    # --------------------------------------------------------
    #
    # Example:
    #
    # Question:
    # "How many attention heads does
    #  the Transformer use?"
    #
    # Answer:
    # "8"
    #
    # There is no lexical overlap, but
    # the answer may still be valid.
    #
    # We therefore don't automatically
    # classify short answers as irrelevant.
    # --------------------------------------------------------

    if len(answer_tokens) <= 3:

        # Numerical answer
        if any(
            token.isdigit()
            for token in answer_tokens
        ):

            return max(
                overlap_score,
                0.8
            )

        # Very short textual answer
        return max(
            overlap_score,
            0.5
        )


    # --------------------------------------------------------
    # Normal answer
    # --------------------------------------------------------

    return overlap_score


# ============================================================
# ANSWER TYPE
# ============================================================

def answer_type(answer):
    """
    Classifies the generated answer.

    Useful for evaluation and debugging.
    """

    answer = normalize_text(
        answer
    )

    if not answer:

        return "empty"

    refusal = (
        "i don t have enough information "
        "in the provided documents"
    )

    if refusal in answer:

        return "refusal"

    tokens = answer.split()

    if len(tokens) <= 3:

        if any(
            token.isdigit()
            for token in tokens
        ):

            return "short_factual"

        return "short_answer"

    return "normal"


# ============================================================
# EVALUATION SUMMARY
# ============================================================

def evaluate_answer(
    question,
    answer,
    context
):
    """
    Returns all answer-level metrics
    in one dictionary.
    """

    return {

        "faithfulness":
            faithfulness(
                answer,
                context
            ),

        "answer_relevancy":
            answer_relevancy(
                question,
                answer
            ),

        "answer_type":
            answer_type(
                answer
            )
    }