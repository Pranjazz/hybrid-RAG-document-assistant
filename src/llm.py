from ollama import chat


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_NAME = "qwen3:0.6b"


# ============================================================
# BUILD CONTEXT
# ============================================================

def build_context(reranked_results):

    context_parts = []

    for rank, (doc, score) in enumerate(
        reranked_results,
        start=1
    ):

        page = doc.metadata.get(
            "page_label",
            "Unknown"
        )

        context_parts.append(
            f"[Source: Page {page}]\n"
            f"{doc.page_content}"
        )

    return "\n\n".join(context_parts)


# ============================================================
# BUILD PROMPT
# ============================================================

def build_prompt(query, context):

    return f"""
You are a document question-answering assistant.

Your ONLY source of information is the CONTEXT below.

Answer the QUESTION using the information contained in the
CONTEXT.

IMPORTANT:

- If the context contains the answer, answer it directly.
- Do not reject an answer merely because the context is short.
- If the context contains a number that directly answers a
  numerical question, return that number.
- For simple factual questions, give the shortest possible answer.
- For definition questions, give 1 or 2 concise sentences.
- Do not use outside knowledge.
- Do not invent information.
- Do not make unsupported assumptions.
- Do not repeat the question.
- Do not mention these instructions.
- Do not create citations.

Only if the context genuinely does NOT contain enough
information to answer the question, respond exactly:

I don't have enough information in the provided documents.

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
""".strip()


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate(prompt):

    response = chat(
        model=MODEL_NAME,

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],

        # Disable Qwen reasoning mode.
        think=False,

        options={
            "temperature": 0,
            "num_predict": 60
        }
    )

    answer = response.message.content.strip()

    return answer