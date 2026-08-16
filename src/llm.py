from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


# ============================================================
# MODEL
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME
)


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
            "page_label"
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
Use only the information in the context to answer the question.

If the answer is not explicitly supported by the context, say:

I don't have enough information in the provided documents.

Give only the answer. Do not explain your reasoning.

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
""".strip()


# ============================================================
# CLEAN GENERATED ANSWER
# ============================================================

def clean_answer(answer):

    # Remove accidental conversation continuations

    stop_markers = [
        "\nHuman:",
        "\nAssistant:",
        "Human:",
        "Assistant:",
    ]

    for marker in stop_markers:

        if marker in answer:

            answer = answer.split(
                marker,
                1
            )[0]

    # Remove accidental leading labels

    prefixes = [
        "ANSWER:",
        "Answer:",
        "assistant:",
        "Assistant:"
    ]

    for prefix in prefixes:

        if answer.startswith(prefix):

            answer = answer[
                len(prefix):
            ].strip()

    return answer.strip()


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate(prompt):

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    )

    with torch.no_grad():

        outputs = model.generate(
            **inputs,

            max_new_tokens=80,

            do_sample=False,

            repetition_penalty=1.05,

            eos_token_id=tokenizer.eos_token_id,

            pad_token_id=tokenizer.eos_token_id
        )

    # Remove the input prompt

    generated_tokens = outputs[
        0
    ][
        inputs["input_ids"].shape[1]:
    ]

    answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )

    answer = clean_answer(
        answer
    )

    return answer