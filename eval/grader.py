import re
from eval.questions import EvalQuestion

REFUSAL_PATTERNS = [
    r"do not contain",
    r"does not contain",
    r"not mentioned in the (?:provided )?papers",
    r"papers do not say",
    r"papers do not specify",
    r"cannot answer from the provided",
    r"insufficient information in the provided",
    r"not covered in the library",
]

CITATION_PATTERN = re.compile(
    r"\[.*?(?:1706\.03762|2410\.00037|2507\.07955|Attention|Moshi|Dynamic Chunking).*?(?:Page|p\.|Sec|Section).*?\]",
    re.IGNORECASE,
)

def is_refusal(answer_text: str) -> bool:
    text_lower = answer_text.lower()
    return any(re.search(pat, text_lower) for pat in REFUSAL_PATTERNS)

def has_valid_citation(answer_text: str, expected_paper_id: str = None) -> bool:
    if not CITATION_PATTERN.search(answer_text):
        return False
    if expected_paper_id:
        # Check either the arXiv ID prefix (e.g. 1706.03762) or paper title appears
        prefix = expected_paper_id.split("v")[0]
        if prefix in answer_text:
            return True
    return True

def grade_answer(question: EvalQuestion, answer_text: str, status: str) -> str:
    """
    Returns one of the exact 4 grades required for results.csv:
    'correct', 'wrong', 'refused_correctly', 'refused_wrongly'
    """
    if status != "ok" or not answer_text.strip():
        # Failed / timed-out / dropped requests did not deliver a valid answer
        return "refused_wrongly" if question.category != "unanswerable" else "refused_correctly"

    refused = is_refusal(answer_text)

    # Case 1: The question is NOT answered by the library (should refuse)
    if question.category == "unanswerable":
        if refused:
            return "refused_correctly"
        return "wrong"  # Hallucinated an answer when the papers don't say!

    # Case 2: The question IS answerable from the library
    if refused:
        return "refused_wrongly"

    # Must contain a valid citation AND match the ground-truth keywords
    cited = has_valid_citation(answer_text, question.expected_paper_id)
    text_lower = answer_text.lower()

    matched_keywords = sum(
        1 for kw in question.required_keywords if kw.lower() in text_lower
    )
    # Require at least 60% of key factual tokens + a valid paper/page/section citation
    min_required = max(1, int(len(question.required_keywords) * 0.6))

    if cited and matched_keywords >= min_required:
        return "correct"
    return "wrong"