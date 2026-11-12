from dataclasses import dataclass
from typing import List, Optional

@dataclass(frozen=True)
class EvalQuestion:
    qid: str
    question: str
    category: str  # "factual", "naive_trap", "unanswerable"
    expected_paper_id: Optional[str]  # None if unanswerable
    required_keywords: List[str]      # At least one/all keywords required for "correct"
    trap_explanation: str = ""

EVAL_QUESTIONS: List[EvalQuestion] = [
    # --- PART 1: Specific Factual & Citation Questions ---
    EvalQuestion(
        qid="Q01",
        question="In 'Attention Is All You Need', what are the beta_1, beta_2, and epsilon hyperparameters used for the Adam optimizer, and how many warmup_steps are used?",
        category="factual",
        expected_paper_id="1706.03762v7",
        required_keywords=["0.9", "0.98", "4000"],
    ),
    EvalQuestion(
        qid="Q02",
        question="What single-model BLEU score does the big Transformer achieve on the WMT 2014 English-to-German translation task, and how many days was it trained on 8 P100 GPUs?",
        category="factual",
        expected_paper_id="1706.03762v7",
        required_keywords=["28.4", "3.5"],
    ),
    EvalQuestion(
        qid="Q03",
        question="In the Transformer paper, what is the complexity per layer for Self-Attention compared to a Recurrent layer in terms of sequence length n and representation dimension d?",
        category="factual",
        expected_paper_id="1706.03762v7",
        required_keywords=["n^2", "n", "d"],
    ),
    EvalQuestion(
        qid="Q04",
        question="In the Moshi paper, what frame rate (in Hz) and frame size (in ms) does the Mimi neural audio codec operate at, and what is its target bandwidth in kbps?",
        category="factual",
        expected_paper_id="2410.00037v1",
        required_keywords=["12.5", "80", "1.1"],
    ),
    EvalQuestion(
        qid="Q05",
        question="How does Moshi's Inner Monologue mechanism structure the sequence of text tokens relative to audio tokens at each timestep, and why does it improve linguistic quality?",
        category="factual",
        expected_paper_id="2410.00037v1",
        required_keywords=["text", "audio", "time", "aligned", "monologue"],
    ),
    EvalQuestion(
        qid="Q06",
        question="What are the two Transformer components in Moshi's RQ-Transformer hierarchy (temporal vs depth), and how many parameters does the main temporal Helium backbone have?",
        category="factual",
        expected_paper_id="2410.00037v1",
        required_keywords=["7b", "depth", "temporal"],
    ),
    EvalQuestion(
        qid="Q07",
        question="In 'Dynamic Chunking for End-to-End Hierarchical Sequence Modeling', how does the routing module compute boundary probabilities between adjacent timesteps inside an H-Net?",
        category="factual",
        expected_paper_id="2507.07955v2",
        required_keywords=["cosine", "similarity", "boundary", "routing"],
    ),
    EvalQuestion(
        qid="Q08",
        question="How does Dynamic Chunking (H-Net) make discrete boundary selections differentiable during end-to-end training without using an external tokenizer?",
        category="factual",
        expected_paper_id="2507.07955v2",
        required_keywords=["smoothing", "ste", "straight-through", "upsampler", "EMA"],
    ),
    EvalQuestion(
        qid="Q09",
        question="What compression ratio target loss or regularization mechanism does Dynamic Chunking use to control the average chunk length in an H-Net?",
        category="factual",
        expected_paper_id="2507.07955v2",
        required_keywords=["ratio", "loss", "target", "boundary"],
    ),

    # --- PART 2: The 3 "Naive Retriever Traps" (For Report Heading after Section 5) ---
    EvalQuestion(
        qid="Q10",
        question="What is the end-to-end theoretical latency and real-world latency of the streaming Transformer model described in the library?",
        category="naive_trap",
        expected_paper_id="2410.00037v1",
        required_keywords=["160", "200", "moshi"],
        trap_explanation=(
            "Trap 1 (Cross-Paper Shared Terminology): A naive retriever searching 'streaming Transformer latency' "
            "retrieves generic self-attention complexity chunks from 1706.03762v7 and chunking overhead from 2507.07955v2, "
            "missing that Moshi (2410.00037v1) is the streaming dialogue Transformer with 160ms theoretical / ~200ms practical latency."
        ),
    ),
    EvalQuestion(
        qid="Q11",
        question="In Table 3 of 'Attention Is All You Need', what happens to the development set BLEU score when the number of attention heads h is increased to 32 in variation (A)?",
        category="naive_trap",
        expected_paper_id="1706.03762v7",
        required_keywords=["25.4", "32", "quality", "drop"],
        trap_explanation=(
            "Trap 2 (Mid-Table Guillotining): Small fixed-size chunking splits Table 3 ('Variations on the Transformer architecture') "
            "separating the header row from variation (A) rows (h=1, 4, 16, 32), causing a naive retriever to lose the BLEU column mapping."
        ),
    ),
    EvalQuestion(
        qid="Q12",
        question="Why does the Dynamic Chunking paper argue that heuristic BPE tokenization degrades character-level robustness and non-Latin language efficiency?",
        category="naive_trap",
        expected_paper_id="2507.07955v2",
        required_keywords=["bpe", "byte", "heuristic", "chunk"],
        trap_explanation=(
            "Trap 3 (Baseline vs. Proposed Method Confusion): Both Moshi (which uses SentencePiece/text tokens) and Attention (which uses BPE/word-piece) "
            "mention BPE positively as their setup, whereas Dynamic Chunking critiques BPE as a brittle baseline."
        ),
    ),

    # --- PART 3: Unanswerable / Out-of-Scope Questions (Must Refuse!) ---
    EvalQuestion(
        qid="Q13",
        question="What is the exact GPU memory footprint in gigabytes when fine-tuning the Moshi 7B model using 4-bit QLoRA rank 16?",
        category="unanswerable",
        expected_paper_id=None,
        required_keywords=[],
    ),
    EvalQuestion(
        qid="Q14",
        question="According to the papers, how many layers and attention heads are used in OpenAI's GPT-4 architecture?",
        category="unanswerable",
        expected_paper_id=None,
        required_keywords=[],
    ),
    EvalQuestion(
        qid="Q15",
        question="What Word Error Rate (WER) does the Mimi codec achieve on the LibriSpeech test-other benchmark when paired with Whisper-large-v3 as the decoder?",
        category="unanswerable",
        expected_paper_id=None,
        required_keywords=[],
    ),
    EvalQuestion(
        qid="Q16",
        question="In 'Dynamic Chunking', what are the exact hyperparameter values used to train the H-Net vision backbone on ImageNet-22K object detection?",
        category="unanswerable",
        expected_paper_id=None,
        required_keywords=[],
    ),
]