import json
import re
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Allow imports from the project root (the-relay/) when running this file directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import fitz
import numpy as np
import tiktoken
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

from config import (
    ARVIX_PAPERS,
    PAPERS_DIR,
    INDEX_DIR,
    RETRIEVAL_SETTINGS,
    RetrievalSetting,
)

#REGEX to catch standard academic section headers like "1 Introduction",
SECTION_HEADER_RE = re.compile(
    r"^(?:(?:\d+(?:\.\d+)*)|(?:[A-Z](?:\.\d+)*))\s+[A-Z][A-Za-z0-9\s\-:,()]{2,70}$"
)

@dataclass
class Chunk:
    chunk_id: str
    paper_id: str
    paper_title: str
    page_num: int
    section: str
    text: str
    raw_text: str
    token_count: int

class LibraryIndexer:
    def __init__(self, load_models: bool = True):
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.embedder: Optional[SentenceTransformer] = None
        self.reranker: Optional[CrossEncoder] = None
        if load_models:
            self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
            self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        self._indices: Dict[int, Dict] = {}

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, disallowed_special=()))
    
    def _clean_block_text(self, text:str) -> str:
        text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return "".join(lines)

    def extract_structure_blocks(self, pdf_path: Path, paper_id: str, paper_title: str) -> List[Dict]:
        """uses  Pymypd block extraction and regex (which repects two_cloumn reading order)
        and tracks the active section header across the pages"""

        doc = fitz.open(pdf_path)
        current_section = "Abstract/ Introduction"
        structured_blocks = []

        for page_idx, page in enumerate(doc):
            page_num = page_idx + 1

            blocks = page.get_text("blocks", sort=True)
            for b in blocks:
                block_type = b[6]
                if block_type !=0:
                    continue
                raw_block = b[4].strip()
                if not raw_block or len(raw_block) < 4:
                    continue

                first_line = raw_block.splitlines()[0].strip()
                if SECTION_HEADER_RE.match(first_line):
                    current_section = first_line

                cleaned = self._clean_block_text(raw_block)
                if len(cleaned) < 15:
                    continue

                structured_blocks.append({
                    "paper_id": paper_id,
                    "paper_title": paper_title,
                    "page_num": page_num,
                    "section": current_section,
                    "text": cleaned,
                })
        doc.close()
        return structured_blocks

    def build_chunks_for_size(self, target_tokens: int, overlap_tokens: int) -> List[Chunk]:
        chunks: List[Chunk] = []

        for paper_id, meta in ARVIX_PAPERS.items():
            pdf_path = PAPERS_DIR/meta["filename"]
            if not pdf_path.exists():
                raise FileNotFoundError(f"Missing file {pdf_path} for paper {paper_id}")

            blocks = self.extract_structure_blocks(pdf_path, paper_id, meta["short_title"])
            
            current_sentences: List[Tuple[str, int, str]] = []
            for blk in blocks:
                sents = re.split(r"(?<=[.!?])\s+", blk["text"])     
                for s in sents:
                    if s.strip():
                        current_sentences.append((s.strip(), blk["page_num"], blk["section"]))    

            idx = 0
            chunk_counter = 0
            while idx < len(current_sentences):
                buf_text = []
                buf_tokens = 0
                start_page = current_sentences[idx][1]
                start_section = current_sentences[idx][2]

                end_idx = idx
                while end_idx < len(current_sentences):
                    sent_text, _, _ = current_sentences[end_idx]
                    send_toks = self.count_tokens(sent_text)
                    if buf_tokens + send_toks > target_tokens and buf_tokens:
                        break
                    buf_text.append(sent_text)
                    buf_tokens += send_toks
                    end_idx += 1

                raw_body = "".join(buf_text)
                header = (
                    f"[Paper: {meta['short_title']} (arXiv:{paper_id}) | "
                    f"Page: {start_page} | Section: {start_section}]\n"
                )
                full_text = header + raw_body
                full_toks = self.count_tokens(full_text)

                chunk_counter += 1
                chunks.append(
                    Chunk(
                        chunk_id=f"{paper_id}_s{target_tokens}_c{chunk_counter:04d}",
                        paper_id=paper_id,
                        paper_title=meta["short_title"],
                        page_num=start_page,
                        section=start_section,
                        text=full_text,
                        raw_text=raw_body,
                        token_count=full_toks,
                    )
                )

                if end_idx >= len(current_sentences):
                    break

                # Step back by overlap_tokens for the next chunk
                back_tokens = 0
                next_idx = end_idx
                while next_idx > idx + 1 and back_tokens < overlap_tokens:
                    next_idx -= 1
                    back_tokens += self.count_tokens(current_sentences[next_idx][0])
                idx = max(idx + 1, next_idx)

        return chunks

    def build_all_indices(self, force: bool = False) -> None:
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        unique_sizes = {
            (cfg.chunk_size_tokens, cfg.chunk_overlap_tokens)
            for cfg in RETRIEVAL_SETTINGS.values()
        }

        for chunk_size, overlap in sorted(unique_sizes):
            chunks_file = INDEX_DIR / f"chunks_{chunk_size}.json"
            emb_file = INDEX_DIR / f"embeddings_{chunk_size}.npy"

            if chunks_file.exists() and emb_file.exists() and not force:
                print(f"[INDEX OK] Size {chunk_size} already cached on disk.")
                continue

            print(f"[BUILDING INDEX] chunk_size={chunk_size}, overlap={overlap} ...")
            chunks = self.build_chunks_for_size(chunk_size, overlap)
            texts = [c.text for c in chunks]

            embeddings = self.embedder.encode(
                texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True
            )

            chunks_file.write_text(
                json.dumps([asdict(c) for c in chunks], indent=2), encoding="utf-8"
            )
            np.save(emb_file, embeddings)
            print(
                f"[SAVED INDEX] chunk_size={chunk_size}: {len(chunks)} chunks, "
                f"avg tokens={np.mean([c.token_count for c in chunks]):.1f}"
            )

    def _get_loaded_index(self, chunk_size: int) -> Dict:
        if chunk_size not in self._indices:
            chunks_file = INDEX_DIR / f"chunks_{chunk_size}.json"
            emb_file = INDEX_DIR / f"embeddings_{chunk_size}.npy"
            raw_list = json.loads(chunks_file.read_text(encoding="utf-8"))
            chunks = [Chunk(**item) for item in raw_list]
            embeddings = np.load(emb_file)
            tokenized_corpus = [re.findall(r"\w+", c.text.lower()) for c in chunks]
            bm25 = BM25Okapi(tokenized_corpus)
            self._indices[chunk_size] = {
                "chunks": chunks,
                "embeddings": embeddings,
                "bm25": bm25,
            }
        return self._indices[chunk_size]

    def search(
        self,
        query: str,
        setting_name: str,
        paper_filter: Optional[str] = None,
        degrade_for_rate_limit: bool = False,
    ) -> Tuple[List[Chunk], float]:
        """
        Executes Hybrid Search (Reciprocal Rank Fusion of BM25 + Dense Cosine)
        followed by optional Cross-Encoder reranking.
        Returns (selected_chunks, latency_ms).
        """
        t0 = time.perf_counter()
        cfg: RetrievalSetting = RETRIEVAL_SETTINGS[setting_name]
        idx_data = self._get_loaded_index(cfg.chunk_size_tokens)
        chunks: List[Chunk] = idx_data["chunks"]
        embeddings: np.ndarray = idx_data["embeddings"]
        bm25: BM25Okapi = idx_data["bm25"]

        # Filter candidate indices if router specified a paper
        valid_indices = [
            i for i, c in enumerate(chunks)
            if (not paper_filter or paper_filter in c.paper_id or paper_filter.lower() in c.paper_title.lower())
        ]
        if not valid_indices:
            valid_indices = list(range(len(chunks)))

        # 1. BM25 Lexical Scores
        query_tokens = re.findall(r"\w+", query.lower())
        bm25_scores = bm25.get_scores(query_tokens)

        # 2. Dense Embedding Cosine Scores
        q_emb = self.embedder.encode([query], normalize_embeddings=True)[0]
        dense_scores = embeddings @ q_emb

        # 3. Reciprocal Rank Fusion (RRF) over valid_indices
        bm25_ranked = sorted(valid_indices, key=lambda i: bm25_scores[i], reverse=True)
        dense_ranked = sorted(valid_indices, key=lambda i: dense_scores[i], reverse=True)

        rrf_scores: Dict[int, float] = {i: 0.0 for i in valid_indices}
        for rank, doc_idx in enumerate(bm25_ranked):
            rrf_scores[doc_idx] += 1.0 / (60 + rank + 1)
        for rank, doc_idx in enumerate(dense_ranked):
            rrf_scores[doc_idx] += 1.0 / (60 + rank + 1)

        # Determine fetch_k and final_k (degrade gracefully if rate limit is close!)
        fetch_k = 2 if degrade_for_rate_limit else cfg.initial_fetch_k
        final_k = 2 if degrade_for_rate_limit else cfg.final_top_k
        use_rerank = False if degrade_for_rate_limit else cfg.use_reranker

        top_candidates_idx = sorted(valid_indices, key=lambda i: rrf_scores[i], reverse=True)[:fetch_k]
        candidate_chunks = [chunks[i] for i in top_candidates_idx]

        # 4. Optional Cross-Encoder Reranking
        if use_rerank and len(candidate_chunks) > 1 and self.reranker is not None:
            pairs = [[query, c.text] for c in candidate_chunks]
            rerank_scores = self.reranker.predict(pairs)
            ranked_pairs = sorted(
                zip(candidate_chunks, rerank_scores), key=lambda x: x[1], reverse=True
            )
            selected = [c for c, _ in ranked_pairs[:final_k]]
        else:
            selected = candidate_chunks[:final_k]

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return selected, elapsed_ms


if __name__ == "__main__":
    indexer = LibraryIndexer(load_models=True)
    indexer.build_all_indices(force=True)

    # Quick sanity test across all 3 settings
    test_q = "What is the frame rate and latency of the Mimi audio codec in Moshi?"
    for s_name in RETRIEVAL_SETTINGS:
        results, ms = indexer.search(test_q, setting_name=s_name)
        total_toks = sum(c.token_count for c in results)
        print(f"\n--- Setting: {s_name} ({ms:.1f} ms, {total_toks} total chunk tokens) ---")
        print(f"Top hit: {results[0].chunk_id} | Page {results[0].page_num} | {results[0].section}")
        