import fitz
from config import ARVIX_PAPERS, PAPERS_DIR

def verify_local_papers() -> None:
    for paper_id, meta in ARVIX_PAPERS.items():
        pdf_path = PAPERS_DIR / meta['filename']
        if not pdf_path.exists():
            raise FileNotFoundError(f"Paper {paper_id} not found")

        doc = fitz.open(pdf_path)
        first_page_text = doc[0].get_text("text")
        size_kb = pdf_path.stat().st_size / 1024

        #checking if we extacthte arXiv ID and version
        version_matched = paper_id in first_page_text
        status = "version matched" if version_matched else "check version"
        print(
            f"[OK] {meta['filename']:20s} -> {paper_id} ({meta['short_title']}) | "
            f"{len(doc):2d} pages | {size_kb:7.1f} KB | {status}"
        )
        doc.close()

if __name__ == "__main__":
    verify_local_papers()