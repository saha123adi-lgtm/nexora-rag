"""Shared helpers for the Nexora RAG project.

Contains: file parsers (PDF, DOCX, MD, TXT), section-aware chunking,
the embedding backends, and the role-based access filter.
"""
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
META_CSV = DATA_DIR / "metadata" / "documents_metadata.csv"
USERS_JSON = DATA_DIR / "metadata" / "users.json"
QA_CSV = DATA_DIR / "evaluation" / "qa_testset.csv"

DEFAULT_DB_DIR = ROOT / "chroma_db"
DEFAULT_COLLECTION = "nexora_kb"
DEFAULT_DUMP = ROOT / "chunks_dump.jsonl"

# 'employee' needs no flag of its own: every employee can read documents marked role_all.
ROLES = ["hr", "manager", "finance", "security", "sales", "admin"]

MINILM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COMPANY_PREFIX = "Nexora Technologies Pvt. Ltd."


# ============================================================== PARSING
def parse_pdf(path):
    """Return the document as a list of text lines. Tables are rebuilt as 'a | b | c' rows."""
    import pdfplumber

    lines = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            tables = sorted(page.find_tables(), key=lambda t: t.bbox[1])
            y = 0.0

            def segment(y0, y1):
                if y1 - y0 < 2:
                    return []
                return (page.within_bbox((0, y0, page.width, y1)).extract_text() or "").splitlines()

            for t in tables:
                _, top, _, bottom = t.bbox
                lines += segment(y, top)
                for row in t.extract():
                    lines.append(" | ".join(" ".join((c or "").split()) for c in row))
                y = bottom
            lines += segment(y, page.height)
    # drop the page footer that every page carries
    return [ln for ln in lines if not (ln.startswith(COMPANY_PREFIX) and " | " in ln)]


def parse_docx(path):
    from docx import Document
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    doc = Document(str(path))
    lines = []
    for child in doc.element.body.iterchildren():
        tag = child.tag.split("}")[1]
        if tag == "p":
            p = Paragraph(child, doc)
            text = p.text.strip()
            if not text:
                continue
            lines.append("- " + text if p.style.name == "List Bullet" else text)
        elif tag == "tbl":
            for row in Table(child, doc).rows:
                lines.append(" | ".join(" ".join(c.text.split()) for c in row.cells))
    return lines


def parse_md(path):
    lines = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        ln = raw.strip()
        if not ln:
            continue
        if re.fullmatch(r"\|?[\s:\-|]+\|?", ln) and "-" in ln:  # table separator row
            continue
        if ln.startswith("|"):
            ln = " | ".join(c.strip() for c in ln.strip("|").split("|"))
        ln = ln.replace("**", "")
        lines.append(ln)
    return lines


def parse_txt(path):
    lines = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        ln = raw.rstrip()
        if not ln.strip() or re.fullmatch(r"[-=+\s]+", ln):
            continue
        ln = re.sub(r"\s*\|\s*", " | ", ln.strip()) if "|" in ln else ln.strip()
        lines.append(ln)
    return lines


PARSERS = {"pdf": parse_pdf, "docx": parse_docx, "md": parse_md, "txt": parse_txt}


# ============================================================== SECTIONS + CHUNKS
HEADING_RE = re.compile(r"^(?:#{1,3}\s*)?(\d{1,2})\.\s+(\S.*)$")


def split_sections(lines):
    """Split lines into [(heading, [body lines])]. Headings look like '3. Something' and must be
    numbered in sequence (1, 2, 3 ...), which stops stray numbered text being mistaken for a heading.
    Anything before section 1 (title and metadata lines) is dropped: it lives in the metadata instead."""
    sections, expected = [], 1
    for ln in lines:
        m = HEADING_RE.match(ln)
        if m and int(m.group(1)) == expected:
            title = m.group(2).strip()
            if title.isupper():
                title = title.title()
            sections.append((f"{expected}. {title}", []))
            expected += 1
        elif sections:
            sections[-1][1].append(ln)
    return sections


def _split_long(text, max_chars, overlap):
    """Split on line boundaries (so table rows stay whole), carrying `overlap` characters forward."""
    chunks, cur, cur_len = [], [], 0
    for ln in text.split("\n"):
        if cur and cur_len + len(ln) + 1 > max_chars:
            chunks.append("\n".join(cur))
            tail, tl = [], 0
            for prev in reversed(cur):
                if tl + len(prev) + 1 > overlap:
                    break
                tail.insert(0, prev)
                tl += len(prev) + 1
            cur, cur_len = tail, tl
        cur.append(ln)
        cur_len += len(ln) + 1
    if cur:
        chunks.append("\n".join(cur))
    return chunks


def chunk_sections(sections, max_chars=1800, overlap=200, min_chars=250):
    """Return [(section_label, chunk_text)].

    1. Tiny sections are merged with the one after them.
    2. A section that fits in max_chars becomes one chunk.
    3. A longer section is split on line boundaries with overlap.
    """
    merged = []
    for heading, body in sections:
        text = heading + "\n" + "\n".join(body)
        if merged and len(merged[-1][1]) < min_chars and len(merged[-1][1]) + len(text) + 2 <= max_chars:
            mh, mt = merged[-1]
            merged[-1] = (f"{mh} / {heading}", mt + "\n\n" + text)
        else:
            merged.append((heading, text))

    out = []
    for label, text in merged:
        if len(text) <= max_chars:
            out.append((label, text))
        else:
            for i, piece in enumerate(_split_long(text, max_chars, overlap)):
                out.append((label, piece if i == 0 else f"{label} (continued)\n{piece}"))
    return out


# ============================================================== EMBEDDINGS
class Embedder:
    """backend 'minilm'  : all-MiniLM-L6-v2 via sentence-transformers (the real one; needs internet once).
    backend 'hashing' : offline keyword-hashing vectors. For testing the pipeline without internet only;
                         it is lexical, not semantic, so do not use it for your final results."""

    def __init__(self, backend=None):
        self.backend = backend or os.environ.get("EMBEDDING_BACKEND", "minilm")
        if self.backend == "minilm":
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(MINILM_MODEL)
        elif self.backend == "hashing":
            from sklearn.feature_extraction.text import HashingVectorizer

            self.model = HashingVectorizer(n_features=2048, alternate_sign=False, norm="l2",
                                           ngram_range=(1, 2), stop_words="english")
        else:
            raise ValueError(f"Unknown embedding backend: {self.backend}")

    def encode(self, texts):
        if self.backend == "minilm":
            return self.model.encode(texts, batch_size=32, normalize_embeddings=True,
                                     show_progress_bar=False).tolist()
        return self.model.transform(texts).toarray().tolist()


# ============================================================== ACCESS CONTROL
def role_flags(allowed_roles):
    """'hr;admin' -> {'role_all': False, 'role_hr': True, ...}. Chroma metadata cannot hold lists,
    so each role gets its own boolean."""
    allowed = {r.strip() for r in allowed_roles.split(";")}
    flags = {"role_all": "all" in allowed}
    for r in ROLES:
        flags[f"role_{r}"] = "all" in allowed or r in allowed
    return flags


def role_filter(user_roles):
    """Chroma 'where' filter: a chunk is visible if it is open to all, or to any of the user's roles."""
    conds = [{"role_all": True}] + [{f"role_{r}": True} for r in user_roles if r in ROLES]
    return conds[0] if len(conds) == 1 else {"$or": conds}


def can_access(allowed_roles, user_roles):
    """True if a document with this allowed_roles string is visible to a user with these roles."""
    allowed = {r.strip() for r in allowed_roles.split(";")}
    return "all" in allowed or bool(allowed & set(user_roles))


def query_chunks(col, query_vec, roles, k=5):
    """Nearest chunks that `roles` may see. Fetches extra candidates and sorts by (score, chunk id)
    so chunks that tie on score come back in the same order every run."""
    n = max(k * 3, k + 10)
    res = col.query(query_embeddings=[query_vec], n_results=n, where=role_filter(roles),
                    include=["documents", "metadatas", "distances"])
    hits = [{"id": cid, "score": 1.0 - dist, "text": text, "meta": meta}
            for cid, text, meta, dist in zip(res["ids"][0], res["documents"][0],
                                             res["metadatas"][0], res["distances"][0])]
    hits.sort(key=lambda h: (-round(h["score"], 6), h["id"]))
    return hits[:k]


def count_visible(col, roles):
    """How many chunks in the collection these roles are allowed to see."""
    return len(col.get(where=role_filter(roles), include=[])["ids"])


def load_docs_meta():
    import csv

    with open(META_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_users():
    import json

    with open(USERS_JSON, encoding="utf-8") as f:
        return json.load(f)


def get_client(db_dir):
    import chromadb

    return chromadb.PersistentClient(path=str(db_dir))
