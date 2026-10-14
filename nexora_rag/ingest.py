"""Build the vector database: parse -> chunk -> embed -> store in ChromaDB.

Usage:
    python ingest.py                                   # default settings
    python ingest.py --max-chars 1000 --collection nexora_c1000   # a variant for chunk-size experiments
"""
import argparse
import json
import sys
import time

import rag_common as rc


def build_index(db_dir=rc.DEFAULT_DB_DIR, collection=rc.DEFAULT_COLLECTION, backend=None,
                max_chars=1800, overlap=200, min_chars=250, dump=rc.DEFAULT_DUMP, verbose=True):
    """Parse every document, chunk, embed and store. Returns the number of chunks stored."""
    log = print if verbose else (lambda *a, **k: None)
    t0 = time.time()
    docs = rc.load_docs_meta()
    log(f"Found {len(docs)} documents in metadata\n")

    ids, texts, metas = [], [], []
    log(f"{'doc_id':8} {'fmt':5} {'chunks':>6} {'avg chars':>10}  file")
    for d in docs:
        path = rc.DATA_DIR / d["path"]
        lines = rc.PARSERS[d["format"]](path)
        sections = rc.split_sections(lines)
        if not sections:
            print(f"WARNING: no sections found in {d['filename']}", file=sys.stderr)
            continue
        chunks = rc.chunk_sections(sections, max_chars, overlap, min_chars)
        flags = rc.role_flags(d["allowed_roles"])
        for i, (label, body) in enumerate(chunks):
            ids.append(f"{d['doc_id']}-{i:03d}")
            # the document title is prepended so every chunk carries its own context
            texts.append(f"{d['title']} ({d['doc_id']})\n{body}")
            metas.append({
                "doc_id": d["doc_id"], "title": d["title"], "department": d["department"],
                "filename": d["filename"], "format": d["format"], "section": label,
                "chunk_index": i, "access_level": d["access_level"], "allowed_roles": d["allowed_roles"],
                "version": d["version"], **flags,
            })
        avg = sum(len(b) for _, b in chunks) // len(chunks)
        log(f"{d['doc_id']:8} {d['format']:5} {len(chunks):6d} {avg:10d}  {d['filename']}")

    log(f"\nTotal chunks: {len(ids)}")

    embedder = rc.Embedder(backend)
    log(f"Embedding with backend '{embedder.backend}' ...")
    vectors = embedder.encode(texts)
    log(f"Vector dimension: {len(vectors[0])}")

    client = rc.get_client(db_dir)
    try:
        client.delete_collection(collection)
    except Exception:
        pass
    col_meta = {"embedding_backend": embedder.backend, "max_chars": max_chars,
                "overlap": overlap, "min_chars": min_chars}
    try:
        col = client.create_collection(collection, embedding_function=None, metadata=col_meta,
                                       configuration={"hnsw": {"space": "cosine"}})
    except TypeError:  # older chromadb versions
        col = client.create_collection(collection, embedding_function=None,
                                       metadata={**col_meta, "hnsw:space": "cosine"})
    col.add(ids=ids, documents=texts, embeddings=vectors, metadatas=metas)

    if dump:
        with open(dump, "w", encoding="utf-8") as f:
            for i, t, m in zip(ids, texts, metas):
                f.write(json.dumps({"id": i, "text": t, **m}, ensure_ascii=False) + "\n")
        log(f"Chunk dump written to {dump}")

    log(f"\nStored {col.count()} chunks in collection '{collection}' at {db_dir}")
    log(f"Done in {time.time() - t0:.1f}s")
    return col.count()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db-dir", default=str(rc.DEFAULT_DB_DIR), help="folder where ChromaDB stores its files")
    ap.add_argument("--collection", default=rc.DEFAULT_COLLECTION)
    ap.add_argument("--backend", default=None, help="minilm (default) or hashing (offline test only)")
    ap.add_argument("--max-chars", type=int, default=1800, help="max characters per chunk (about 450 tokens)")
    ap.add_argument("--overlap", type=int, default=200, help="characters carried into the next chunk when a section is split")
    ap.add_argument("--min-chars", type=int, default=250, help="sections shorter than this are merged with the next one")
    ap.add_argument("--dump", default=str(rc.DEFAULT_DUMP), help="write all chunks to this .jsonl file for inspection")
    a = ap.parse_args()
    build_index(a.db_dir, a.collection, a.backend, a.max_chars, a.overlap, a.min_chars, a.dump)


if __name__ == "__main__":
    main()
