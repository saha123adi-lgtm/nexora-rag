"""Search the vector database, with role-based access control.

Usage:
    python search.py "How many sick leave days are there?"                # as a plain employee
    python search.py "What is the CTC band for L5?" --role hr             # as HR
    python search.py "..." --user U003                                    # as a sample user from users.json
    python search.py                                                      # interactive mode
    python search.py --eval                                               # run the whole test set and print metrics
"""
import argparse
import csv
import json
import re
from collections import defaultdict

import rag_common as rc


def open_collection(db_dir, name):
    client = rc.get_client(db_dir)
    col = client.get_collection(name, embedding_function=None)
    backend = (col.metadata or {}).get("embedding_backend", "minilm")
    return col, rc.Embedder(backend)  # always embed queries with the same model used at ingestion


def search(col, embedder, query, roles, k=5):
    return rc.query_chunks(col, embedder.encode([query])[0], roles, k)


def print_hits(hits, width=200):
    if not hits:
        print("  (no accessible chunks found)")
    for i, h in enumerate(hits, 1):
        m = h["meta"]
        body = " ".join(h["text"].split("\n", 1)[1].split())  # drop the title line, flatten
        print(f"  {i}. [{h['score']:.3f}] {m['doc_id']} | {m['section']} | {m['filename']}")
        print(f"     {body[:width]}{'...' if len(body) > width else ''}")


def norm(s):
    s = re.sub(r"\s*\|\s*", " | ", s)
    return " ".join(s.split()).lower()


# ------------------------------------------------------------------ evaluation
def run_eval(col, embedder, k, out_csv):
    with open(rc.QA_CSV, newline="", encoding="utf-8") as f:
        qa = list(csv.DictReader(f))

    rows = []
    for q in qa:
        hits = search(col, embedder, q["question"], [q["asking_role"]], k)
        texts = [norm(h["text"]) for h in hits]
        ev = norm(q["evidence_snippet"]) if q["evidence_snippet"] else ""
        src = set(q["source_doc_ids"].split(";")) if q["source_doc_ids"] else set()
        got_docs = [h["meta"]["doc_id"] for h in hits]
        rank = next((i + 1 for i, t in enumerate(texts) if ev and ev in t), 0)
        rows.append({
            "qid": q["qid"], "category": q["category"], "role": q["asking_role"], "question": q["question"],
            "top1_score": hits[0]["score"] if hits else 0.0,
            "evidence_rank": rank, "doc_hit": bool(src & set(got_docs)),
            "leak": bool(src & set(got_docs)) or bool(ev and any(ev in t for t in texts)),
            "retrieved_docs": ";".join(got_docs),
        })

    answerable = [r for r in rows if r["category"] in ("factual", "paraphrase", "multi_hop", "access_allowed")]
    print(f"\n=== Retrieval evaluation (k={k}, {len(rows)} questions, backend={embedder.backend}) ===\n")
    print(f"{'category':16} {'n':>3} {'hit@1':>7} {'hit@'+str(k):>7} {'MRR':>7} {'doc hit':>8}")
    by_cat = defaultdict(list)
    for r in answerable:
        by_cat[r["category"]].append(r)
    for cat, rs in list(by_cat.items()) + [("ALL ANSWERABLE", answerable)]:
        n = len(rs)
        h1 = sum(r["evidence_rank"] == 1 for r in rs) / n
        hk = sum(r["evidence_rank"] > 0 for r in rs) / n
        mrr = sum(1 / r["evidence_rank"] for r in rs if r["evidence_rank"]) / n
        dh = sum(r["doc_hit"] for r in rs) / n
        print(f"{cat:16} {n:3d} {h1:7.1%} {hk:7.1%} {mrr:7.3f} {dh:8.1%}")

    denied = [r for r in rows if r["category"] == "access_denied"]
    leaks = [r for r in denied if r["leak"]]
    print(f"\nAccess control: {len(denied)} access_denied questions, leaked content in {len(leaks)} of them")
    for r in leaks:
        print(f"  LEAK {r['qid']} ({r['role']}): {r['question']}  -> {r['retrieved_docs']}")

    unans = [r for r in rows if r["category"] == "unanswerable"]
    avg = lambda xs: sum(xs) / len(xs) if xs else 0
    print(f"\nTop-1 similarity: answerable avg {avg([r['top1_score'] for r in answerable]):.3f}, "
          f"unanswerable avg {avg([r['top1_score'] for r in unans]):.3f}")
    print("(use the gap to choose a score threshold below which the assistant answers 'I don't know')")

    misses = [r for r in answerable if r["evidence_rank"] == 0]
    print(f"\nMisses ({len(misses)}): evidence not found in the top {k}")
    for r in misses:
        print(f"  {r['qid']} [{r['category']}] {r['question']}\n      retrieved: {r['retrieved_docs']}")

    if out_csv:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nPer-question results written to {out_csv}")


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("question", nargs="?")
    ap.add_argument("--role", default="employee", help="comma-separated roles, e.g. hr or employee,manager")
    ap.add_argument("--user", help="user id from users.json (overrides --role)")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--db-dir", default=str(rc.DEFAULT_DB_DIR))
    ap.add_argument("--collection", default=rc.DEFAULT_COLLECTION)
    ap.add_argument("--eval", action="store_true", help="run the full test set")
    ap.add_argument("--out", help="with --eval: write per-question results to this CSV")
    args = ap.parse_args()

    col, embedder = open_collection(args.db_dir, args.collection)
    roles = [r.strip() for r in args.role.split(",")]
    if args.user:
        users = json.load(open(rc.USERS_JSON, encoding="utf-8"))
        u = next(x for x in users if x["user_id"] == args.user)
        roles = u["roles"]
        print(f"Acting as {u['name']} ({u['job_title']}), roles: {roles}")

    if args.eval:
        run_eval(col, embedder, args.k, args.out)
    elif args.question:
        print(f"Q: {args.question}   (roles: {roles})")
        print_hits(search(col, embedder, args.question, roles, args.k))
    else:
        print(f"Interactive mode, roles: {roles}. Empty line to quit.")
        while True:
            q = input("\nQuestion> ").strip()
            if not q:
                break
            print_hits(search(col, embedder, q, roles, args.k))


if __name__ == "__main__":
    main()
