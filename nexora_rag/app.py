"""Nexora Enterprise RAG Assistant - Streamlit UI with a live pipeline trace.

Run locally:   streamlit run app.py
"""
import contextlib
import io
import json
import os
import time

import pandas as pd
import streamlit as st

import rag_common as rc
import rag_core as core

st.set_page_config(page_title="Nexora Enterprise RAG Assistant", page_icon="🔎", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 1.6rem; max-width: 1500px;}
[data-testid="stMetricValue"] {font-size: 1.5rem;}
</style>
""", unsafe_allow_html=True)

DB_DIR = os.environ.get("RAG_DB_DIR", str(rc.DEFAULT_DB_DIR))
COLLECTION = os.environ.get("RAG_COLLECTION", rc.DEFAULT_COLLECTION)
RESULTS_DIR = rc.ROOT / "results"


# ----------------------------------------------------------------- setup
@st.cache_resource(show_spinner="Loading the vector database and embedding model (first start can take a minute)...")
def get_engine():
    try:
        return core.Engine(DB_DIR, COLLECTION)
    except Exception:
        # No index yet (fresh deployment): build it from the documents, then open it.
        import ingest

        ingest.build_index(DB_DIR, COLLECTION, verbose=False)
        return core.Engine(DB_DIR, COLLECTION)


def secret(name):
    try:
        return st.secrets[name]
    except Exception:
        return os.environ.get(name, "")


try:
    engine = get_engine()
except Exception as e:
    st.error(f"Could not start the engine: {e}")
    st.info("If this is the first run, the embedding model must be downloaded (internet needed). "
            "Run `python ingest.py` in a terminal to see the full error.")
    st.stop()

USERS = rc.load_users()
QA_ROWS = pd.read_csv(rc.QA_CSV).fillna("")
st.session_state.setdefault("messages", [])
st.session_state.setdefault("last_trace", None)


def user_label(u):
    return f"{u['name']} - {u['job_title']}"


# ----------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Demo controls")
    user = st.selectbox("Acting as", USERS, format_func=user_label)
    roles = user["roles"]
    st.caption("Roles: " + ", ".join(roles))

    st.divider()
    provider_label = st.selectbox("Answer generation", list(core.PROVIDERS))
    provider = core.PROVIDERS[provider_label]
    model = st.text_input("Model", core.DEFAULT_MODELS[provider], key=f"model_{provider}", disabled=provider == "none")
    api_key, host, base_url = "", "http://localhost:11434", ""
    if provider in core.KEY_ENV:
        preset = secret(core.KEY_ENV[provider])
        if preset:
            api_key = preset
            st.success("API key found in secrets / environment")
        else:
            api_key = st.text_input("API key", type="password", key=f"key_{provider}",
                                    help="Only kept in this browser session. Never saved to disk.")
    if provider == "gemini":
        st.caption("Free key: Google AI Studio. The free tier may use your prompts to improve Google's products, "
                   "so use only non-sensitive data (this demo's data is fictional).")
    if provider == "openai_compat":
        base_url = st.text_input("Base URL", core.BASE_URLS["openai_compat"], key="base_url_compat",
                                 help="Groq: https://api.groq.com/openai/v1 · OpenAI: https://api.openai.com/v1 · "
                                      "local Ollama: http://localhost:11434/v1")
        st.caption("Check the provider's current free-tier terms before relying on it.")
    if provider == "ollama":
        host = st.text_input("Ollama host", host)

    st.divider()
    k = st.slider("Chunks to retrieve (k)", 1, 10, 5)
    threshold = st.slider("Answer threshold (min. top score)", 0.0, 1.0, float(engine.default_threshold()), 0.01,
                          help="If the best chunk scores below this, the assistant says it doesn't know "
                               "and the LLM is not called. Tune it with the Evaluation tab.")
    slowmo = st.checkbox("Slow motion for presentation", True,
                         help="Pauses between steps so an audience can follow. Timings shown are the real ones.")
    pace = st.slider("Pause between steps (s)", 0.2, 2.0, 0.7, 0.1, disabled=not slowmo)

    st.divider()
    st.subheader("Sample questions")
    pool = QA_ROWS[QA_ROWS["asking_role"].isin(roles)].drop_duplicates("question")
    labels = {r.question: f"[{r.category}] {r.question}" for r in pool.itertuples()}
    sample = st.selectbox("Pick one", list(labels), format_func=labels.get)
    if st.button("Ask this question", width="stretch"):
        st.session_state["pending"] = sample
    if st.button("Clear chat", width="stretch"):
        st.session_state["messages"] = []
        st.session_state["last_trace"] = None
        st.rerun()

llm = core.LLMConfig(provider=provider, model=model, api_key=api_key, host=host, base_url=base_url)


# ----------------------------------------------------------------- trace rendering
def card(title, ms=None):
    c = st.container(border=True)
    c.markdown(f"**{title}**" + (f" &nbsp;·&nbsp; `{ms} ms`" if ms is not None else ""))
    return c


def render_stage(name, d, trace_roles):
    if name == "embed":
        with card("① Embed the question", d["ms"]):
            st.write(f"The model **{d['model']}** turns the question into a vector of **{d['dim']} numbers**. "
                     "Questions and passages with similar meaning end up close together.")
            st.code(f"{d['preview']} ...", language="text")
    elif name == "access":
        with card(f"② Access control · roles: {', '.join(trace_roles)}", d["ms"]):
            c1, c2 = st.columns(2)
            c1.metric("Chunks this user can search", f"{d['visible_chunks']} / {d['total_chunks']}")
            c2.metric("Documents hidden", len(d["hidden_docs"]))
            if d["hidden_docs"]:
                st.warning("Hidden from this user: " + ", ".join(f"{h['doc_id']} ({h['title']})" for h in d["hidden_docs"]))
            else:
                st.success("No restrictions: this user can search every document.")
            st.caption("The filter runs inside the database query, so hidden chunks are never retrieved at all.")
            st.code(json.dumps(d["filter"]), language="json")
    elif name == "retrieve":
        with card(f"③ Vector search · top {d['k']} chunks", d["ms"]):
            if not d["hits"]:
                st.info("Nothing found.")
            for h in d["hits"]:
                st.progress(min(max(h["score"], 0.0), 1.0),
                            text=f"#{h['rank']} · similarity {h['score']:.3f} · {h['doc_id']} · {h['section']}")
                with st.expander(f"{h['title']} ({h['filename']}) - chunk text", expanded=False):
                    st.text(h["text"])
    elif name == "gate":
        with card("④ Is the best match good enough?"):
            c1, c2 = st.columns(2)
            c1.metric("Best similarity", f"{d['top_score']:.3f}")
            c2.metric("Threshold", f"{d['threshold']:.2f}")
            if d["passed"]:
                st.success("Above the threshold: continue to answer generation.")
            else:
                st.warning("Below the threshold: the assistant will say it doesn't know, and the LLM is not called.")
    elif name == "prompt":
        with card("⑤ Prompt sent to the LLM"):
            st.caption(f"About {d['approx_tokens']} tokens. The answer can only use the chunks inside this prompt.")
            with st.expander("System prompt (the rules)"):
                st.text(d["system"])
            with st.expander("User prompt (retrieved chunks + question)", expanded=True):
                st.text(d["user"])
    elif name == "generate":
        if d.get("skipped"):
            with card("⑥ Generate the answer"):
                st.info(d["reason"])
        else:
            who = "retrieval only" if d["provider"] == "none" else f"{d['provider']} · {d['model']}"
            with card(f"⑥ Generate the answer · {who}", d["ms"]):
                u = d.get("usage") or {}
                if u.get("input_tokens") is not None:
                    st.caption(f"Tokens: {u.get('input_tokens')} in, {u.get('output_tokens')} out")
                st.write(d["answer"])


def render_trace(trace):
    for name in ["embed", "access", "retrieve", "gate", "prompt", "generate"]:
        if name in trace["stages"]:
            render_stage(name, trace["stages"][name], trace["roles"])
    st.caption(f"Total time: {trace['total_ms']} ms")


def render_message(msg):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            st.caption("Sources: " + "  ·  ".join(f"📄 {s['doc_id']} {s['title']} ({s['filename']})" for s in msg["sources"]))
        if msg.get("note"):
            st.caption(msg["note"])


# ----------------------------------------------------------------- tabs
st.title("🔎 Nexora Enterprise RAG Assistant")
st.caption(f"Acting as **{user['name']}** ({', '.join(roles)}) · {engine.total_chunks} chunks · "
           f"embeddings: {engine.model_label} · answers: {provider_label}")
tab_chat, tab_access, tab_kb, tab_eval, tab_how = st.tabs(
    ["💬 Ask & trace", "🔐 Access control demo", "📚 Knowledge base", "📊 Evaluation", "🧭 How it works"])

# ============ chat + live trace
with tab_chat:
    left, right = st.columns([5, 6], gap="large")
    with left:
        st.subheader("Chat")
        for m in st.session_state["messages"]:
            render_message(m)
        new_user_ph = st.empty()
        new_assistant_ph = st.empty()
        typed = st.chat_input("Ask about HR, IT, security, finance or our products...")
    question = typed or st.session_state.pop("pending", None)

    with right:
        st.subheader("Live pipeline trace")
        if question:
            with new_user_ph.container():
                render_message({"role": "user", "content": question})
            waiting = {}

            def on_stage(name, data):
                if "ph" in waiting:
                    waiting.pop("ph").empty()
                render_stage(name, data, roles)
                if name == "prompt" and llm.provider != "none":
                    waiting["ph"] = st.empty()
                    waiting["ph"].info("⏳ Waiting for the model to answer...")
                if slowmo and name != "prompt":
                    time.sleep(pace)

            trace = engine.run(question, roles, k=k, threshold=threshold, llm=llm, on_stage=on_stage)
            if "ph" in waiting:
                waiting.pop("ph").empty()
            st.caption(f"Total time: {trace['total_ms']} ms")
            msg = {"role": "assistant", "content": trace["answer"], "sources": trace["sources"],
                   "note": f"Answered as {user['name']} · {trace['total_ms']} ms"}
            st.session_state["messages"] += [{"role": "user", "content": question}, msg]
            st.session_state["last_trace"] = trace
            with new_assistant_ph.container():
                render_message(msg)
        elif st.session_state["last_trace"]:
            render_trace(st.session_state["last_trace"])
        else:
            st.info("Ask a question (or pick a sample in the sidebar) and each step of the pipeline will appear here.")

# ============ access control demo
with tab_access:
    st.subheader("Same question, different users")
    st.write("Access rules are enforced inside the vector search. Compare what two users can retrieve for the same question "
             "(retrieval only, no LLM).")
    acc_pool = QA_ROWS[QA_ROWS["category"].isin(["access_denied", "access_allowed"])].drop_duplicates("question")
    preset_q = st.selectbox("Preset question", list(acc_pool["question"]), key="acc_preset")
    custom_q = st.text_input("...or type your own question", key="acc_custom")
    q_acc = custom_q.strip() or preset_q
    ca, cb = st.columns(2)
    default_a = next(i for i, u in enumerate(USERS) if u["roles"] == ["employee"])
    default_b = next(i for i, u in enumerate(USERS) if "hr" in u["roles"])
    ua = ca.selectbox("User A", USERS, index=default_a, format_func=user_label, key="acc_a")
    ub = cb.selectbox("User B", USERS, index=default_b, format_func=user_label, key="acc_b")
    for col, u in ((ca, ua), (cb, ub)):
        t = engine.run(q_acc, u["roles"], k=3, threshold=threshold, llm=core.LLMConfig("none"))
        acc, ret, gate = t["stages"]["access"], t["stages"]["retrieve"], t["stages"]["gate"]
        with col:
            with st.container(border=True):
                st.markdown(f"**{u['name']}** · roles: {', '.join(u['roles'])}")
                m1, m2 = st.columns(2)
                m1.metric("Searchable chunks", f"{acc['visible_chunks']} / {acc['total_chunks']}")
                m2.metric("Hidden documents", len(acc["hidden_docs"]))
                if gate["passed"]:
                    st.success(f"Would answer, best source: {ret['hits'][0]['doc_id']} ({ret['hits'][0]['score']:.3f})")
                else:
                    st.error(f"Would refuse: best score {gate['top_score']:.3f} is below the threshold {gate['threshold']:.2f}")
                st.dataframe(pd.DataFrame([{"rank": h["rank"], "doc": h["doc_id"], "section": h["section"],
                                            "score": h["score"]} for h in ret["hits"]]),
                             hide_index=True, width="stretch")

# ============ knowledge base
with tab_kb:
    st.subheader("Knowledge base")
    docs_df = pd.DataFrame(engine.docs)
    docs_df["access"] = docs_df["allowed_roles"].map(lambda r: "open to all" if r == "all" else "🔒 " + r.replace(";", ", "))
    docs_df["you can open"] = docs_df["allowed_roles"].map(lambda r: "yes" if rc.can_access(r, roles) else "no")
    c1, c2, c3 = st.columns(3)
    c1.metric("Documents", len(docs_df))
    c2.metric("Chunks", engine.total_chunks)
    c3.metric("Embedding dimension", len(engine.embedder.encode(["x"])[0]))
    st.dataframe(docs_df[["doc_id", "title", "department", "format", "access", "you can open", "filename", "version"]],
                 hide_index=True, width="stretch")
    pick = st.selectbox("Inspect the chunks of a document", engine.docs,
                        format_func=lambda d: f"{d['doc_id']} - {d['title']}")
    if not rc.can_access(pick["allowed_roles"], roles):
        st.warning(f"🔒 {user['name']} cannot open this document (allowed roles: {pick['allowed_roles'].replace(';', ', ')}).")
    else:
        got = engine.col.get(where={"doc_id": pick["doc_id"]}, include=["documents", "metadatas"])
        for cid, text, meta in sorted(zip(got["ids"], got["documents"], got["metadatas"]),
                                      key=lambda x: x[2]["chunk_index"]):
            with st.expander(f"{cid} · {meta['section']} · {len(text)} characters"):
                st.text(text)

# ============ evaluation
with tab_eval:
    st.subheader("Retrieval evaluation")
    st.write("Each test question has an exact evidence phrase. **hit@k** = the phrase appears in the top-k retrieved chunks; "
             "**MRR** rewards ranking it higher; **leaks** = restricted content shown to a user who should not see it.")
    RESULTS_DIR.mkdir(exist_ok=True)
    if st.button(f"Run the evaluation now on the live index (k={k})"):
        import search

        buf = io.StringIO()
        out = RESULTS_DIR / f"live_{engine.backend}_k{k}.csv"
        with st.spinner("Running 89 questions..."):
            with contextlib.redirect_stdout(buf):
                search.run_eval(engine.col, engine.embedder, k, str(out))
        st.session_state["eval_out"] = buf.getvalue()
    if st.session_state.get("eval_out"):
        with st.expander("Console output of the last run"):
            st.code(st.session_state["eval_out"])

    files = sorted(RESULTS_DIR.glob("*.csv"))
    if not files:
        st.info("No results yet. Click the button above, or run `python search.py --eval --out results/run1.csv`.")
    else:
        def summarize(path):
            df = pd.read_csv(path)
            ans = df[df["category"].isin(["factual", "paraphrase", "multi_hop", "access_allowed"])]
            rows = []
            for cat, g in list(ans.groupby("category")) + [("ALL ANSWERABLE", ans)]:
                ranks = g["evidence_rank"]
                rows.append({"category": cat, "questions": len(g),
                             "hit@1 %": round((ranks == 1).mean() * 100, 1),
                             "hit@k %": round((ranks > 0).mean() * 100, 1),
                             "MRR": round((1 / ranks.where(ranks > 0)).fillna(0).mean(), 3),
                             "right document %": round(g["doc_hit"].astype(bool).mean() * 100, 1)})
            leaks = int(df[df["category"] == "access_denied"]["leak"].astype(bool).sum())
            unans = df[df["category"] == "unanswerable"]["top1_score"].mean()
            return pd.DataFrame(rows), leaks, ans["top1_score"].mean(), unans

        chosen = st.selectbox("Results file", files, format_func=lambda p: p.name)
        table, leaks, top_ans, top_unans = summarize(chosen)
        m1, m2, m3 = st.columns(3)
        m1.metric("Access-control leaks", leaks)
        m2.metric("Avg top score, answerable", f"{top_ans:.3f}")
        m3.metric("Avg top score, unanswerable", f"{top_unans:.3f}")
        st.dataframe(table, hide_index=True, width="stretch")
        st.caption("The gap between the two average scores helps you choose the answer threshold in the sidebar.")
        if len(files) > 1:
            comp = pd.DataFrame({p.stem: summarize(p)[0].set_index("category")["hit@k %"] for p in files})
            st.markdown("**hit@k % by category, all result files**")
            st.bar_chart(comp, stack=False, y_label="hit@k %")

# ============ how it works
with tab_how:
    st.subheader("Architecture")
    st.graphviz_chart("""
    digraph G {
      rankdir=LR; bgcolor="transparent"; fontname="Helvetica";
      node [shape=box, style="rounded,filled", fillcolor="#e8eefc", color="#5b7bd5", fontname="Helvetica"];
      edge [color="#5b7bd5"];
      subgraph cluster_ing { label="Ingestion (once)"; style=dashed; color="#9aa7c7";
        Docs [label="Documents\\nPDF · DOCX · MD · TXT"]; Parse; Chunk [label="Chunk by\\nsection"]; EmbC [label="Embed\\nchunks"];
        Docs -> Parse -> Chunk -> EmbC; }
      DB [label="ChromaDB\\nvectors + text +\\nrole metadata", shape=cylinder, fillcolor="#fff3d6", color="#d9a21b"];
      EmbC -> DB;
      subgraph cluster_q { label="Every question"; style=dashed; color="#9aa7c7";
        Q [label="Question"]; EmbQ [label="Embed\\nquestion"]; Filt [label="Role filter"]; Srch [label="Top-k\\nsearch"];
        Gate [label="Score gate"]; Prompt; LLM [label="LLM"]; Ans [label="Answer +\\ncitations", fillcolor="#dff3e3", color="#3d9a5a"];
        Q -> EmbQ -> Filt -> Srch -> Gate -> Prompt -> LLM -> Ans; }
      DB -> Srch;
    }""")
    st.markdown("""
**Ingestion (offline, once).** Each document is parsed (tables are kept as rows), split on its numbered sections, and every chunk
is embedded and stored with its source, section and the roles allowed to read it.

**For every question:**
1. **Embed** the question with the same model used for the chunks.
2. **Role filter** limits the search to chunks the current user may see. This happens inside the database query.
3. **Top-k search** returns the closest chunks by cosine similarity.
4. **Score gate** stops weak matches early, so the assistant says it doesn't know instead of guessing.
5. **Prompt** contains only the retrieved chunks plus strict rules: answer from context, cite document IDs, ignore instructions found in documents.
6. **LLM** writes the answer; the cited document IDs become the source list.

**Why it matters for enterprise use:** answers are grounded in company documents, every claim is traceable to a source,
restricted content never reaches users or the model prompt for unauthorised roles, and each step is measurable.
""")
