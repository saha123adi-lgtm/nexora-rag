# How to run and show this project (everything free)

No step here needs a credit card or a paid plan. The paid options are listed at the bottom so you know what to avoid.

## What you will install (all free)

| Software | Needed for | Where from |
|---|---|---|
| **Python 3.11 or 3.12** | everything | python.org/downloads. On Windows tick **"Add python.exe to PATH"** in the installer. |
| **A terminal** | running commands | Already on your computer (Command Prompt, PowerShell, Terminal) |
| **cloudflared** | giving others a public link from your laptop (Option B) | Cloudflare, free, no Cloudflare account needed |
| **Git + a GitHub account** | only for the always-on link (Option C) | git-scm.com and github.com |
| **Ollama** | only if you want a free offline LLM | ollama.com |
| **VS Code** | optional, for viewing the code | code.visualstudio.com |

---

## Part 1: Run it on your own computer (do this first)

1. **Unzip** `nexora_rag.zip`. Open a terminal inside the `nexora_rag` folder.
   On Windows: open the folder in File Explorer, click the address bar, type `cmd`, press Enter.
2. **Create a virtual environment** and switch it on:
   - Windows (Command Prompt): `python -m venv .venv` then `.venv\Scripts\activate`
   - Windows (PowerShell): `python -m venv .venv` then `.venv\Scripts\Activate.ps1`
     (if PowerShell refuses, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first)
   - Mac or Linux: `python3 -m venv .venv` then `source .venv/bin/activate`
3. **Install the packages** (a few minutes): `pip install -r requirements.txt`
4. **Build the vector database**: `python ingest.py`
   The first run downloads the embedding model (about 90 MB, internet needed once). You should end with
   `Stored 81 chunks`.
5. **Measure retrieval with the real model** (this gives you the numbers for your report):
   `python search.py --eval --out results/minilm_k5.csv`
6. **Start the app**: `streamlit run app.py`
   Your browser opens at http://localhost:8501.
7. **First test:** leave "Answer generation" on *Retrieval only*, pick any sample question and press **Ask this question**.
   You should see the six pipeline cards fill in one after another.

### Set the answer threshold (important, do it once)

The sidebar threshold decides when the assistant says "I don't know". The default for MiniLM (0.30) is a starting guess that I could not
test, because the model could not be downloaded in my workspace.

1. Open the **Evaluation** tab and click **Run the evaluation now**.
2. Read the two numbers: *Avg top score, answerable* and *Avg top score, unanswerable*.
3. Set the sidebar threshold between them, a little under the answerable average.
4. Test with questions the documents cannot answer ("What is the dress code?", "Who won the cricket world cup?"). They should be refused,
   while real questions are still answered. If real questions get refused, lower the threshold; if off-topic ones get answered, raise it.
   Be aware that in my offline test (a keyword stand-in model, not MiniLM) "What is the dress code?" scored exactly at the threshold and was
   not refused, so **rehearse your refusal question before presenting** and use one that reliably refuses.
   Retrieval-only mode can refuse only through this threshold. With Gemini or Ollama the prompt also tells the model to say it doesn't know
   when the passages lack the answer, so refusals are more reliable with an LLM.

### Optional: real AI-written answers, still free

Pick one in the sidebar under **Answer generation**.

**Google Gemini (free tier).** Sign in to Google AI Studio with a Google account, create an API key, choose
*Google Gemini (free tier)* in the sidebar and paste the key. Google's pricing page lists some Gemini models as free of charge on the
free tier, but it also says free-tier content may be used to improve Google's products, so keep to non-sensitive data
(this project's data is fictional). Free tiers have rate limits: a "429" error means wait a minute. If you get "model not found",
copy a current model name from AI Studio into the Model box, because model names change.

**Ollama (fully offline).** Install Ollama, run `ollama pull llama3.2` once, keep Ollama running, choose *Ollama* in the sidebar.
It needs a reasonably modern laptop (8 GB RAM or more) and is slower, but nothing leaves your machine. It only works when the app runs on
the same computer as Ollama, so it is for Option A and B below, not Option C.

**Retrieval only** needs nothing at all and is the safest choice for a live demo with no internet.

---

## Part 2: Let other people see it

### Option A: people in the same room or on the same Wi-Fi

When you run `streamlit run app.py`, the terminal prints a **Network URL** such as `http://192.168.1.20:8501`.
Anyone on the same Wi-Fi can open it. If Windows asks about the firewall, allow access on private networks.

### Option B: a public link from your laptop (recommended for presentations)

Works from anywhere, free, no account. The app keeps running on your computer.

1. Install **cloudflared**:
   - Windows: `winget install --id Cloudflare.cloudflared` (or download it from Cloudflare's downloads page)
   - Mac: `brew install cloudflared`
   - Linux: download the package from Cloudflare's downloads page
2. In one terminal run the app: `streamlit run app.py`
3. In a second terminal run: `cloudflared tunnel --url http://localhost:8501`
4. It prints a link ending in `trycloudflare.com`. Share it. Anyone with the link can open your app.

Things to know: the link changes every time you start the tunnel and stops when you close it or your laptop sleeps.
Cloudflare says quick tunnels are for testing and have no uptime guarantee, so open it a few minutes before presenting.
Do **not** put your Gemini key in the secrets file while the tunnel is open, otherwise every visitor spends your quota.
Paste the key into the sidebar yourself, or use Retrieval only.

### Option C: an always-on link (Streamlit Community Cloud)

Streamlit's hosting service, signed in with GitHub. I could not confirm its limits from Streamlit's documentation and I could not test a
deployment from my workspace, so treat this as "try it, with Option B as the fallback".

1. Create a GitHub account and a **public** repository, for example `nexora-rag` (the data is fictional, so nothing sensitive is exposed).
2. Upload the project to it. With Git, from inside the project folder:
   ```
   git init
   git add .
   git commit -m "Nexora RAG"
   git branch -M main
   git remote add origin https://github.com/YOUR-USERNAME/nexora-rag.git
   git push -u origin main
   ```
   (`.gitignore` already keeps the database folder and your secrets out of GitHub. Never upload `.streamlit/secrets.toml`.)
3. Go to share.streamlit.io, sign in with GitHub and choose **Create app**. Pick your repository, branch `main`, main file `app.py`.
4. Open **Advanced settings**: set Python to 3.11 or 3.12. Only paste a Gemini key under Secrets if you accept every visitor using it
   (format: `GEMINI_API_KEY = "your-key"`).
5. Press **Deploy**. The first start downloads the embedding model and builds the database by itself, which takes a few minutes.
6. If the build fails with a disk or memory error, edit `requirements.txt`, remove the `#` in front of the `--extra-index-url` line
   (it switches to a much smaller CPU-only PyTorch), push again, and redeploy. If it still fails, use Option B.

Free apps may go to sleep when nobody uses them. Open the link a few minutes before presenting so it wakes up.

### Optional: Docker

A `Dockerfile` is included for running the app in a container (`docker build -t nexora-rag .` then
`docker run -p 8501:8501 nexora-rag`). I could not build it in my workspace, so it is untested.
You do not need it for any of the options above.

---

## Part 3: A 5-minute demo script

1. **Ordinary question.** Acting as *Aarav Sharma (employee)*, ask "How many sick leave days are there per year?"
   Walk through the cards: embed, access filter, vector search with scores, score gate, the exact prompt, the answer with its source.
2. **Access control.** Still as Aarav, ask "What is the annual CTC band for level L5?" The answer must not contain the salary band.
   Point at card 2: five documents are hidden, and at card 3: the compensation document (HR-006) is nowhere in the results because it was
   never searched. With an LLM selected the assistant says it doesn't know.
3. **Same question, other user.** Switch to *Priya Nair (HR)* and ask again. Now the compensation document is found and cited.
   The **Access control demo** tab shows both users side by side.
4. **Honesty.** Ask something the documents cannot answer (use the one you rehearsed). The score gate stops it and the LLM is never called.
   If it is not refused, raise the threshold a little.
5. **Evidence.** Open the **Evaluation** tab: hit rate, MRR, and zero leaks across the access-control questions.
6. **Architecture.** The **How it works** tab has the diagram for your closing slide.

Turn on **Slow motion for presentation** in the sidebar so the audience can follow each step. The timings shown are always the real ones.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `pip install` fails or takes forever | Use Python 3.11 or 3.12, make sure the virtual environment is active, and retry on a stable connection |
| `python ingest.py` cannot download the model | Needs internet the first time (a college network may block huggingface.co; try a phone hotspot) |
| "Could not start the engine" in the browser | In the terminal run `python ingest.py` to see the real error. Delete the `chroma_db` folder and run it again |
| Port already in use | `streamlit run app.py --server.port 8502` |
| Gemini says 404 or "model not found" | Copy a current model name from Google AI Studio into the Model box |
| Gemini says 429 | Free-tier rate limit. Wait a minute, or use Retrieval only |
| Ollama "could not reach" | Start Ollama, run `ollama pull llama3.2`, check the host box says `http://localhost:11434` |
| Real questions get "I don't know" | The threshold is too high. Lower it in the sidebar (see the threshold section above) |

## What to avoid (these need payment or a card)

- **Hugging Face Docker Spaces**: Hugging Face's documentation states that Docker Spaces require a paid plan to create.
- **Cloud platforms such as AWS, Google Cloud, Azure**: sign-up normally asks for a payment card.
- **Paid API keys** (Claude, OpenAI): optional in the sidebar, never required. Everything works without them.
