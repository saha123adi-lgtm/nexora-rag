# Optional: run the whole app in a container (needs Docker installed; free for personal and educational use).
#   docker build -t nexora-rag .
#   docker run -p 8501:8501 nexora-rag        then open http://localhost:8501
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/app/.cache/huggingface
WORKDIR /app

# CPU-only PyTorch first: the default Linux wheel pulls ~3 GB of GPU libraries we do not need
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Build the vector database and cache the embedding model inside the image (needs internet while building)
ARG EMBEDDING_BACKEND=minilm
RUN EMBEDDING_BACKEND=$EMBEDDING_BACKEND python ingest.py

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
