FROM python:3.11-slim

WORKDIR /app

# System deps for any compiled Python packages (safe default, small)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY data/ ./data/

# Qdrant and vLLM/Groq are external services this container talks to,
# not bundled here — see docker-compose.yml (Day 11) for full-stack wiring.
ENV QDRANT_URL=http://qdrant:6333
ENV GROQ_API_KEY=""
ENV VLLM_BASE_URL=http://host.docker.internal:8001/v1

CMD ["python", "-m", "src.graph"]