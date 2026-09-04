FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY data/ ./data/
COPY docs/ ./docs/

# Qdrant and vLLM/Groq are external services this container talks to,
# not bundled here — see docker-compose.yml (Day 11) for full-stack wiring.
# vLLM specifically stays outside any container — it runs natively in WSL2
# due to GPU passthrough (see NOTES.md, Day 5); host.docker.internal lets
# this container reach it on the host machine.
ENV QDRANT_URL=http://qdrant:6333
ENV VLLM_BASE_URL=http://host.docker.internal:8001/v1
ENV GROQ_API_KEY=""

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]