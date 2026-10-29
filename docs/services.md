# Service Setup

Start the following services and enter their addresses and credentials in `.env`.

## Models with vLLM

Use Hugging Face model IDs `Qwen/Qwen3-8B` and `BAAI/bge-m3`; vLLM downloads
their weights automatically, or accepts a local model directory.
Follow the [official vLLM guide](https://docs.vllm.ai/en/stable/) for installation
and GPU allocation. Run each server in a separate terminal with sufficient GPU memory:

```bash
vllm serve Qwen/Qwen3-8B --served-model-name qwen3-8b \
  --host 127.0.0.1 --port 8011 --max-model-len 32768
```

```bash
vllm serve BAAI/bge-m3 --served-model-name bge-m3 --runner pooling \
  --hf-overrides '{"architectures":["BgeM3EmbeddingModel"]}' \
  --pooler-config.task embed \
  --host 127.0.0.1 --port 8001 --max-model-len 8192
```

The embedding command uses the current vLLM pooling interface and returns dense
BGE-M3 vectors (1,024 dimensions). Configure:

```dotenv
LLM_BASE_URL=http://localhost:8011/v1
EMBEDDING_BASE_URL=http://localhost:8001/v1
EMBEDDING_MODEL=bge-m3
```

For another backbone, serve its corresponding weights and model name.

## Neo4j

Start a dedicated Neo4j 5.26 instance following the
[official Neo4j guide](https://neo4j.com/docs/operations-manual/current/docker/introduction/).
Set `NEO4J_URI`, `NEO4J_USER` and `NEO4J_PASSWORD` in `.env`;
the default local Bolt address is `bolt://localhost:7687`.

Use a dedicated experimental database. The runner deletes its run-specific
graph data after each user's run, including when an error interrupts it.

Keep all services running during experiments. For remote services, replace
`localhost` with their addresses and use authenticated, access-controlled connections.
