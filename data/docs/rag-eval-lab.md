# rag-eval-lab: RAG Evaluation and GraphRAG

The rag-eval-lab project is a retrieval-augmented generation evaluation lab. The
baseline pipeline embeds a 55-document, 323-chunk corpus with the BGE-M3 model
(1024-dim vectors) into Qdrant and measures recall@5, mrr@10 and hit@5 against a
SHA-256 snapshotted golden question set.

On top of the vector baseline sits a Neo4j GraphRAG layer: a deterministic extractor
mines technology, agent, tag and date entities from the chunks and writes RELATES_TO
edges into Neo4j with idempotent MERGE Cypher. Retrieval runs in three modes:
pure-vector, graph (vector-seeded expansion over `[:RELATES_TO*0..H]`), and hybrid
(reciprocal rank fusion plus a reranker).

Honest measured results: pure-vector recall@5 0.938, hybrid 1.000, graph 0.875.
The hybrid mode recovered a multi-hop answer through a shared tech-entity bridge.
Graph mode p95 latency was 418 ms versus 34 seconds for the reranked vector path,
roughly 80x cheaper because it skips the reranker.
