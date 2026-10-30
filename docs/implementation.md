# Configuration

## Retrieval

| Setting | 32k 8B | 32k 27B | 128k 8B | 128k 27B |
|---|---:|---:|---:|---:|
| Factual anchors | 50 | 90 | 90 | 110 |
| Trajectory retention | 0.8 | 0.6 | 0.8 | 0.6 |
| Continuation threshold | 0.4 | 0.4 | 0.4 | 0.4 |
| Drop tolerance | 0.1 | 0.1 | 0.1 | 0.1 |
| Beam width | 3 | 3 | 3 | 3 |
| Hop cap | 6 | 6 | 6 | 6 |

First-hop transitions touch factual anchors; each hop follows an incident transition
to its other endpoint, preserving the record's original direction and text.
Expansion uses the trajectory state, avoids repeating facts, and retains completed paths.
Factual anchors and transitions are separate records. The context budget is
8,192 tokens by default, configurable with `--context-token-budget`.

## Construction

| Setting | Default |
|---|---:|
| Macro / micro thresholds | 0.65 / 0.25 |
| Maturity (messages per topic) | 15 |
| Candidate buffer | 5 |
| Historical facts for detection | 10 |
| Grounding threshold | 0.5 |
| Topic-summary refresh (deepening messages) | 10 |
| Routing entity window | 50 |

Historical facts are retrieved across the user's topics. Incomplete buffers are
not flushed at question boundaries. Failed summary updates retain the old summary.
Micro consistency uses lemmatized noun phrases and named entities, matched by
substring containment against normalized entity names and their non-stopword lemmas.
Graph extraction repairs structured responses where possible; logged extraction
failures can leave missing facts or unresolved nodes.

## Graphiti Cost Controls

| Parameter | Default |
|---|---:|
| `GRAPHITI_INVALIDATION_LIMIT` | 3 |
| `GRAPHITI_NODE_DEDUPE_CANDIDATE_LIMIT` | 8 |
| `GRAPHITI_DISABLE_EDGE_INVALIDATION_SEARCH` | 0 |
| `GRAPHITI_RELEVANT_SCHEMA_LIMIT` | 3 |

These limit candidate evidence, not total graph size. Defaults are set before
Graphiti loads. See [source changes](graphiti_changes.patch) and the
[file manifest](graphiti_source_manifest.json).

Resolved settings are saved in each run's `config.json`.
See `python -m transmem.main --help` for command-line overrides.

## Third-Party Notices

`src/graphiti_core` contains modified [Graphiti](https://github.com/getzep/graphiti)
code, copyright 2024 Zep Software, Inc., under [Apache-2.0](../LICENSE).
Original copyright headers and per-file modification notices are retained.
The [source diff](graphiti_changes.patch) and [manifest](graphiti_source_manifest.json)
compare the bundled source against Graphiti `v0.24.3`
(`24105b955669f6473ad0324618f452cda07896f9`). Use the bundled code rather than
installing a separate `graphiti-core` package.

PersonaMem data is separately [MIT-licensed](../data/personamem/LICENSE).
Qwen, BGE-M3, and spaCy model weights are not bundled and retain their own licenses.
