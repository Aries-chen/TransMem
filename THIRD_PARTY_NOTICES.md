# Third-Party Notices

## Graphiti

TransMem includes a modified [Graphiti](https://github.com/getzep/graphiti)
source tree in `src/graphiti_core`.
Copyright 2024 Zep Software, Inc.; licensed under Apache-2.0.
Original headers are retained; see [LICENSE](LICENSE).

The source comparison uses official tag `v0.24.3`, commit
`24105b955669f6473ad0324618f452cda07896f9`.
This is a comparison reference, not a claim about the historical fork's origin.

- [Source changes](docs/graphiti_changes.patch): complete diff against the reference.
- [File manifest](docs/graphiti_source_manifest.json): file comparisons and checksums.
- [Implementation details](docs/implementation.md#graphiti-cost-controls): extraction,
  response repair, deduplication and invalidation settings.

Use the included source; installing a separate `graphiti-core` package into the
same environment may conflict with it.

## Data and Models

[PersonaMem](data/personamem/README.md) has its own MIT license.
Qwen, BGE-M3 and the spaCy language model retain their respective licenses.
Model weights are not distributed with this repository.
