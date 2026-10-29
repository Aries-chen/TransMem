# Processed PersonaMem

Based on [PersonaMem](https://github.com/bowen-upenn/PersonaMem):
[official dataset](https://huggingface.co/datasets/bowen-upenn/PersonaMem-v1) |
[paper](https://arxiv.org/abs/2504.14225).

Jiang et al. (2025), *Know Me, Respond to Me: Benchmarking LLMs for Dynamic User
Profiling and Personalized Responses at Scale*. Please cite the original work
as well as TransMem when using these processed subsets.

| Split | User memories | Questions |
|---|---:|---:|
| 32k | 37 | 475 |
| 128k | 55 | 2,161 |

## Processing and Format

We retain the fixed quality-filtered question selection used for TransMem.
Each `shared_context_id` is an independent memory, even when personas recur.
The original conversation text is retained; histories end at the last evaluated
question's boundary. Synthetic timestamps start at 2015-01-01 UTC and advance
one day per message: they encode order, not verified real-world event dates.

- `histories.jsonl`: user ID and ordered messages with role, content and timestamp.
- `questions.jsonl`: question ID, user ID, query, four options, answer, category,
  topic and `history_end` (inclusive, zero-based).
- `manifest.json`: counts, category frequencies and file SHA256 checksums.

Only messages through a question's `history_end` enter memory before answering.

## License

The source dataset declares the MIT license; its terms are retained in
[LICENSE](LICENSE). These files modify selection, schema and timestamp fields
and are not the unmodified official release. Dataset licensing is separate from
the TransMem software license.
