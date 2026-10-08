# Content limits use tokens

Content limits retain their previous numeric values, now counted locally with
tiktoken 0.12.0 `o200k_base`. This is a stable model-independent measurement,
not a claim to reproduce every provider's proprietary tokenizer or billing.
Context compaction continues to calibrate its estimate using actual upstream
input usage. It remains a soft, adjustable 250,000-token trigger by default.

| Content | Token allowance |
| --- | ---: |
| Keyless/DDG search snippet, per result | 500 |
| Parallel search excerpt, per result | 1,200 |
| Webpage body, per fetch | 8,000 |
| Cached webpage body, per source | 12,000 |
| Ordinary cross-turn webpage replay, per source / total | 6,000 / 16,000 |
| Document upload extract, per file / combined | 30,000 / 80,000 |
| File read, numbered-line window | 100,000 |
| Changed-file excerpt | 6,000 |
| Ordinary / Agent file-search matched line | 300 / 500 |
| Ordinary stdout and stderr, each | 12,000 |
| Agent stdout and stderr, each | 40,000 |
| Checkpoint snapshots, combined maximum | 200,000 |
| Checkpoint source summary / progress note | 200 / 400 |
| Cross-turn work log | 1,500 |
| User message | 100,000 |

Search queries, extraction objectives, plan text, Skill descriptions and error
text also use token clipping. URLs, IDs, paths, filenames, account validation,
record counts, image limits, upload byte limits and internal memory/byte safety
guards retain their original units. Numbered file reads keep complete lines;
an indivisible line may exceed a window, as before. Truncation annotations and
JSON envelopes are not part of the text allowance.

Char-only upstream parameters do not receive token counts as if they were
characters. Keenable fetch requests overfetch with a conservative character
ceiling, then the application applies its 8,000-token allowance. Keenable
search requests the provider's maximum supported 10,000-character snippet,
then applies 500 tokens locally. Providers can still impose their own limits.

Already-truncated uploaded files and cached records are preserved, not silently
deleted or regenerated. Re-upload/re-fetch when the missing original content
is needed. This change does not alter live-fetch policy, objective forwarding,
tool quotas, retry policy, permissions or history-replay strategy.

Verification is offline: Unicode and special-token text, head/tail clipping,
char-only MCP requests, numbered-file continuation, uploads, caches and the
50-case smoke suite. No paid model, web-search or fetch tests are required.
