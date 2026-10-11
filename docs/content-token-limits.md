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

A fetched page's entire provider-returned text is saved outside the model's
context in `data/page-cache`, isolated by conversation. Short pages return in
full. Long pages initially return a directory, relevant headings and a short
preview rather than an unrelated 8,000-token prefix. Subsequent reads return
contiguous numbered text; the model
can use `fetch_webpage(mode="find", query=...)` to locate a chapter and then
`start_line` / `end_line` to read it completely. Each read defaults to 8,000
tokens and can request up to 32,000; `complete=false` and `next_line` explicitly
identify unread text. Subsequent reads use the full local source, even after
network fetch quota is exhausted, without issuing a provider request.
Legacy excerpt caches and known provider truncation are marked
`source_complete=false`; `refresh=true` explicitly requests a new network
fetch when that partial source is insufficient. No local retrieval scheme can
recover content that the provider itself did not return.
Cache files expire after seven days without use. Cleanup runs every six hours
and evicts oldest files if the cache exceeds 512 MB. Only owned cache files
are removed; uploaded files and workspace files are not affected.
When Keenable's live crawl is refused it retries from Keenable's cache
(`live: false`); Parallel retries with `allow_live_fetch: false`. Jina already
serves its cache by default.

Already-truncated uploaded files and cached records are preserved, not silently
deleted or regenerated. Re-upload/re-fetch when the missing original content
is needed. Search providers, network quotas, permissions and history-replay
strategy remain unchanged; local reads do not consume network fetch quota.

Verification is offline: Unicode and special-token text, head/tail clipping,
char-only MCP requests, numbered-file continuation, uploads, caches and the
50-case smoke suite. No paid model, web-search or fetch tests are required.
