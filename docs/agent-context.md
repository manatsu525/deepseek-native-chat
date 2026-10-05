# Chat and Agent context: Grok Build source port

Reference: [xai-org/grok-build, 2bdd1d6a6369de0e8c68132ea4539e9abd9e14a8](https://github.com/xai-org/grok-build/tree/2bdd1d6a6369de0e8c68132ea4539e9abd9e14a8).
The port targets the regular, non-Cursor session defaults, not just an
approximate summary/archive pattern. Both ordinary chat and Agent use this
context lifecycle. Their existing workspace permissions, tools and plan types
remain separate. The legacy character-budget setting is no longer used.
Ordinary chat accesses its own private archives through read-only `.context/`
paths in `read_file` and `search_files`; archives do not count against workspace
file quotas or appear among generated/downloadable workspace files.

## Source mapping

| Behavior | Upstream source | Python implementation |
|---|---|---|
| Token-based 85% trigger; unknown window 256000 | shell `session/compaction.rs`, `remote/client.rs` | `AgentCompactor`, existing `/models` window lookup |
| UTF-8 bytes/4; image 765; reasoning max(text, decoded ciphertext), no mirror double count | chat-state `actor/state.rs`, token-estimation | `estimate_item`, `TokenMeter` |
| Most recent provider input+output plus appended item estimates, not cumulative billing | chat-state `actor/mutations.rs`, `actor/mod.rs` | `TokenMeter.observe/used` |
| Reseed using provider/estimate ratio, capped by previous provider total | chat-state `actor/mutations.rs` | `TokenMeter.reseed` |
| Background pass1 at 75%, 95%-weighted prefix, tool-group boundaries | shell `session/compaction.rs`, `session/two_pass.rs`, feature registry | `maybe_prefire`, `split_two_pass` |
| NOTE1 extraction >1000 chars; cap 60000; prefix fingerprint; stale cache invalidation; pass2 rejection falls back to full pass | shell `session/two_pass.rs`, `session/compaction.rs` | `note_for_pass2`, `fingerprint`, `compact` |
| Exact nine-section structured prompt | shared `code_compaction/templates/full_replace_summary_prompt.txt` | unchanged `app/templates/grok_compaction_prompt.txt` |
| Retain tools/schema and auto choice for summary prefix alignment; never execute summary tool calls | shell `helpers/session_compact.rs` | `_sample_compaction` |
| Verbatim → fitted (window minus 32768 and tools) → lossy (70% window minus tools) on overflow | shell `session/compaction.rs`, chat-state `compaction_utils.rs` | `_single_pass`, `prepared`, `fit_history` |
| Fit oldest whole message units out, preserve system, truncate final unit with owning tool call | chat-state `compaction_utils.rs` | `fit_history`, `truncate_item` |
| Three total transient/degenerate attempts per stage, 3s interval; cleaned seed minimum 500 chars | shared `code_compaction/config.rs`, `sample.rs` | `_single_pass`, `clean_summary` |
| 300s stream wall-clock backstop; cancellation; deterministic vs overflow vs transient errors | shell `helpers/session_compact.rs` | `_sample_compaction`, `CompactError` |
| Leading scratchpad stripping, section-body control-tag neutralization, continuation carrier | shared `code_compaction/summary.rs` | `clean_summary`, `rebuilt_history` |
| Full history replacement, latest real user query/images, no assistant/tool working tail, state reminder and auto-continue | chat-state `compaction_utils.rs`, shell compaction/turn | `rebuilt_history` |
| Verbose segment_000.md, INDEX.md, stats and section8 keywords; 5MiB whole-turn Markdown cap | compaction-transcript `lib.rs`, shell `compaction_segments.rs` | `render_segment`, `keywords`, `SegmentStore` |
| Full durable raw transcript remains outside active context | shell session persistence | existing SQLite journal plus private `updates.jsonl` |
| Request-copy pruning above half-window: protect 3 user turns, old >4000 chars → 1500+1500, age10 hard clear | chat-state `types.rs`, `actor/request_builder.rs` | `prune_history` |
| Retained age10 pruning counts real prompt turns, allows extra synthetic User items | chat-state `actor/mutations.rs` | `prune_history(retained=True)` |
| Inline image eviction high-water cap minus 3MiB, reclaim to half; Chat/Responses 50MiB, Messages 30MB | chat-state `image_budget.rs`, sampling-types | `image_budget` |
| Failure suppression categories, auth abort, original history unchanged on rejected summaries | shell `session/compaction.rs` | `suppress_reason`, `AgentCompactor` |

## Application adapters (not a claim of byte-identical Grok HTTP traffic)

Custom has an optional per-model `context_window_tokens` working window. Blank
uses the provider catalog window (or the existing 256k fallback when unknown).
A configured value is clamped to the known model capacity and controls the
existing half-window pruning and 75%/85% compaction thresholds in both modes.
It is a local setting, not an upstream request field, and remains active with
advanced JSON enabled. It adds no hard truncation or change to reasoning replay.
The local setting does not change the provider history scope; an actual working
window change invalidates the compactor's window-dependent state as before.

We retain Chat Completions, Responses and Anthropic Messages, the current API
credentials/routing options, and our existing Agent tools. Grok-only request
headers, remote fleet flags, Cursor wire templates, memory flush, forks and
subagent scheduling are not introduced into this project. Existing system/Skill
instructions stay in the system message; our checklist and loaded tool groups
are the relevant runtime state restored after compaction.

Summary calls use the current model/effort and temperature 1, as upstream.
Anthropic requires `max_tokens` and adaptive thinking prohibits temperature;
those protocol requirements are retained. Chat/Responses summarization omits
the normal-answer output cap, matching Grok's session summarization request.
The shell sampler ignores the shared engine's 120s timeout argument; its stream
wall-clock backstop is 300s. The shell's default idle timeout is 600s; with the
default wall-clock budget, the wall-clock backstop normally fires first.

Our jobs have per-job HTTP clients, not Grok's persistent session actor. On
normal completion we finish an in-flight prefire before closing that client
and persist its cache; stopping/failing cancels it. This may add tail latency
to a turn that ends while prefire is running. Next turns restore the cache,
meter, plan and loaded groups; route/config/window changes invalidate
provider-dependent state. Auth failures do not proceed to oversized sampling.

Responses summary calls are independent, never appended to the live
`previous_response_id` chain. Compaction, image eviction and request pruning
reset the old chain before sending the rewritten full input. Native item
mirrors and our synthetic-user labels are not double-counted or leaked as
unknown Chat message fields.

Archives live at `DATA_DIR/agent_sessions/<sha256(conversation_id)>/`, not the
shared `/home/share` browser. Directories are private, files mode0600. The
Agent can use its existing read/grep tools to recover exact text. Markdown has
the upstream 5MiB cap; `updates.jsonl` and SQLite keep complete original tool
arguments/results. Conversation deletion also removes its archive. Disk
failure prevents replacing the live history; no automatic character hard cap
silently discards the complete raw transcript.

Web tool implementation/response-size limits are not changed. Agent cache
receives freshly fetched pages so clearing live-evidence dedupe sets after
compaction does not require paying to fetch the same page again.

## Cost and verification

Summarization is a real model request during normal use and is billed by the
chosen upstream. Its reported usage is included in the job's usage; it is not
counted as a tool round. This change does not promise free compaction or
guarantee cache hits across rewritten history.

Default `python tests/run_smoke.py` remains exactly50 offline cases.
Focused `test_agent_compaction` uses source-derived vectors and scripted
upstream responses for all three protocols; no model/search/fetch service is
contacted. This verifies protocol balance, original archive visibility,
successive compactions, prefire/cache, failed-summary preservation and
cancellation. Model-specific summary quality remains unverified without live
API calls, which are deliberately not used for this migration.
