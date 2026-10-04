# Grok Build prompt, tools, skills and web results

Source revision: `xai-org/grok-build@2bdd1d6a6369de0e8c68132ea4539e9abd9e14a8`.
License and adaptation notice: `third_party/grok-build`.

## Prompt and execution

`app/templates/grok_system_prompt.md` is the upstream prompt template, unmodified.
`app/prompts.py` renders its actual feature switches. Product identity, the web
UI, account permissions and workspace boundaries are application adapters.
Interactive prompt selection is used in both modes. The upstream user-info
fields (OS, shell, workspace and local date), workspace AGENTS.md/rules, and
skill listing are placed in the first user-message prefix. User text is wrapped
in `<user_query>`. These fields are not appended to the system prompt.
Memory, subagents, browser verification and automatic task-completion reminders
are not advertised because this application does not provide those features.

The former whole-file-only instructions, command rewriting, compulsory plan
execution receipts, semantic search refusal, read-result replacement, and
read-only/mutation stall nudges are no longer used by the live loop. Every
accepted tool call has its actual result committed to the protocol history.
Calls in a batch execute in emitted order; leading independent reads may be
prefetched. HTTP retries and Custom advanced JSON precedence are unchanged.

## Text tool contracts

The core advertised names are `read_file`, `search_replace`, `grep`, `list_dir`,
`bash`, `todo_write`, `get_task_output` and `kill_task` in both modes. `skill` is
offered when enabled skills exist. Conversation/skill-management and syntax-check
extensions are no longer advertised to the model; the administrator UI remains.
Old handler implementations remain for compatibility, not in the new tool schema.
Model-visible results use text/cards rather than JSON envelopes containing
path/revision/UI bookkeeping. Backend tool failures report the error without
instructing the model to read the file again.

- Read: `target_file`, 1-based `offset`, optional `limit`; default 1000 lines,
  25,000 estimated tokens / 100,000 UTF-8 bytes. Anchors appear at the first
  returned line and every tenth file line. Requests for small-file ranges are
  honored. Oversized windows return the upstream token-cap error, including
  the long-single-line extraction hint, instead of clipping a line and implying
  that the remainder can be recovered with the next line offset. Instruction
  and skill markdown is returned whole when it fits. PDF (paged text), PPTX,
  and notebook text extraction is supported locally; visual image/PDF-page
  output remains unavailable and is not advertised as supported.
- Edit: `file_path`, exact `old_string` / `new_string`, optional `replace_all`.
  Empty old text creates or rewrites a file. No fuzzy match or read-before-edit
  gate is applied to this contract.
- Grep: ripgrep regular expressions, normal ignore handling, context flags,
  glob/type and multiline support; omitted head limit is 200, ceiling 2000;
  output byte budget 40,000, per-line cap 1000 characters, deadline 20 seconds.
  Output uses the upstream workspace-result wrapper, grouped match/context
  lines and counts. Hidden files/count output modes are accepted. Ripgrep is a
  host dependency.
- Directory output: tree with root header, breadth-first expansion within a
  10,000-byte budget, collapsed subtree counts and top-three extension buckets.
  Immediate siblings are seeded before the deep walk, which uses ripgrep's
  ignore filtering. This is a Python port, not execution of the Rust renderer;
  nested empty-directory discovery still differs from its WalkBuilder.
- Agent bash: unchanged commands, default 30,000ms foreground wait, then a
  background task. Full command output remains in a private log. Task output
  supports ID arrays and waits; task access is conversation/account scoped.
  Application job cancellation also stops its conversation-owned commands.
- Ordinary bash: same foreground/background task contract, with persistent
  writes to `/workspace` (the conversation's existing storage). DynamicUser and
  bind mounts preserve account/conversation isolation; networking stays disabled.
  `.context` recovery files are mounted read-only, without copying the entire
  archive for each command. File permissions remain compatible with a command
  already running while the model edits a file. Legacy validation helpers still
  use disposable copies but are not part of the advertised native toolset.
- Todo: ID-based partial merge by default; `merge=false` replaces the list;
  pending/in_progress/completed/cancelled, multiple active items. No execution
  permission system or tool receipts are required.

## Skills

Discovery emits names/descriptions/paths, not full bodies. The `skill` call loads
the selected body in the upstream `<skill>` envelope. It supports the 25,000-token
body cap, zero-based `$0`, `$ARGUMENTS[N]`, `$ARGUMENTS`, `${SKILL_DIR}` and session
substitutions. Managed installed/enabled skills and the administrator UI remain.
Native `.grok/skills` and `.agents/skills` directories are discovered as well.
An enabled list containing only retired application-default IDs is interpreted
as the obsolete default configuration when none of those IDs still exists;
the discovered/installed skills then use the normal enabled-by-default policy.
Explicit empty lists and user selections remain unchanged. Folded YAML skill
descriptions are read as text rather than the literal `>` marker.
Enabled is distinct from model-invocable: `disable-model-invocation: true`
excludes a skill from the model's listing and prevents model tool invocation.
Upstream CLI slash-command invocation is not implemented in this web application.

All eleven old application-bundled skill files have been removed, so old enabled
IDs cannot cause their instructions to enter a new context. User-installed
skills and their administrative controls remain. Both modes use the shared
metadata-only XML skill listing and load selected bodies on demand.

The open-source tree does not ship the platform's bundled skill bodies. Product
catalog transport (`remote/skills_client.rs`) advertises bundled entries with no
body; expansion happens on Grok's product/gateway side and requires first-party
session authentication. No fabricated replacement skills are installed.

## Web acquisition and downstream processing

Parallel MCP, Keenable/keyless providers, DuckDuckGo and Jina retain their
acquisition paths and cleaning methods. Backend defaults can still determine
how many sources or how much text they return; an adapter cannot recreate text
the provider never supplied.

Search output uses the upstream textual search-results header and retains source
URLs; UI citations are deduplicated separately. The application no
longer discards sources after ten results or slices snippets at 500/1200 chars.
Upstream Grok's 8192-token search setting is a generation limit on its separate
xAI search model, not a character-truncation rule; this adapter does not add a
paid summarization model to the existing providers.

For fetched text, the upstream overflow policy is used:

- Inline preview: `min(context_window_tokens * 4 * 0.03, 100000)` UTF-8 bytes;
  default fetch context window is 128,000 tokens when unknown.
- Untruncated small content is returned whole. Larger content is saved exactly
  under the private session `web_fetch/N.md` directory; the result contains a
  bounded preview, actual byte counts, path and chunk-read/query instructions.
- Artifact numbers are locked, monotonically reserved and atomically persisted;
  the session artifact budget is 1GiB. Persistence failure is reported by the
  upstream-style no-path truncation footer and a server log warning.
- Base64 data URIs are removed using the upstream header/payload rules.
- The raw download safety limit is 10MiB; oversized content returns an error,
  not an apparently complete silently sliced page. Parallel requests full content so the
  artifact can contain the received full body instead of only excerpts. Other
  providers' own upstream limitations still apply.

The evidence database remains for UI/session history but runtime fetch reuse
follows upstream: only fully inline text, 15-minute TTL, at most 128 entries,
oldest insertion evicted first. Truncated/path-bearing pages are not reusable
fetch-cache entries; recover their content from the saved local artifact.
Storage no longer slices bodies at 12,000 characters. Legacy incomplete rows
are not presented as full responses. Cached bodies are not injected into every
user turn: committed tool history and Grok compaction supply the context.

There is no implicit 3-search/3-fetch/6-round policy anymore. Explicit call budgets
are still supported by explicit callers, but production defaults to no turn
ceiling, as upstream `max_turns=None` does. Restored todo state uses GrokTodo in
both modes. Persisted Responses chains are version-scoped so old prompt/tool
contracts do not continue invisibly after this deployment. Thus this port does
not claim the entire Grok CLI or every optional upstream tool has been copied.

## Verification

The routine smoke suite remains capped at 50 tests. It covers all three Custom
protocols, ordinary/Agent execution, 20 reads followed by an edit, 45 consecutive
tool rounds with no implicit cutoff, source todo
merges, artifact tails after truncation, private recovery and sandbox queries,
multi-call web batches, complete cache persistence, background logs, compaction,
retry, persistent sandbox writes, read-only archive mounts, cross-turn background
tasks and UI syntax. Model/search/fetch responses are simulated; no paid API calls
are made by these tests.
