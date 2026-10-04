# Grok Build prompt, tools, skills and web results

Source revision: `xai-org/grok-build@2bdd1d6a6369de0e8c68132ea4539e9abd9e14a8`.
License and adaptation notice: `third_party/grok-build`.

## Prompt and execution

`app/templates/grok_system_prompt.md` is the upstream prompt template, unmodified.
`app/prompts.py` renders its actual feature switches. Product identity, the web
UI, account permissions and workspace boundaries are application adapters.
The upstream user-info fields (OS, shell, workspace and local date) are placed
in the web application's system context instead of a terminal-specific first
user-message wrapper.
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
`bash`, `todo_write`; Agent additionally has `skill`, `get_task_output` and
`kill_task`. Existing conversation/skill-management and syntax-check tools remain
application extensions. Old names remain executable for existing conversations.

- Read: `target_file`, 1-based `offset`, optional `limit`; default 1000 lines,
  25,000 estimated tokens / 100,000 UTF-8 bytes. Anchors appear at the first
  returned line and every tenth file line. Requests for small-file ranges are
  honored. This adapter advertises text reads, not unsupported image/PDF tools.
- Edit: `file_path`, exact `old_string` / `new_string`, optional `replace_all`.
  Empty old text creates or rewrites a file. No fuzzy match or read-before-edit
  gate is applied to this contract.
- Grep: ripgrep regular expressions, normal ignore handling, context flags,
  glob/type and multiline support; omitted head limit is 200, ceiling 2000;
  output byte budget 40,000. Ripgrep is a host dependency.
- Directory output: 10,000-character budget. Listing is a Python filesystem/UI
  adapter, not the upstream Rust directory tree card renderer.
- Agent bash: unchanged commands, default 30,000ms foreground wait, then a
  background task. Full command output remains in a private log. Task output
  supports ID arrays and waits; task access is conversation/account scoped.
  Application job cancellation also stops its conversation-owned commands.
- Ordinary bash: upstream foreground-only configuration, preserving the existing
  no-network disposable workspace. `.context` recovery files are exported into
  that disposable copy for local queries; the real archive remains untouched.
- Todo: ID-based partial merge by default; `merge=false` replaces the list;
  pending/in_progress/completed/cancelled, multiple active items. No execution
  permission system or tool receipts are required.

## Skills

Discovery emits names/descriptions/paths, not full bodies. The `skill` call loads
the selected body in the upstream `<skill>` envelope. It supports the 25,000-token
body cap, zero-based `$0`, `$ARGUMENTS[N]`, `$ARGUMENTS`, `${SKILL_DIR}` and session
substitutions. Managed installed/enabled skills and the administrator UI remain.
Native `.grok/skills` and `.agents/skills` directories are discovered as well.

The open-source tree does not ship the platform's bundled skill bodies. We do
not label the application's old built-in planning/test workflows as Grok skills
or enable them by default. Explicit existing user enablement remains respected.

## Web acquisition and downstream processing

Parallel MCP, Keenable/keyless providers, DuckDuckGo and Jina retain their
acquisition paths and cleaning methods. Backend defaults can still determine
how many sources or how much text they return; an adapter cannot recreate text
the provider never supplied.

Search output contains content and deduplicated citations. The application no
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

The application's provider-response cache is retained as an acquisition adapter;
preview artifacts are materialized per call. Cache storage no longer slices bodies
at 12,000 characters. Legacy cache rows are marked incomplete on migration and
are not presented as full responses. Cached bodies are no longer injected into
every user turn: committed tool history and Grok compaction supply the context.

There is no implicit 3-search/3-fetch/6-round policy anymore. Explicit call budgets
are still supported, as is this application's total-turn ceiling (ordinary 40,
Agent 96), equivalent to configuring upstream `max_turns`. Thus this port does
not claim the entire Grok CLI or every optional upstream tool has been copied.

## Verification

The routine smoke suite remains capped at 50 tests. It covers all three Custom
protocols, ordinary/Agent execution, 20 reads followed by an edit, source todo
merges, artifact tails after truncation, private recovery and sandbox queries,
multi-call web batches, complete cache persistence, background logs, compaction,
retry and UI syntax. Model/search/fetch responses are simulated; no paid API calls
are made by these tests.
