# develop branch scope

Base: GitHub main `49e4b33`. dev remains pinned at `ba9cd6f`.

develop retains the main API/UI/retry behavior, plus Agent's durable execution
journal, DSH-style checklist, fixed budget fixes, and ordinary
on-demand file-tool loading. The VPS tracks develop.

Excluded: Grok compaction engine/prompt, 75%/85% window triggers, segmented
compaction archives, the working-token-window setting, and ordinary full tool
history/reasoning replay. No Grok context module is imported or shipped here.

Both modes use the existing stub/checkpoint policy with a token trigger. It replaces old tool
results with stubs/checkpoints when the fixed configured budget is exceeded;
original Agent events remain immutable. Native reasoning is retained inside
active tool loops, as before this branch split.

Ordinary chats restore user/final-assistant messages and bounded web evidence
from main's original path. They persist only tool-group availability, not the
full tool transcript. Agent restores its actual tool transcript; duplicate web
evidence is not injected there, and old generated user suffixes are cleaned in
the replay projection without deleting original messages or tool results.

Existing conversations, credentials, files, and archived logs are preserved.
Old working-window values are not used; Custom again exposes the fixed
token budget, default 250000. Provider input usage calibrates local estimates;
the threshold is not a hard request ceiling. Old default character budgets
migrate to 250000 tokens, custom character values approximately divide by four.
Routine smoke tests remain capped at 50, offline only.

No web/read-only progress gate requires a file mutation, command or validation.
The old mutation counters, write nudges and their refusals have been removed;
ordinary research never needs an unrelated file operation to unlock tools.
Normal per-tool quotas and filesystem permissions remain unchanged.

Tool-loop continuity and result-based progress reporting are common system
rules in both modes, including ordinary web research before file tools load.
Intermediate text reports new findings or a concrete evidence gap and the next
action; the substantive user answer belongs at completion. Loading file tools
does not duplicate these rules or reset the active request.
