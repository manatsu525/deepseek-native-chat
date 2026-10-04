# Agent checklist and execution history

Agent mode follows the DSH checklist approach. Ordinary chat retains its
existing plan, workspace, tool and history behavior.

## Checklist

`update_plan` receives the entire `todos` array. Each entry has only `content`
and `status` (`pending`, `in_progress`, `completed`). Every successful update
replaces the previous list, including clearing it with an empty array.
Several items may be in progress. Empty/duplicate descriptions, unknown fields
and invalid statuses are rejected atomically.

The checklist does not grant tool permissions or certify execution. There are
no step IDs, evidence IDs, outcomes, replan reasons, mandatory plan gates or
forced finalization requests in Agent mode. Unfinished checklist items alone
do not mark an answer failed; the model reports remaining work honestly.

`jobs.plan_json` and message metadata preserve the checklist for the existing
trace UI. New turns start new checklists; restarted jobs restore their own.
Version-1 plans remain readable and convert to minimal lists on Agent restart.
The frontend's expand/collapse and scroll behavior is unchanged.

## Real execution state

SQLite `agent_events` is an append-only journal owned by a job/conversation.
It stores initial history, actual model request bodies (including Responses
fallback requests), committed assistant messages, tool dispatches and full
results, unsuccessful attempts, context projections and turn termination.
HTTP authentication headers are not stored. Database protection and account
ownership still apply. Complete file bodies are sensitive conversation data.

UI streaming is separate from commits. Unsuccessful drafts are diagnostic
events, not successful messages in later model requests. Committed tool calls
and matching results replay together. Later Agent turns use this transcript
instead of final-answer-only history with a short operation summary.
Ordinary chat does not use the journal.

Interrupted calls without results are distinguished as not started or started
with unknown outcome. Unknown is not failure: inspect current state before
repeating a side effect that may have applied. This is not exactly-once
execution. Restarted jobs resume committed history instead of silently
starting again from the original question.

Existing deterministic context budgets still bound replay. Context projections
preserve balanced tool groups and are logged separately; original events stay
on disk. No additional paid summarization requests are introduced. Native
reasoning fields are retained only for the same route/configuration scope.

The configured `context_budget_chars` is always a fixed compaction trigger,
including the default 240000. Model window metadata and reported token usage
do not expand it. Protected recent exchanges are not hard-truncated, so the
trigger is not a guarantee that every wire request is below that character
count. Numeric bounds remain 40000–4000000.

Responses function-call replay and request-size measurement use canonical
execution arguments, including their post-execution compacted projection.
Native reasoning items and signatures remain intact. When arguments are
compacted during a stateful turn, the next request rebuilds the stored chain
from compacted local history; tools already executed are not run again.

## Diagnosis and verification

Authenticated `GET /api/jobs/{job_id}/execution?after=0` returns up to 100 events
owned by the current account, with `next_after` for pagination. It is not part
of ordinary frontend polling. Request bodies/file contents can be inspected
without adding them to the UI or hiding intermediate model output.

Default smoke testing stays capped at 50 offline tests. Mocked upstreams cover
all three protocols, checklist replacement, exact request capture, tool replay,
unknown outcomes, restart recovery and ownership. No paid API calls are used.
