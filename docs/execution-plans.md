# Execution plans

The shared Chat Completions, Responses and Messages loop owns plan state;
visible reasoning is not required. Simple answers have no extra model call.

- At most two initial workspace/Agent operations can run without a plan.
  Continued execution advertises only `update_plan` until a current step exists.
  The same rule is checked before every operation, including batched calls.
- A plan describes deliverables and verification, with one `in_progress` step.
  Successful tool-call IDs are associated with that step by the server.
- Completing a step requires an outcome and evidence IDs from that step.
  Failed calls, other steps' results and pre-replan receipts cannot complete it.
- Plan changes require an explicit reason. Completed steps stay in the record;
  revisiting one requires reactivation and fresh evidence.
- Blocked work has an explicit outcome. Early final answers with an active step
  get at most two reconciliation opportunities; remaining work is reported as
  incomplete, not silently marked done.
- Each batch appends current state to its last tool result before building
  Responses incremental input. Compaction also retains the plan.
- `jobs.plan_json` checkpoints every operation/update. Restarting the same job
  restores the plan but requires fresh evidence for its interrupted step.
  This is not exactly-once execution: check files before repeating side effects.
- Final, failed and stopped message metadata retain the plan. Later messages
  receive the prior plan as history, not as mandatory instructions for a new task.
- The existing trace panel displays live steps and outcomes, without changing
  its expand/collapse or scroll behavior.

Evidence means an operation succeeded, not that its output semantically solves
the task. Outcome quality and meaningful verification still require model
judgment; the runtime does not infer success merely from a write or command.

Tests use mocked upstream streams, without paid API calls. The default smoke
suite remains capped at 50 tests and covers all three protocol paths.
