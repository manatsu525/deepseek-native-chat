"""Per-task profiles for the shared Custom agent loop."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal


ModeName = Literal["auto", "chat", "coding"]


CODING_SYSTEM_PROMPT = """You are a coding agent working in a persistent workspace.

Do not implement or debug complete programs inside reasoning. Reasoning is only for
choosing the next tool and should stay under a few short sentences. Code belongs in
workspace tool arguments, not in your thoughts.

On a new coding task, act immediately: save a minimal runnable version with write_file.
It may contain TODOs or known bugs. Then inspect real results with run_python or
check_web_syntax, patch what failed, and check again. For an existing project, inspect
the relevant saved file and modify it locally instead of regenerating every file.

run_python executes only saved .py files. There is no JavaScript or browser runtime
execution tool. For HTML/JavaScript, use check_web_syntax on the actual project file;
do not create a separate Node/test harness that cannot be executed, do not call
run_python for JavaScript, and do not claim runtime or browser testing. Review logic
carefully and state the runtime-testing limitation briefly.

Using many workspace tools is expected. A nonzero exit or ok=false is a real failure;
fix it before claiming success. A syntax check proves syntax only, not browser behavior.
When finished, briefly summarize changed files. The UI supplies download links."""


WORKSPACE_TOOL_PROMPT = """Workspace notes: include every required field. Paths are
workspace-relative exactly as listed by list_files, except tools explicitly described
as pre-bound. Use apply_patch_batch for several edits from one read. Do not repeat an
unchanged read_file. Execution and checks run offline in disposable limited copies."""


STALL_NUDGE_PROMPT = """You have spent too long reasoning without acting. Stop planning
now and call one appropriate workspace tool that materially advances the coding task.
For a new task, call write_file with the best version currently available; it need not
be perfect. For an existing task, read or patch the relevant saved file. Do not answer
with a complete program in prose. 立即停止空想并调用工作区工具推进任务。"""


@dataclass(frozen=True)
class ModeProfile:
    name: str
    coding: bool
    web_tools_enabled: bool
    workspace_tools_enabled: bool
    default_effort: str
    first_round_effort: str
    first_round_max_tokens: int
    max_web_rounds: int
    max_workspace_rounds: int
    first_round_tool_choice: str
    reasoning_stall_chars: int
    max_stall_nudges: int


CHAT_PROFILE = ModeProfile(
    name="chat",
    coding=False,
    web_tools_enabled=True,
    workspace_tools_enabled=False,
    default_effort="high",
    first_round_effort="high",
    first_round_max_tokens=0,
    max_web_rounds=6,
    max_workspace_rounds=0,
    first_round_tool_choice="auto",
    reasoning_stall_chars=60_000,
    max_stall_nudges=1,
)


CODING_PROFILE = ModeProfile(
    name="coding",
    coding=True,
    web_tools_enabled=True,
    workspace_tools_enabled=True,
    default_effort="medium",
    first_round_effort="low",
    first_round_max_tokens=8192,
    max_web_rounds=4,
    max_workspace_rounds=30,
    first_round_tool_choice="required",
    reasoning_stall_chars=12_000,
    max_stall_nudges=2,
)


_CODING_ACTION_RE = re.compile(
    r"写|做|制作|创建|生成|实现|开发|修改|改一下|修复|调试|重构|补全|优化|运行|检查"
    r"|write|create|build|make|implement|develop|modify|change|fix|debug|refactor|complete|optimi[sz]e|run|test",
    re.I,
)
_CODING_SUBJECT_RE = re.compile(
    r"代码|程序|脚本|网页|网站|页面|项目|文件|游戏|组件|接口|前端|后端|数据库|插件|爬虫|机器人"
    r"|html|css|java\s*script|typescript|python|node(?:\.js)?|react|vue|svelte|sql|shell|bash"
    r"|code|program|script|app|website|webpage|project|file|component|api|frontend|backend|database|plugin|bot"
    r"|\.(?:html?|css|m?js|cjs|tsx?|jsx|py|sql|sh|json)\b",
    re.I,
)


def looks_like_coding_request(text: str) -> bool:
    value = str(text or "")
    return bool(
        (_CODING_ACTION_RE.search(value) and _CODING_SUBJECT_RE.search(value))
        or re.search(r"```(?:py|python|js|javascript|ts|html|css)\b", value, re.I)
    )


def resolve_profile(requested: ModeName | str, latest_user_text: str, workspace_has_files: bool) -> ModeProfile:
    value = str(requested or "auto").casefold()
    if value == "coding":
        return CODING_PROFILE
    if value == "chat":
        return CHAT_PROFILE
    if workspace_has_files or looks_like_coding_request(latest_user_text):
        return CODING_PROFILE
    return CHAT_PROFILE
