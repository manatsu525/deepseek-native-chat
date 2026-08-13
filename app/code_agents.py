"""Plan/write then independently review/test coding agents.

The two agents share the parent answer's remaining web_search / fetch_webpage
quota (3 + 3). They never get a separate budget.
"""

from __future__ import annotations

import contextvars
import json
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional
from urllib.parse import urlsplit

import httpx

from . import code_sandbox
from .config import settings
from .custom_tool_normalization import normalize_tool_calls
from .mimo import (
    DEFAULT_REASONING_EFFORT,
    FETCH_WEBPAGE_TOOL,
    JINA_MAX_FETCHES_PER_RESPONSE,
    MIMO_MAX_SEARCH_RESULTS,
    MIMO_MAX_SEARCHES,
    PARALLEL_FETCH_WEBPAGE_TOOL,
    PARALLEL_SEARCH_WEB_TOOL,
    SEARCH_WEB_TOOL,
    _canonical_url,
    _duckduckgo_search,
    _merge_tool_call,
    _merge_usage,
    _normalize_usage,
    _page_source,
    _read_with_jina,
    _safe_fetch_url,
    _tool_calls,
    _url,
    custom_auth_headers,
)
from .keyless_web import KEYLESS_FETCH_WEBPAGE_TOOL, KEYLESS_SEARCH_WEB_TOOL


MAX_AGENT_ITERATIONS = 3
MAX_AGENT_TOOL_ROUNDS = 4
CODE_STEP_ACTIONS = {"code_plan", "code_write", "code_review", "code_test"}
coding_job_active: contextvars.ContextVar[bool] = contextvars.ContextVar("coding_job_active", default=False)
_FENCE_RE = re.compile(r"```(?:([\w./+-]+))?\n(.*?)```", re.S)
_FENCE_NAMES = {
    "html": "index.html",
    "htm": "index.html",
    "python": "main.py",
    "py": "main.py",
    "javascript": "main.js",
    "js": "main.js",
    "css": "style.css",
}

CODING_HINTS = (
    "写代码", "写个代码", "写一段代码", "写一个脚本", "写个脚本", "写一个程序", "写个程序",
    "写一个函数", "写个函数", "写一个工具", "实现一个", "实现下", "帮我实现",
    "脚本", "编程", "写单测", "写测试", "算法题", "leetcode",
    "write a script", "write a program", "write a function", "implement ",
    "implement a", "write code", "unit test",
)
NON_CODING_HINTS = (
    "什么意思", "解释一下", "讲解", "为什么", "是什么意思", "怎么读",
    "explain", "what does", "why does",
)

CODING_TOOL_PROMPT = (
    " If the user needs you to write, implement, generate, or substantially revise a program "
    "(HTML/JS app, script, module, or runnable function), call write_and_verify_code exactly once "
    "with the full user task. Do not dump unverified programs into the final answer. "
    "Do not call it again after it returns, do not wrap the same task as a Python file-writer, "
    "and do not call it merely to write unit tests — tests belong inside that one job. "
    "Explaining existing code, tiny illustrative snippets, config, SQL-only, or regex does not need this tool. "
    "The coding agents share this answer's remaining web_search and fetch_webpage quota."
)

WRITE_AND_VERIFY_CODE_TOOL = {
    "type": "function",
    "function": {
        "name": "write_and_verify_code",
        "description": (
            "Plan, write, independently review, and run tests for executable code on this server. "
            "Required once per answer when you need to produce a working program. "
            "Never call this from inside an already running coding job, and never call it only to write tests. "
            "Shares this answer's remaining web_search/fetch_webpage quota (max 3 searches and 3 fetches total). "
            "Returns verified files, review notes, and real test output."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "Self-contained coding task, including language, inputs, outputs, and constraints",
                },
                "language": {"type": "string", "description": "Primary language, for example python"},
            },
            "required": ["task"],
            "additionalProperties": False,
        },
        "strict": False,
    },
}

SUBMIT_CODE_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_code",
        "description": "Submit the complete implementation after a short plan. Include every file needed to run and test.",
        "parameters": {
            "type": "object",
            "properties": {
                "plan": {"type": "string", "description": "Short implementation plan"},
                "files": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "content": {"type": "string"},
                        },
                        "required": ["path", "content"],
                    },
                },
                "test_hint": {"type": "string", "description": "How the reviewer should test this"},
            },
            "required": ["plan", "files"],
            "additionalProperties": False,
        },
        "strict": False,
    },
}

SUBMIT_REVIEW_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_review",
        "description": "Submit an independent review plus sandbox-safe test commands (python3 or node only).",
        "parameters": {
            "type": "object",
            "properties": {
                "passed": {"type": "boolean"},
                "issues": {"type": "string"},
                "test_commands": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["passed", "issues", "test_commands"],
            "additionalProperties": False,
        },
        "strict": False,
    },
}


def harvest_files_from_text(text: str) -> list[dict[str, str]]:
    """Recover files when a model dumps fenced code instead of calling submit_code."""
    files: list[dict[str, str]] = []
    used_names: set[str] = set()
    for index, (info, body) in enumerate(_FENCE_RE.findall(text or ""), 1):
        content = str(body).strip("\n")
        if len(content) < 20:
            continue
        token = str(info or "").strip()
        path = ""
        if "/" in token or "." in token:
            try:
                path = str(code_sandbox.safe_relpath(token.split()[0]))
            except Exception:
                path = ""
        if not path:
            lang = token.split()[0].lower() if token else ""
            path = _FENCE_NAMES.get(lang, f"snippet_{index}.txt")
        if path in used_names:
            stem = Path(path).stem
            suffix = Path(path).suffix or ".txt"
            path = f"{stem}_{index}{suffix}"
        used_names.add(path)
        files.append({"path": path, "content": content})
    return files


def looks_like_coding_request(text: str) -> bool:
    """Conservative detector used for optional DeepSeek routing and tests."""
    blob = str(text or "")
    lowered = blob.casefold()
    if any(hint in blob or hint in lowered for hint in CODING_HINTS):
        if any(hint in blob or hint in lowered for hint in NON_CODING_HINTS) and "```" not in blob:
            return False
        return True
    if re.search(r"```(?:python|py|javascript|js|ts|go|rust|java|c\+\+|bash|sh)\b", lowered):
        return "解释" not in blob and "explain" not in lowered
    return False


@dataclass
class SharedWebBudget:
    search_count: int = 0
    fetch_count: int = 0
    searched_queries: set[str] = field(default_factory=set)
    known_urls: dict[str, str] = field(default_factory=dict)
    attempted_urls: set[str] = field(default_factory=set)
    last_search_objective: str = ""
    last_search_queries: list[str] = field(default_factory=list)
    reader_enabled: bool = False
    sources: dict[str, dict[str, str]] = field(default_factory=dict)
    steps: list[dict[str, Any]] = field(default_factory=list)


def _web_tools(backend: str, budget: SharedWebBudget) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    if budget.search_count < MIMO_MAX_SEARCHES:
        if backend == "parallel":
            tools.append(PARALLEL_SEARCH_WEB_TOOL)
        elif backend == "legacy":
            tools.append(SEARCH_WEB_TOOL)
        else:
            tools.append(KEYLESS_SEARCH_WEB_TOOL)
    if budget.fetch_count < JINA_MAX_FETCHES_PER_RESPONSE and budget.reader_enabled:
        if backend == "parallel":
            tools.append(PARALLEL_FETCH_WEBPAGE_TOOL)
        elif backend == "legacy":
            tools.append(FETCH_WEBPAGE_TOOL)
        else:
            tools.append(KEYLESS_FETCH_WEBPAGE_TOOL)
    return tools


def _quota_note(budget: SharedWebBudget) -> str:
    return (
        f"Remaining shared web quota for this whole answer: "
        f"{max(0, MIMO_MAX_SEARCHES - budget.search_count)} searches, "
        f"{max(0, JINA_MAX_FETCHES_PER_RESPONSE - budget.fetch_count)} page reads."
    )


async def _complete_round(
    *,
    api_client: httpx.AsyncClient,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    config: dict[str, Any],
    effort: str,
    timeout: int,
    stopped: Callable[[], bool],
) -> dict[str, Any]:
    from .mimo import is_mimo_model
    from .mimo_local import _apply_thinking_options

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_completion_tokens" if is_mimo_model(model) else "max_tokens": int(config["max_completion_tokens"]),
        "stream": True,
    }
    _apply_thinking_options(
        payload,
        base_url,
        model,
        config["thinking"],
        effort,
        bool(config.get("reasoning_effort_enabled", True)),
        int(config["max_completion_tokens"]),
    )
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    if not is_mimo_model(model) or config["thinking"] == "disabled":
        payload["temperature"] = float(config["temperature"])
        payload["top_p"] = float(config["top_p"])
    headers = custom_auth_headers(api_key, stream=True)
    answer = ""
    reasoning = ""
    usage: dict[str, Any] = {}
    found: dict[int, dict[str, Any]] = {}
    async with api_client.stream("POST", _url(base_url, "/chat/completions"), headers=headers, json=payload) as response:
        if response.status_code >= 400:
            body = (await response.aread()).decode(errors="replace")[:2000]
            raise RuntimeError(f"Custom API {response.status_code}: {body}")
        async for line in response.aiter_lines():
            if stopped():
                raise InterruptedError("stopped")
            if not line or not line.startswith("data:"):
                continue
            raw = line[5:].strip()
            if not raw or raw == "[DONE]":
                if raw == "[DONE]":
                    break
                continue
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if data.get("error"):
                raise RuntimeError(f"Custom 响应失败: {data['error']}")
            raw_usage = data.get("usage")
            if isinstance(raw_usage, dict):
                usage = _normalize_usage(raw_usage)
            for choice in data.get("choices") or []:
                delta = choice.get("delta") or {}
                message = choice.get("message") or {}
                answer += str(delta.get("content") or "")
                reasoning += str(delta.get("reasoning_content") or delta.get("reasoning") or "")
                if message.get("content") and not delta.get("content"):
                    answer += str(message.get("content") or "")
                if (message.get("reasoning_content") or message.get("reasoning")) and not (
                    delta.get("reasoning_content") or delta.get("reasoning")
                ):
                    reasoning += str(message.get("reasoning_content") or message.get("reasoning") or "")
                for index, call in enumerate(delta.get("tool_calls") or []):
                    _merge_tool_call(found, call, index)
                for index, call in enumerate(message.get("tool_calls") or []):
                    _merge_tool_call(found, call, index)
    return {
        "content": answer,
        "reasoning": reasoning,
        "usage": usage,
        "tool_calls": normalize_tool_calls(_tool_calls(found, 0)),
    }


async def _run_web_tool(
    name: str,
    arguments: dict[str, Any],
    *,
    budget: SharedWebBudget,
    backend: str,
    clients: dict[str, Any],
    model: str,
    stopped: Callable[[], bool],
) -> str:
    if name == "web_search":
        if budget.search_count >= MIMO_MAX_SEARCHES:
            return f"web_search 已达到本回答上限（最多 {MIMO_MAX_SEARCHES} 次），不能再搜索。"
        if backend == "parallel":
            objective = " ".join(str(arguments.get("objective") or "").split())[:1000]
            raw_queries = arguments.get("search_queries") or []
            if not isinstance(raw_queries, list):
                raise ValueError("search_queries 必须是数组")
            queries = [" ".join(str(item).split())[:200] for item in raw_queries[:3]]
            queries = list(dict.fromkeys(item for item in queries if item))
            if not objective or not queries:
                raise ValueError("Parallel 搜索需要 objective 和至少一个 search_query")
            query_key = json.dumps([objective.casefold(), *[item.casefold() for item in queries]], ensure_ascii=False)
            display = queries
        else:
            query = " ".join(str(arguments.get("query") or "").split())[:500]
            if not query:
                raise ValueError("搜索词不能为空")
            query_key = query.casefold()
            queries = [query]
            objective = query
            display = query
        step = {
            "id": f"code-search-{len(budget.steps)+1}",
            "status": "running",
            "action": "search",
            "query": display,
            "url": "",
            "error": "",
        }
        budget.steps.append(step)
        if query_key in budget.searched_queries:
            step["status"] = "skipped"
            return "该查询已经搜索过，不重复请求。"
        budget.searched_queries.add(query_key)
        budget.search_count += 1
        if backend == "parallel":
            data = await clients["parallel"].call_tool(
                "web_search",
                {
                    "objective": objective,
                    "search_queries": queries,
                    "session_id": clients["session_id"],
                    "model_name": model[:100],
                },
            )
            results = []
            for raw in (data.get("results") or [])[:MIMO_MAX_SEARCH_RESULTS]:
                if not isinstance(raw, dict):
                    continue
                excerpts = "\n\n".join(str(item) for item in raw.get("excerpts") or [])
                results.append(
                    {
                        "url": str(raw.get("url") or ""),
                        "title": str(raw.get("title") or raw.get("url") or ""),
                        "snippet": excerpts[:1200],
                        "publish_date": str(raw.get("publish_date") or ""),
                    }
                )
            budget.last_search_objective = objective
            budget.last_search_queries = queries
        elif backend == "legacy":
            results = await _duckduckgo_search(clients["search"], objective, MIMO_MAX_SEARCH_RESULTS, stopped)
        else:
            results = await clients["keyless"].search(objective, MIMO_MAX_SEARCH_RESULTS)
        for item in results:
            try:
                canonical = _canonical_url(item["url"])
            except (KeyError, ValueError):
                continue
            budget.known_urls[canonical] = item["url"]
            budget.sources.setdefault(
                item["url"],
                {
                    "url": item["url"],
                    "title": item.get("title") or item["url"],
                    "summary": item.get("snippet") or "",
                    "site_name": urlsplit(item["url"]).netloc.removeprefix("www."),
                    "publish_time": item.get("publish_date") or "",
                    "logo_url": "",
                },
            )
        budget.reader_enabled = bool(budget.known_urls)
        step["status"] = "completed"
        return json.dumps({"objective": objective, "search_queries": queries, "results": results}, ensure_ascii=False)

    if name == "fetch_webpage":
        if budget.fetch_count >= JINA_MAX_FETCHES_PER_RESPONSE:
            return f"fetch_webpage 已达到本回答上限（最多 {JINA_MAX_FETCHES_PER_RESPONSE} 次），不能再读取。"
        target_url = _safe_fetch_url(arguments.get("url"))
        step = {
            "id": f"code-fetch-{len(budget.steps)+1}",
            "status": "running",
            "action": "open_page",
            "query": "",
            "url": target_url,
            "error": "",
        }
        budget.steps.append(step)
        canonical = _canonical_url(target_url)
        if canonical in budget.attempted_urls:
            step["status"] = "skipped"
            return f"该网页本回答已经尝试过：{target_url}"
        budget.attempted_urls.add(canonical)
        budget.fetch_count += 1
        if backend == "parallel":
            objective = " ".join(str(arguments.get("objective") or budget.last_search_objective or "").split())[:200]
            fetch_arguments: dict[str, Any] = {
                "urls": [target_url],
                "full_content": False,
                "session_id": clients["session_id"],
                "model_name": model[:100],
            }
            if objective:
                fetch_arguments["objective"] = objective
            if budget.last_search_queries:
                fetch_arguments["search_queries"] = budget.last_search_queries
            data = await clients["parallel"].call_tool("web_fetch", fetch_arguments)
            fetched = next((item for item in data.get("results") or [] if isinstance(item, dict)), None)
            if not fetched:
                raise RuntimeError("Parallel MCP 未返回正文")
            content = str(fetched.get("full_content") or "\n\n".join(str(item) for item in fetched.get("excerpts") or [])).strip()[:8000]
            budget.sources[target_url] = {
                "url": target_url,
                "title": str(fetched.get("title") or target_url)[:160],
                "summary": " ".join(content.split())[:320],
                "site_name": urlsplit(target_url).netloc.removeprefix("www."),
                "publish_time": str(fetched.get("publish_date") or ""),
                "logo_url": "",
            }
        elif backend == "legacy" or backend == "you":
            content = await _read_with_jina(clients["jina"], target_url, stopped)
            budget.sources[target_url] = _page_source(target_url, content)
        else:
            objective = " ".join(str(arguments.get("objective") or budget.last_search_objective or "").split())[:200]
            content = await clients["keyless"].fetch(target_url, objective)
            budget.sources[target_url] = _page_source(target_url, content)
        step["status"] = "completed"
        return f"网页 URL：{target_url}\n\n{content}"

    raise ValueError(f"不支持的工具：{name}")


def classify_test_run(*, tests: list[dict[str, Any]], sandbox_blocked: bool, executed: bool) -> str:
    """Only executed tests can send the coder back to work.

    harness_error: commands never ran (illegal flags, path escape, etc.)
    code_failed: tests ran and the program failed them
    passed: tests ran and succeeded
    """
    if sandbox_blocked or not executed or not tests:
        return "harness_error"
    if all(item.get("ok") for item in tests):
        return "passed"
    return "code_failed"


def _append_step(budget: SharedWebBudget, action: str, query: str, status: str = "completed", error: str = "") -> None:
    budget.steps.append(
        {
            "id": f"{action}-{len(budget.steps)+1}",
            "status": status,
            "action": action,
            "query": query[:800],
            "url": "",
            "error": error[:1000],
        }
    )


def _files_payload(workspace: Path) -> str:
    chunks = []
    for rel in code_sandbox.list_files(workspace):
        chunks.append(f"### {rel}\n```\n{code_sandbox.read_workspace_file(workspace, rel)}\n```")
    return "\n\n".join(chunks) or "(empty workspace)"


def _format_tool_result(plan: str, files: list[str], review: str, tests: list[dict[str, Any]], passed: bool) -> str:
    test_lines = []
    for item in tests:
        test_lines.append(
            f"$ {item.get('command')}\nexit {item.get('exit_code')}\n{item.get('stdout','')}\n{item.get('stderr','')}"
        )
    bodies = []
    return json.dumps(
        {
            "passed": passed,
            "plan": plan,
            "files": files,
            "review": review,
            "tests": tests,
            "instruction": (
                "Use these verified files in the final answer. Do not rewrite them unless the user asked for more changes."
                if passed
                else "Verification did not fully pass. Present the latest files, the review, and the real test output."
            ),
        },
        ensure_ascii=False,
    ) + ("" if not bodies else "")


async def run_verified_coding(
    *,
    task: str,
    api_client: httpx.AsyncClient,
    base_url: str,
    api_key: str,
    model: str,
    config: dict[str, Any],
    effort: str,
    timeout: int,
    stopped: Callable[[], bool],
    update: Callable[[dict[str, Any]], Awaitable[None]],
    budget: SharedWebBudget,
    backend: str,
    clients: dict[str, Any],
    parent_answer: str,
    parent_reasoning: str,
    parent_usage: dict[str, Any],
    language: str = "",
    complete_round: Optional[Callable[..., Awaitable[dict[str, Any]]]] = None,
    run_tests: Optional[Callable[[list[str], Path], list[dict[str, Any]]]] = None,
) -> dict[str, Any]:
    """Drive coder then reviewer until tests pass or the iteration cap is hit."""
    if coding_job_active.get():
        raise RuntimeError("编码任务进行中，不能再次套用写代码流程")
    active_token = coding_job_active.set(True)
    complete = complete_round or _complete_round
    tester = run_tests or code_sandbox.run_test_commands
    run_id = uuid.uuid4().hex
    workspace: Path | None = None
    plan = ""
    files: list[str] = []
    review = ""
    tests: list[dict[str, Any]] = []
    passed = False
    usage = dict(parent_usage)

    async def publish() -> None:
        await update(
            {
                "answer": parent_answer,
                "reasoning": parent_reasoning,
                "searches": budget.steps,
                "usage": usage,
                "sources": list(budget.sources.values()),
            }
        )

    try:
        workspace = code_sandbox.create_workspace(settings.data_dir, run_id)
        code_sandbox.cleanup_old_workspaces(settings.data_dir)
        brief = " ".join(str(task or "").split())
        if language:
            brief += f" Language: {language}."
        if not brief:
            raise ValueError("编码任务不能为空")
        feedback = ""
        for iteration in range(1, MAX_AGENT_ITERATIONS + 1):
            if stopped():
                raise InterruptedError("stopped")
            coder_messages = [
                {
                    "role": "system",
                    "content": (
                        "You are the coding agent. First think of a short plan, then call submit_code with complete files. "
                        "Do not call write_and_verify_code. Do not write a generator script that only emits the real program. "
                        "If the user wants HTML/JS, submit those files directly. Include tests in the same submit_code call when useful. "
                        "You may use web_search/fetch_webpage only if necessary and only within the shared quota. "
                        f"{_quota_note(budget)}"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Iteration {iteration}/{MAX_AGENT_ITERATIONS}.\nTask:\n{brief}\n\n"
                        f"Current workspace:\n{_files_payload(workspace)}\n\n"
                        f"Reviewer/test feedback:\n{feedback or '(none yet)'}"
                    ),
                },
            ]
            submitted = False
            arguments: dict[str, Any] = {}
            for _round in range(MAX_AGENT_TOOL_ROUNDS):
                tools = [SUBMIT_CODE_TOOL]
                if iteration < MAX_AGENT_ITERATIONS:
                    tools = [*_web_tools(backend, budget), SUBMIT_CODE_TOOL]
                result = await complete(
                    api_client=api_client,
                    base_url=base_url,
                    api_key=api_key,
                    model=model,
                    messages=coder_messages,
                    tools=tools,
                    config=config,
                    effort=effort,
                    timeout=timeout,
                    stopped=stopped,
                )
                usage = _merge_usage(usage, result.get("usage") or {})
                calls = result.get("tool_calls") or []
                if not calls:
                    harvested = harvest_files_from_text(str(result.get("content") or ""))
                    if harvested:
                        plan = plan or "Recovered files from assistant markdown."
                        written = code_sandbox.write_files(workspace, harvested)
                        files = written
                        _append_step(budget, "code_plan", plan)
                        _append_step(budget, "code_write", "、".join(written) + "（从正文代码块回收）")
                        submitted = True
                        await publish()
                        break
                    if result.get("content"):
                        coder_messages.append({"role": "assistant", "content": result["content"]})
                        coder_messages.append({"role": "user", "content": "Do not answer in prose. Call submit_code with the actual files."})
                    continue
                call = calls[0]
                function = call.get("function") or {}
                name = str(function.get("name") or "")
                try:
                    arguments = json.loads(str(function.get("arguments") or "{}"))
                except json.JSONDecodeError:
                    arguments = {}
                coder_messages.append(
                    {"role": "assistant", "content": result.get("content") or "", "tool_calls": [call]}
                )
                if name in {"web_search", "fetch_webpage"}:
                    try:
                        tool_text = await _run_web_tool(
                            name, arguments if isinstance(arguments, dict) else {},
                            budget=budget, backend=backend, clients=clients, model=model, stopped=stopped,
                        )
                    except Exception as exc:
                        tool_text = f"{name} 失败：{exc}"
                    coder_messages.append({"role": "tool", "tool_call_id": call.get("id"), "content": tool_text})
                    await publish()
                    continue
                if name != "submit_code" or not isinstance(arguments, dict):
                    coder_messages.append({"role": "tool", "tool_call_id": call.get("id"), "content": "Call submit_code with plan and files."})
                    continue
                plan = str(arguments.get("plan") or "").strip()
                try:
                    written = code_sandbox.write_files(workspace, arguments.get("files") or [])
                except code_sandbox.SandboxError as exc:
                    coder_messages.append({"role": "tool", "tool_call_id": call.get("id"), "content": str(exc)})
                    continue
                files = written
                _append_step(budget, "code_plan", plan or "已完成计划")
                _append_step(budget, "code_write", "、".join(written))
                submitted = True
                await publish()
                break
            if not submitted:
                raise RuntimeError("编码 Agent 没有提交可审查的代码（未调用 submit_code，正文里也没有可回收的代码块）")

            reviewer_messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an independent reviewer. You did not write this code. "
                        "Find bugs, then call submit_review. "
                        "For Python, prefer `python3 -m unittest ...` or `python3 -m py_compile file.py`. "
                        "For a single HTML/JS page, leave test_commands empty; do not invent python -c or /dev/stdin tests. "
                        "For JS files you may use `node --check file.js`. Never use python -c. "
                        "Do not start another write_and_verify_code job just to create tests. "
                        f"{_quota_note(budget)}"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Task:\n{brief}\n\nPlan:\n{plan}\n\nFiles:\n{_files_payload(workspace)}\n\n"
                        f"Author test hint: {str((arguments or {}).get('test_hint') or '(none)')}"
                    ),
                },
            ]
            reviewed = False
            send_to_coder = False
            last_test_outcome = ""
            for _round in range(MAX_AGENT_TOOL_ROUNDS):
                tools = [SUBMIT_REVIEW_TOOL]
                if iteration < MAX_AGENT_ITERATIONS:
                    tools = [*_web_tools(backend, budget), SUBMIT_REVIEW_TOOL]
                result = await complete(
                    api_client=api_client,
                    base_url=base_url,
                    api_key=api_key,
                    model=model,
                    messages=reviewer_messages,
                    tools=tools,
                    config=config,
                    effort=effort,
                    timeout=timeout,
                    stopped=stopped,
                )
                usage = _merge_usage(usage, result.get("usage") or {})
                calls = result.get("tool_calls") or []
                if not calls:
                    if result.get("content"):
                        reviewer_messages.append({"role": "assistant", "content": result["content"]})
                        reviewer_messages.append({"role": "user", "content": "Call submit_review with test_commands."})
                    continue
                call = calls[0]
                function = call.get("function") or {}
                name = str(function.get("name") or "")
                try:
                    arguments = json.loads(str(function.get("arguments") or "{}"))
                except json.JSONDecodeError:
                    arguments = {}
                reviewer_messages.append(
                    {"role": "assistant", "content": result.get("content") or "", "tool_calls": [call]}
                )
                if name in {"web_search", "fetch_webpage"}:
                    try:
                        tool_text = await _run_web_tool(
                            name, arguments if isinstance(arguments, dict) else {},
                            budget=budget, backend=backend, clients=clients, model=model, stopped=stopped,
                        )
                    except Exception as exc:
                        tool_text = f"{name} 失败：{exc}"
                    reviewer_messages.append({"role": "tool", "tool_call_id": call.get("id"), "content": tool_text})
                    await publish()
                    continue
                if name != "submit_review" or not isinstance(arguments, dict):
                    reviewer_messages.append({"role": "tool", "tool_call_id": call.get("id"), "content": "Call submit_review."})
                    continue
                review = str(arguments.get("issues") or "").strip()
                claimed_pass = bool(arguments.get("passed"))
                commands = [str(item) for item in (arguments.get("test_commands") or []) if str(item).strip()]
                _append_step(
                    budget,
                    "code_review",
                    f"循环 {iteration}/{MAX_AGENT_ITERATIONS}。{review or ('审查通过' if claimed_pass else '审查未通过')}",
                )
                frontend = code_sandbox.is_frontend_only(workspace)
                static = code_sandbox.static_verify(workspace)
                sandbox_blocked = False
                executed = False
                if commands:
                    try:
                        tests = tester(list(commands), workspace)
                        executed = True
                    except code_sandbox.SandboxError as exc:
                        sandbox_blocked = True
                        tests = [{"command": "", "ok": False, "exit_code": 2, "stdout": "", "stderr": str(exc)}]
                else:
                    tests = []
                if frontend and (sandbox_blocked or not commands):
                    tests = [
                        {
                            "command": "static_verify",
                            "ok": static["ok"],
                            "exit_code": 0 if static["ok"] else 1,
                            "stdout": "、".join(static["files"]),
                            "stderr": "；".join(static["issues"]),
                        }
                    ]
                    sandbox_blocked = False
                    executed = True
                outcome = classify_test_run(tests=tests, sandbox_blocked=sandbox_blocked, executed=executed)
                last_test_outcome = outcome
                summary = "；".join(
                    f"{item.get('command') or 'test'}={'ok' if item.get('ok') else 'fail'}" for item in tests
                ) or "测试没有跑起来"
                if outcome == "harness_error":
                    _append_step(
                        budget,
                        "code_test",
                        f"循环 {iteration}/{MAX_AGENT_ITERATIONS}。测试命令没跑起来，不交给写代码 Agent。{summary}",
                        status="skipped",
                        error=(tests[-1].get("stderr") if tests else "no tests") or "测试命令无效",
                    )
                    reviewer_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.get("id"),
                            "content": (
                                "These test commands never executed. That is a harness error, not a code bug. "
                                "Call submit_review again with legal commands "
                                "(python3 -m unittest / python3 -m py_compile / node --check), "
                                "or empty test_commands for HTML. Do not ask the coder to fix this."
                            ),
                        }
                    )
                    await publish()
                    continue
                _append_step(
                    budget,
                    "code_test",
                    f"循环 {iteration}/{MAX_AGENT_ITERATIONS}。{summary}",
                    status="completed" if outcome == "passed" else "failed",
                    error="" if outcome == "passed" else (tests[-1].get("stderr") if tests else "code failed tests"),
                )
                passed = outcome == "passed"
                send_to_coder = outcome == "code_failed"
                reviewed = True
                await publish()
                break
            if not reviewed:
                if last_test_outcome == "harness_error":
                    review = (review + "\n" if review else "") + "测试命令一直没跑起来，已停止，不再让写代码 Agent 空转。"
                    break
                raise RuntimeError("审查 Agent 没有提交审查结果")
            if passed or not send_to_coder or iteration >= MAX_AGENT_ITERATIONS:
                break
            feedback = (
                f"Tests ran successfully and the code failed them.\n"
                f"Reviewer notes: {review}\n"
                f"Real test results: {json.dumps(tests, ensure_ascii=False)}\n"
                f"Fix the code. Loop {iteration + 1}/{MAX_AGENT_ITERATIONS}."
            )
        if not passed and iteration >= MAX_AGENT_ITERATIONS:
            review = (review + "\n" if review else "") + f"已达到 {MAX_AGENT_ITERATIONS} 轮编码上限，停止继续改。"
        return {
            "passed": passed,
            "plan": plan,
            "files": files,
            "review": review,
            "tests": tests,
            "usage": usage,
            "search_count": budget.search_count,
            "fetch_count": budget.fetch_count,
            "reader_enabled": budget.reader_enabled,
            "tool_result": _format_tool_result(plan, files, review, tests, passed),
            "workspace": str(workspace),
        }
    finally:
        coding_job_active.reset(active_token)
        if workspace is not None:
            code_sandbox.cleanup_workspace(workspace)
