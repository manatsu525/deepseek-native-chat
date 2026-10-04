"""Grok Build's source prompt, rendered for the tools available in this app."""
from __future__ import annotations

import ast
import re
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

TEMPLATE = (Path(__file__).parent / "templates/grok_system_prompt.md").read_text(encoding="utf-8")

def _condition(expression: str, values: dict) -> bool:
    def resolve(node):
        if isinstance(node, ast.Name):
            return values.get(node.id, False)
        if isinstance(node, ast.Attribute):
            base = resolve(node.value)
            return base.get(node.attr, False) if isinstance(base, dict) else False
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return not resolve(node.operand)
        if isinstance(node, ast.BoolOp):
            parts = [bool(resolve(item)) for item in node.values]
            return all(parts) if isinstance(node.op, ast.And) else any(parts)
        raise ValueError("Unsupported Grok prompt condition")
    return bool(resolve(ast.parse(expression, mode="eval").body))

def render_prompt(*, execute: bool) -> str:
    values = {"is_non_interactive": False, "memory_v2_enabled": False,
              "include_browser_verification": False, "system_reminders_enabled": False,
              "tools": {"by_kind": {"execute": "bash" if execute else False}}}
    # Preserve the upstream template; only render its actual feature switches.
    stack = []
    enabled = True
    output = []
    for part in re.split(r"(\$\{%[-]?\s*.*?\s*%\})", TEMPLATE):
        if part.startswith("${%"):
            directive = re.sub(r"^\$\{%[-]?\s*|\s*%\}$", "", part)
            if directive.startswith("if "):
                matched = _condition(directive[3:], values)
                stack.append([enabled, matched])
                enabled = enabled and matched
            elif directive.startswith("elif "):
                parent, matched = stack[-1]
                current = not matched and _condition(directive[5:], values)
                stack[-1][1] = matched or current
                enabled = parent and current
            elif directive == "else":
                parent, matched = stack[-1]
                enabled = parent and not matched
                stack[-1][1] = True
            elif directive == "endif":
                enabled = stack.pop()[0]
            else:
                raise ValueError("Unknown Grok prompt directive")
        elif enabled:
            output.append(part)
    text = "".join(output).replace("${{ system_prompt_label }}", "an AI assistant")
    text = text.replace("${{ tools.by_kind.execute }}", "bash").replace("${{ scratch_dir }}", "/tmp/")
    # Identity and session surface differ; do not impersonate xAI or a CLI.
    text = text.replace(" released by xAI", "")
    text = text.replace("an interactive CLI tool that helps users with software engineering tasks.",
                        "an interactive web assistant that helps users with software engineering tasks.")
    text = text.replace("an autonomous agent that completes software engineering tasks. There is no human operator in this session.",
                        "an assistant in a web chat that completes user requests, including software engineering tasks.")
    # Terminal-specific help has no corresponding documentation in this web UI.
    text = re.sub(r"\n<user_guide>.*?</user_guide>\n", "\n", text, flags=re.S)
    if "${" in text:
        raise ValueError("Unrendered Grok prompt variable")
    return text.strip()

WORKSPACE_RULES = {
    "full": "Files live in a persistent isolated conversation workspace at /workspace. bash commands operate on these same files; changes persist. The command environment has no network or access to other conversations. Paths are workspace-relative or absolute under /workspace.",
    "edit": "Files live in a persistent isolated conversation workspace. Paths are workspace-relative. File editing is available; command execution is unavailable.",
    "read_only": "Files live in a persistent isolated conversation workspace. Paths are workspace-relative. Only read-only operations and verification tools are available.",
}
FILES_DEFERRED_NOTE = 'Workspace tools are available through load_tools with groups ["files"].'
# Compatibility exports for integrations; no homegrown file/research policy is injected.
IDENTITY = render_prompt(execute=False)
TOOL_RULES = ""
AGENT_RULES = "Agent tools operate on the actual host. Relative paths resolve from /home/share; absolute paths are supported. Skills and API configuration are shared; only administrators may change them. Conversation management is restricted to the current account."
AGENT_RESEARCH = ""
RESEARCH_RULES = ""
WEB_RULES = {"parallel": "Web acquisition uses Parallel Search MCP.", "legacy": "Web acquisition uses DuckDuckGo and Jina Reader.", "keyless": "Web acquisition uses the selected MCP provider."}

def files_group_rules(workspace_access: str) -> str:
    return WORKSPACE_RULES.get(workspace_access, WORKSPACE_RULES["full"])

def date_context(user_timezone: str) -> str:
    try:
        timezone = ZoneInfo(user_timezone)
    except (ZoneInfoNotFoundError, ValueError):
        timezone = ZoneInfo("UTC")
    current = datetime.now(timezone)
    return "Today's date: " + current.strftime("%A %b ") + str(current.day) + current.strftime(", %Y")

def build_system_prompt(*, agent_mode: bool, web_enabled: bool, web_backend: str,
                        workspace_access: str | None, user_timezone: str,
                        skills_prompt: str = "", file_tools_loaded: bool = True) -> str:
    sections = [render_prompt(execute=agent_mode or workspace_access == "full")]
    if agent_mode:
        sections.append(AGENT_RULES)
    elif workspace_access in WORKSPACE_RULES:
        sections.append(files_group_rules(workspace_access) if file_tools_loaded else FILES_DEFERRED_NOTE)
    if web_enabled:
        sections.append(WEB_RULES.get(web_backend, WEB_RULES["keyless"]))
    return "\n\n".join(sections)


def user_message_prefix(*, workspace_path: str, user_timezone: str,
                        skills_prompt: str = "", rules: list[tuple[str, str]] = ()) -> str:
    """Upstream first-user-message context, separate from the stable system prompt."""
    sections = [f"<user_info>\nOS Version: Linux\nShell: /bin/bash\nWorkspace Path: {workspace_path}\n{date_context(user_timezone)}\n</user_info>"]
    if rules:
        from html import escape
        intro = "The rules section has a number of possible rules/memories/context that you should consider. In each subsection, we provide instructions about what information the subsection contains and how you should consider/follow the contents of the subsection."
        entries = []
        for path, body in rules:
            for tag in ("rules", "system-reminder", "system_reminder"):
                body = body.replace(f"<{tag}>", f"&lt;{tag}>").replace(f"</{tag}>", f"&lt;/{tag}>")
            entries.append(f'<always_applied_workspace_rule name="{escape(path, quote=True)}">\n{body.strip()}\n</always_applied_workspace_rule>')
        sections.append('<rules>\n' + intro + '\n\n\n<always_applied_workspace_rules description="These are workspace-level rules that the agent must always follow.">\n' + '\n\n'.join(entries) + '\n</always_applied_workspace_rules>\n</rules>')
    if skills_prompt.strip():
        sections.append(skills_prompt.strip())
    return "\n\n".join(sections)
