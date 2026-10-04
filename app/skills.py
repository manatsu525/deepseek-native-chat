"""Skill discovery and management for the host-level Agent mode.

Skills are deliberately plain directories containing a ``SKILL.md`` file. The
format follows the open Agent Skills convention while keeping dependencies
minimal: YAML front matter is parsed only for name and description.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from html import escape

from .config import settings


BUILTIN_SKILLS_ROOT = Path(__file__).resolve().parent.parent / "skills"
USER_SKILLS_ROOT = settings.data_dir / "skills"
SKILL_CONFIG_PATH = settings.data_dir / "agent-skills.json"

DEFAULT_SKILLS = ()

# These IDs identify the shipped pre-Grok default configuration, not a user's
# deliberately empty selection. Keep migration limited to that obsolete list.
RETIRED_DEFAULT_SKILLS = frozenset({
    "conversation-management", "executing-plans", "frontend-page-management",
    "react-best-practices", "requesting-code-review", "skill-management",
    "systematic-debugging", "test-driven-development", "verification-before-completion",
    "web-design-guidelines", "writing-plans",
})


@dataclass(frozen=True)
class Skill:
    skill_id: str
    name: str
    description: str
    path: Path
    builtin: bool
    disable_model_invocation: bool = False

    @property
    def markdown_path(self) -> Path:
        return self.path / "SKILL.md"


def _frontmatter(text: str, fallback_name: str) -> tuple[str, str]:
    name = fallback_name
    description = ""
    lines = text.splitlines()
    if lines and lines[0].strip() == "---":
        for index, line in enumerate(lines[1:], 1):
            if line.strip() == "---":
                break
            match = re.match(r"^\s*(name|description)\s*:\s*(.*?)\s*$", line, re.I)
            if not match:
                continue
            value = match.group(2).strip().strip("\"'")
            if value in {">", ">-", ">+", "|", "|-", "|+"}:
                continuation = []
                for following in lines[index + 1:]:
                    if following.strip() and not following.startswith((" ", "\t")):
                        break
                    continuation.append(following.strip())
                value = " ".join(continuation).strip()
            if match.group(1).casefold() == "name":
                name = value or name
            else:
                description = value
    if not description:
        for line in lines:
            if line.strip() and not line.startswith("#") and line.strip() != "---":
                description = line.strip()
                break
    return name[:120], description[:500]


def _safe_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value or "").strip()).strip(".-")
    if not cleaned:
        raise ValueError("Skill 名称不能为空")
    return cleaned[:100]


class SkillRegistry:
    """Discover built-in and user-installed skills without importing code."""

    def __init__(self, builtin_root: Path = BUILTIN_SKILLS_ROOT, user_root: Path = USER_SKILLS_ROOT) -> None:
        self.builtin_root = Path(builtin_root)
        self.user_root = Path(user_root)

    def _scan_root(self, root: Path, builtin: bool) -> list[Skill]:
        if not root.is_dir():
            return []
        result: list[Skill] = []
        for markdown in sorted(root.rglob("SKILL.md"), key=lambda item: item.as_posix().casefold()):
            if not markdown.is_file() or markdown.is_symlink():
                continue
            folder = markdown.parent
            try:
                relative = folder.relative_to(root).as_posix()
            except ValueError:
                continue
            skill_id = relative if relative != "." else folder.name
            try:
                text = markdown.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            name, description = _frontmatter(text, folder.name)
            frontmatter = text.split("---", 2)[1] if text.startswith("---\n") and text.count("---") >= 2 else ""
            disabled = bool(re.search(r"^disable-model-invocation\s*:\s*(?:true|yes|1)\s*$", frontmatter, re.M | re.I))
            result.append(Skill(skill_id, name, description, folder, builtin, disabled))
        return result

    def all(self) -> list[Skill]:
        found: dict[str, Skill] = {item.skill_id: item for item in self._scan_root(self.builtin_root, True)}
        found.update({item.skill_id: item for item in self._scan_root(self.user_root, False)})
        project = Path(os.getenv("AGENT_WORKSPACE_ROOT", "/home/share"))
        for root in (Path("/root/.agents/skills"), Path("/root/.grok/skills"),
                     project / ".agents/skills", project / ".grok/skills"):
            for item in self._scan_root(root, False):
                scoped = Skill("discovered:" + item.skill_id, item.name, item.description, item.path, True, item.disable_model_invocation)
                found[scoped.skill_id] = scoped
        return sorted(found.values(), key=lambda item: (not item.builtin, item.skill_id.casefold()))

    def find(self, skill_id: str) -> Optional[Skill]:
        wanted = str(skill_id or "").strip()
        if not wanted:
            return None
        return next((item for item in self.all() if item.skill_id == wanted or item.name == wanted), None)

    def read(self, skill_id: str) -> str:
        skill = self.find(skill_id)
        if skill is None:
            raise ValueError(f"Skill 不存在：{skill_id}")
        return skill.markdown_path.read_text(encoding="utf-8")

    def _configured(self) -> Optional[list[str]]:
        try:
            raw = json.loads(SKILL_CONFIG_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return None
        values = raw.get("enabled") if isinstance(raw, dict) else None
        return [str(item) for item in values if str(item).strip()] if isinstance(values, list) else None

    def enabled_ids(self) -> list[str]:
        configured = self._configured()
        if configured and set(configured) <= RETIRED_DEFAULT_SKILLS:
            # Only migrate when none of the listed skills still exists. This
            # preserves customized installations and explicit empty/disabled lists.
            if not any(item.skill_id in configured for item in self.all()):
                configured = None
        if configured is None:
            return [item.skill_id for item in self.all() if not item.builtin or item.skill_id.startswith("discovered:")]
        available = {item.skill_id for item in self.all()}
        return [item for item in configured if item in available]

    def set_enabled(self, skill_id: str, enabled: bool) -> list[str]:
        skill = self.find(skill_id)
        if skill is None:
            raise ValueError(f"Skill 不存在：{skill_id}")
        values = self.enabled_ids()
        if enabled and skill.skill_id not in values:
            values.append(skill.skill_id)
        if not enabled:
            values = [item for item in values if item != skill.skill_id]
        SKILL_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        SKILL_CONFIG_PATH.write_text(json.dumps({"enabled": values}, ensure_ascii=False, indent=2), encoding="utf-8")
        return values

    def prompt(self) -> str:
        # Names and descriptions only: full bodies cost thousands of tokens
        # every round, and skill_read fetches one on demand.
        sections: list[str] = []
        for skill_id in self.enabled_ids():
            skill = self.find(skill_id)
            if skill is None or skill.disable_model_invocation:
                continue
            from .grok_tools import utf8_prefix
            description = utf8_prefix(" ".join(str(skill.description or "").split()), 400)
            sections.append(f'<agent_skill fullPath="{escape(str(skill.markdown_path), quote=True)}">{escape(description, quote=False)}</agent_skill>')
        if not sections:
            return ""
        return (
            "<agent_skills>\nThe following skills are available for use. Load the applicable skill with the skill tool.\n\n"
            + "\n".join(sections)
            + "\n</agent_skills>"
        )

    def invoke(self, name: str, args: str = "", session_id: str = "") -> str:
        from .grok_tools import utf8_prefix
        skill = self.find(name)
        if skill is None or skill.skill_id not in self.enabled_ids():
            raise ValueError(f"Skill unavailable or disabled: {name}")
        if skill.disable_model_invocation:
            raise ValueError(f"Skill does not permit model invocation: {name}")
        body = skill.markdown_path.read_text(encoding="utf-8")
        if body.startswith("---\n"):
            parts = body.split("---", 2)
            if len(parts) == 3:
                body = parts[2].lstrip("\r\n")
        if len(body.encode()) > 100_000:
            prefix = utf8_prefix(body, 100_000)
            body = prefix.rsplit("\n", 1)[0] + f"\n[Skill content truncated. Read {skill.markdown_path} with offset and limit to continue.]"
        positional = args.split()
        consumed = bool(re.search(r"\$ARGUMENTS|\$\d+", body))
        body = re.sub(r"\$ARGUMENTS\[(\d+)\]|\$(\d+)", lambda match:
                      positional[int(match[1] or match[2])] if int(match[1] or match[2]) < len(positional) else "", body)
        body = body.replace("$ARGUMENTS", args)
        body = body.replace("${SKILL_DIR}", str(skill.path)).replace("${CLAUDE_SKILL_DIR}", str(skill.path))
        body = body.replace("${SESSION_ID}", session_id).replace("${CLAUDE_SESSION_ID}", session_id)
        if args and not consumed:
            body += "\n\n**ARGUMENTS:** " + args
        return (f'<skill name="{escape(skill.name, quote=True)}" description="{escape(skill.description, quote=True)}" '
                f'path="{escape(str(skill.markdown_path), quote=True)}">\n{body}\n</skill>')

    def install(self, source: str, name: str = "") -> Skill:
        source_value = str(source or "").strip()
        if not source_value:
            raise ValueError("Skill 来源不能为空")
        target_name = _safe_id(name or Path(source_value.rstrip("/")).stem or "skill")
        destination = self.user_root / target_name
        self.user_root.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise ValueError(f"Skill 目录已存在：{target_name}")
        if source_value.startswith(("http://", "https://", "git@")) or source_value.endswith(".git"):
            try:
                completed = subprocess.run(
                    ["git", "clone", "--depth", "1", source_value, str(destination)],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=300, check=False,
                )
            except subprocess.TimeoutExpired as exc:
                shutil.rmtree(destination, ignore_errors=True)
                raise RuntimeError("git clone 超时（超过 300 秒）") from exc
            if completed.returncode != 0:
                shutil.rmtree(destination, ignore_errors=True)
                raise RuntimeError((completed.stderr or completed.stdout or "git clone 失败")[-4_000:])
        else:
            source_path = Path(source_value).expanduser().resolve()
            if not source_path.is_dir():
                raise ValueError("本地 Skill 来源必须是目录")
            shutil.copytree(source_path, destination)
        if not (destination / "SKILL.md").is_file():
            shutil.rmtree(destination, ignore_errors=True)
            raise ValueError("Skill 目录中没有 SKILL.md")
        skill = next((item for item in self._scan_root(self.user_root, False) if item.path == destination), None)
        if skill is None:
            raise ValueError("Skill 安装后无法读取 SKILL.md")
        return skill

    def remove(self, skill_id: str) -> None:
        skill = self.find(skill_id)
        if skill is None:
            raise ValueError(f"Skill 不存在：{skill_id}")
        if skill.builtin:
            raise ValueError("内置 Skill 不能删除；可以禁用它")
        self.set_enabled(skill.skill_id, False)
        shutil.rmtree(skill.path)
