from __future__ import annotations

import importlib.util
import inspect
import json
import shutil
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import httpx

from local_tools import SKILL_DIR, ensure_app_dirs, safe_filename


@dataclass
class LocalSkill:
    """A Python skill loaded from local_skills/*.py."""

    name: str
    description: str
    parameters: dict[str, Any]
    file_path: Path
    run: Callable[..., Any]

    def to_openai_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def to_anthropic_tool(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }


def import_skill_files(paths: list[str]) -> list[Path]:
    """Copy user-selected .py or .json skill files into local_skills/."""
    ensure_app_dirs()
    imported: list[Path] = []
    allowed_suffixes = {".py", ".json"}

    for raw_path in paths:
        source = Path(raw_path)
        if not source.exists() or not source.is_file() or source.suffix.lower() not in allowed_suffixes:
            continue
        if source.name.startswith("_"):
            continue

        stem = safe_filename(source.stem, "local_skill")
        suffix = source.suffix.lower()
        destination = SKILL_DIR / f"{stem}{suffix}"
        index = 2
        while destination.exists():
            destination = SKILL_DIR / f"{stem}_{index}{suffix}"
            index += 1

        shutil.copy2(source, destination)
        imported.append(destination)

    return imported


def load_local_skills() -> tuple[dict[str, LocalSkill], list[str]]:
    """Load all valid skills from local_skills/. Returns (skills, errors)."""
    ensure_app_dirs()
    skills: dict[str, LocalSkill] = {}
    errors: list[str] = []

    for path in sorted(list(SKILL_DIR.glob("*.py")) + list(SKILL_DIR.glob("*.json"))):
        if path.name.startswith("_"):
            continue
        try:
            if path.suffix.lower() == ".json":
                skill = _load_json_skill(path)
            else:
                skill = _load_one_skill(path)
            if skill.name in skills:
                errors.append(f"{path.name}: skill 名称重复，已跳过：{skill.name}")
                continue
            skills[skill.name] = skill
        except Exception as exc:
            errors.append(f"{path.name}: {exc}\n{traceback.format_exc(limit=2)}")

    return skills, errors


def _load_one_skill(path: Path) -> LocalSkill:
    module_name = f"novel_agent_local_skill_{path.stem}_{abs(hash((str(path), path.stat().st_mtime_ns)))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("无法创建模块加载器。")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    name = str(getattr(module, "SKILL_NAME", path.stem)).strip()
    if not name.isidentifier():
        raise ValueError(f"SKILL_NAME 必须是 Python 标识符/Function Calling 名称：{name}")

    run = getattr(module, "run", None)
    if not callable(run):
        raise ValueError("skill 文件必须提供可调用的 run() 函数。")

    description = str(
        getattr(module, "SKILL_DESCRIPTION", None)
        or inspect.getdoc(run)
        or f"本地导入 Skill：{name}"
    ).strip()

    parameters = getattr(module, "PARAMETERS", None)
    if not isinstance(parameters, dict):
        parameters = {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "用户查询或任务描述。"}
            },
        }

    return LocalSkill(
        name=name,
        description=description,
        parameters=parameters,
        file_path=path,
        run=run,
    )


def _safe_format(template: str, kwargs: dict[str, Any]) -> str:
    """Format a string with {key} placeholders. Missing keys are left as-is."""
    result = template
    for key, value in kwargs.items():
        result = result.replace("{" + key + "}", str(value))
    return result


def _load_json_skill(path: Path) -> LocalSkill:
    """Parse a .json skill definition file and return a LocalSkill with a dynamic run callable."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON parse error: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError("JSON root must be an object (dict).")

    name = str(data.get("name", "")).strip()
    if not name:
        raise ValueError("Missing required field: name")
    if not name.isidentifier():
        raise ValueError(f"name must be a valid Function Calling name: {name}")

    description = str(data.get("description", f"JSON Skill: {name}")).strip()

    parameters = data.get("parameters")
    if not isinstance(parameters, dict):
        parameters = {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search content"}
            },
        }

    run_type = str(data.get("run_type", "")).strip()
    run_config = data.get("run_config")
    if not isinstance(run_config, dict):
        run_config = {}

    # Build the run callable based on run_type
    if run_type == "static_response":
        template = str(run_config.get("response", ""))

        def _run(**kwargs: Any) -> str:
            return _safe_format(template, kwargs)

    elif run_type == "prompt_template":
        template = str(run_config.get("prompt", ""))

        def _run(**kwargs: Any) -> str:
            return _safe_format(template, kwargs)

    elif run_type == "http_api":
        url = str(run_config.get("url", "")).strip()
        if not url:
            raise ValueError("run_type=http_api requires run_config.url to be non-empty.")
        method = str(run_config.get("method", "POST")).upper().strip()
        headers = run_config.get("headers")
        if not isinstance(headers, dict):
            headers = {"Content-Type": "application/json"}
        body_template = run_config.get("body_template")

        def _run(**kwargs: Any) -> str:
            try:
                req_headers = {k: _safe_format(v, kwargs) for k, v in headers.items()}
                if body_template is not None:
                    body_str = _safe_format(str(body_template), kwargs)
                    try:
                        body = json.loads(body_str)
                    except json.JSONDecodeError:
                        body = body_str
                else:
                    body = kwargs
                timeout = httpx.Timeout(30.0, connect=15.0)
                with httpx.Client(timeout=timeout) as client:
                    if method == "GET":
                        resp = client.get(url, headers=req_headers, params=kwargs)
                    else:
                        resp = client.request(method, url, headers=req_headers, json=body)
                    resp.raise_for_status()
                    return resp.text[:8000]
            except Exception as exc:
                return f"HTTP API Skill call failed: {exc}"

    else:
        raise ValueError(
            f"Unsupported run_type: {run_type}. "
            f"Supported types: static_response, prompt_template, http_api"
        )

    return LocalSkill(
        name=name,
        description=description,
        parameters=parameters,
        file_path=path,
        run=_run,
    )


def run_local_skill(skill: LocalSkill, args: dict[str, Any]) -> str:
    """Execute a loaded skill with JSON tool-call arguments."""
    args = args or {}
    signature = inspect.signature(skill.run)
    params = signature.parameters

    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()):
        result = skill.run(**args)
    elif len(params) == 1:
        only_name = next(iter(params))
        if only_name in {"args", "payload", "kwargs"}:
            result = skill.run(args)
        elif only_name in args:
            result = skill.run(args[only_name])
        elif "query" in args:
            result = skill.run(args["query"])
        else:
            result = skill.run(**{only_name: ""})
    else:
        accepted = {name: value for name, value in args.items() if name in params}
        result = skill.run(**accepted)

    if result is None:
        return ""
    return str(result)


def local_skills_to_openai_specs(skills: dict[str, LocalSkill]) -> list[dict[str, Any]]:
    return [skill.to_openai_tool() for skill in skills.values()]


def local_skills_to_anthropic_specs(skills: dict[str, LocalSkill]) -> list[dict[str, Any]]:
    return [skill.to_anthropic_tool() for skill in skills.values()]
