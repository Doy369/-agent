from __future__ import annotations

import json
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable

import httpx

from local_tools import (
    generate_image,
    local_archive_and_stats,
    local_text_audit,
    search_local_knowledge,
)
from skill_loader import (
    LocalSkill,
    load_local_skills,
    local_skills_to_anthropic_specs,
    local_skills_to_openai_specs,
    run_local_skill,
)


LogCallback = Callable[[str], None]


SYSTEM_PROMPT = """你是一名金牌网络小说作家、资深剧情架构师和连载编辑。
你的任务是帮助用户创作高完成度、强爽点、强连贯性的中文类型小说。

写作要求：
1. 严格尊重用户给定的世界观、人物设定、章节大纲和前文事实。
2. 每章要有清晰的目标、冲突、反转或钩子，结尾尽量留下继续阅读的动力。
3. 文风要贴合用户选择的类型；避免空洞套话，优先写动作、选择、代价和细节。
4. 不要凭空推翻已有设定；如需要补充设定，要自然、可追踪。
5. 当需要历史、专业知识、硬科幻设定或本地设定集时，可以使用工具。
6. 如果剧情出现适合插图的高光场景，可以调用 generate_image 工具生成插图。
"""


OPENAI_TOOL_SPECS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_info",
            "description": "联网查资料技能。用于查询历史背景、专业术语、硬科幻知识等。当前实现为 Mock，可替换真实搜索 API。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "要搜索的关键词或问题。"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_image",
            "description": "AI 插图技能。根据章节高光场景生成插图。当前实现为本地 Mock PNG，可替换 Stable Diffusion 或 DALL-E API。",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "插图画面描述。"}
                },
                "required": ["prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_local_knowledge",
            "description": "本地设定集检索。读取 knowledge.txt 与 knowledge/*.txt|*.md 中的中英文知识文件，支持跨语言语义搜索。查找世界观、境界、人物能力等设定。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "要检索的本地设定关键词。"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "local_text_audit",
            "description": "本地敏感词与错别字检测。无需消耗 API Token。",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "要检查的章节正文。"}
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "local_archive_and_stats",
            "description": "本地章节归档与全书统计。遍历 history/ 并生成书籍大纲进度报表.txt。",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
]


ANTHROPIC_TOOL_SPECS: list[dict[str, Any]] = [
    {
        "name": item["function"]["name"],
        "description": item["function"]["description"],
        "input_schema": item["function"]["parameters"],
    }
    for item in OPENAI_TOOL_SPECS
]


@dataclass
class AgentConfig:
    """GUI 层传入的模型配置。"""

    api_key: str
    base_url: str
    model: str
    provider: str = "openai"
    max_words: int = 2500
    temperature: float = 0.82
    timeout_seconds: float = 180.0


@dataclass
class ToolResult:
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationResult:
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


class NovelAgent:
    """负责 Prompt 组织、上下文记忆、Function Calling 和 API 请求。"""

    def __init__(self) -> None:
        self._active_client: httpx.Client | None = None
        self._cancel_requested = False
        self.local_skills: dict[str, LocalSkill] = {}
        self.local_skill_errors: list[str] = []
        self.reload_local_skills()

    def cancel_current_request(self) -> None:
        """从 GUI 线程触发中止；关闭 httpx client 可尽快打断阻塞请求。"""
        self._cancel_requested = True
        client = self._active_client
        if client is not None:
            try:
                client.close()
            except Exception:
                pass

    def reload_local_skills(self) -> list[str]:
        """Reload Python skills imported into local_skills/."""
        skills, errors = load_local_skills()
        builtin_names = {item["function"]["name"] for item in OPENAI_TOOL_SPECS}
        for name in list(skills):
            if name in builtin_names:
                errors.append(f"{name}: 与内置工具重名，已跳过。")
                del skills[name]
        self.local_skills = skills
        self.local_skill_errors = errors
        return errors

    def get_local_skill_summary(self) -> str:
        if not self.local_skills:
            return "当前未加载本地 Skill。"
        lines = ["已加载本地 Skill："]
        for skill in self.local_skills.values():
            lines.append(f"- {skill.name}: {skill.description} ({skill.file_path.name})")
        return "\n".join(lines)

    def generate_outline(
        self,
        config: AgentConfig,
        novel_setting: str,
        style: str,
        current_text: str,
        log_callback: LogCallback | None = None,
    ) -> GenerationResult:
        auto_context = self._auto_local_context(novel_setting, current_text, log_callback)
        user_prompt = f"""请基于用户设定，生成一份可连载执行的全书/分卷大纲。

小说类型：{style}

用户设定：
{novel_setting or "用户暂未填写详细设定，请先给出一个可扩展的原创框架。"}

{auto_context}

输出要求：
1. 给出书名候选、核心卖点、主线矛盾、主要人物弧光。
2. 至少规划 3 卷，每卷包含目标、冲突升级、关键反转和结尾钩子。
3. 给出前 10 章章节标题和每章一句话剧情。
4. 保留后续扩展空间，不要把所有谜底一次性揭开。
"""
        return self._call_model(
            config,
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            log_callback=log_callback,
        )

    def generate_next_chapter(
        self,
        config: AgentConfig,
        novel_setting: str,
        style: str,
        outline: str,
        chapters: list[dict[str, str]],
        current_text: str,
        log_callback: LogCallback | None = None,
    ) -> GenerationResult:
        memory_context = self._build_memory_context(novel_setting, outline, chapters)
        auto_context = self._auto_local_context(novel_setting + "\n" + outline, current_text, log_callback)
        next_index = len(chapters) + 1
        user_prompt = f"""请生成第 {next_index} 章正文。

小说类型：{style}
目标篇幅：约 {config.max_words} 字

{memory_context}

{auto_context}

写作要求：
1. 章节标题放在第一行。
2. 承接前文，不要重启故事，不要重复大段设定说明。
3. 至少安排一个推进主线的行动、一个人物选择、一个结尾钩子。
4. 正文可直接发布，避免解释“我将如何写”。
"""
        return self._call_model(
            config,
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            log_callback=log_callback,
        )

    def polish_text(
        self,
        config: AgentConfig,
        selected_text: str,
        style: str,
        instruction: str,
        log_callback: LogCallback | None = None,
    ) -> GenerationResult:
        if not selected_text.strip():
            raise ValueError("请先选中要润色的文本，或让程序使用当前全文。")

        auto_context = self._auto_local_context(instruction, selected_text, log_callback)
        user_prompt = f"""请润色以下小说文本。

小说类型：{style}
润色目标：{instruction or "增强画面感、节奏和情绪张力，同时保持原剧情事实不变。"}

{auto_context}

原文：
{selected_text}

输出要求：
1. 只输出润色后的文本。
2. 不改变核心人物关系、剧情事实和叙事视角。
3. 可适度扩写动作、环境和心理，但不要灌水。
"""
        return self._call_model(
            config,
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            log_callback=log_callback,
        )

    def generate_image_for_chapter(
        self,
        prompt: str,
        log_callback: LogCallback | None = None,
    ) -> GenerationResult:
        """手动按钮触发的插图生成，不需要消耗文本模型 Token。"""
        tool_result = self._execute_tool("generate_image", {"prompt": prompt}, log_callback)
        return GenerationResult(tool_result.content, tool_result.metadata)

    def _call_model(
        self,
        config: AgentConfig,
        messages: list[dict[str, Any]],
        log_callback: LogCallback | None = None,
    ) -> GenerationResult:
        self._cancel_requested = False
        if not config.api_key.strip():
            raise ValueError("请先填写 API Key。")

        provider = (config.provider or "openai").lower()
        if provider == "anthropic":
            return self._call_anthropic(config, messages, log_callback)
        return self._call_openai_compatible(config, messages, log_callback)

    def _call_openai_compatible(
        self,
        config: AgentConfig,
        messages: list[dict[str, Any]],
        log_callback: LogCallback | None,
    ) -> GenerationResult:
        """OpenAI Chat Completions 兼容格式，适配 DeepSeek、OpenAI 兼容网关等。"""
        url = self._endpoint(config.base_url, "/chat/completions", known_tail="/chat/completions")
        headers = {
            "Authorization": f"Bearer {config.api_key.strip()}",
            "Content-Type": "application/json",
        }
        working_messages = deepcopy(messages)
        metadata: dict[str, Any] = {"tool_results": [], "image_paths": []}

        timeout = httpx.Timeout(config.timeout_seconds, connect=30.0)
        with httpx.Client(timeout=timeout) as client:
            self._active_client = client
            try:
                for _ in range(4):
                    self._raise_if_cancelled()
                    payload = {
                        "model": config.model,
                        "messages": working_messages,
                        "temperature": config.temperature,
                        "max_tokens": self._max_tokens(config.max_words),
                        "tools": self._openai_tool_specs(),
                        "tool_choice": "auto",
                    }
                    response = client.post(url, headers=headers, json=payload)
                    self._raise_for_status(response)
                    data = response.json()
                    message = data["choices"][0]["message"]
                    tool_calls = message.get("tool_calls") or []

                    if tool_calls:
                        working_messages.append(message)
                        for call in tool_calls:
                            function = call.get("function", {})
                            name = function.get("name", "")
                            args = self._loads_json(function.get("arguments", "{}"))
                            tool_result = self._execute_tool(name, args, log_callback)
                            metadata["tool_results"].append({"name": name, "content": tool_result.content})
                            if image_path := tool_result.metadata.get("image_path"):
                                metadata["image_paths"].append(image_path)
                            working_messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": call.get("id"),
                                    "name": name,
                                    "content": tool_result.content,
                                }
                            )
                        continue

                    content = self._extract_text(message.get("content", ""))
                    return GenerationResult(content=content, metadata=metadata)

                raise RuntimeError("工具调用轮次过多，请缩小需求或关闭自动工具调用后重试。")
            finally:
                self._active_client = None

    def _call_anthropic(
        self,
        config: AgentConfig,
        messages: list[dict[str, Any]],
        log_callback: LogCallback | None,
    ) -> GenerationResult:
        """Anthropic Messages API 格式，保留同一套本地工具执行能力。"""
        url = self._endpoint(config.base_url, "/v1/messages", known_tail="/messages")
        system_text, anthropic_messages = self._to_anthropic_messages(messages)
        headers = {
            "x-api-key": config.api_key.strip(),
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        metadata: dict[str, Any] = {"tool_results": [], "image_paths": []}

        timeout = httpx.Timeout(config.timeout_seconds, connect=30.0)
        with httpx.Client(timeout=timeout) as client:
            self._active_client = client
            try:
                for _ in range(4):
                    self._raise_if_cancelled()
                    payload = {
                        "model": config.model,
                        "system": system_text,
                        "messages": anthropic_messages,
                        "max_tokens": self._max_tokens(config.max_words),
                        "temperature": config.temperature,
                        "tools": self._anthropic_tool_specs(),
                    }
                    response = client.post(url, headers=headers, json=payload)
                    self._raise_for_status(response)
                    data = response.json()
                    content_blocks = data.get("content", [])
                    tool_uses = [block for block in content_blocks if block.get("type") == "tool_use"]

                    if tool_uses:
                        anthropic_messages.append({"role": "assistant", "content": content_blocks})
                        tool_results = []
                        for block in tool_uses:
                            name = block.get("name", "")
                            args = block.get("input") or {}
                            tool_result = self._execute_tool(name, args, log_callback)
                            metadata["tool_results"].append({"name": name, "content": tool_result.content})
                            if image_path := tool_result.metadata.get("image_path"):
                                metadata["image_paths"].append(image_path)
                            tool_results.append(
                                {
                                    "type": "tool_result",
                                    "tool_use_id": block.get("id"),
                                    "content": tool_result.content,
                                }
                            )
                        anthropic_messages.append({"role": "user", "content": tool_results})
                        continue

                    text = "\n".join(
                        block.get("text", "") for block in content_blocks if block.get("type") == "text"
                    )
                    return GenerationResult(content=text.strip(), metadata=metadata)

                raise RuntimeError("工具调用轮次过多，请缩小需求或稍后重试。")
            finally:
                self._active_client = None

    def _execute_tool(
        self,
        name: str,
        args: dict[str, Any],
        log_callback: LogCallback | None,
    ) -> ToolResult:
        """统一执行大模型 Function Calling 或 GUI 按钮触发的工具。"""
        self._raise_if_cancelled()
        args = args or {}

        if name == "search_info":
            query = str(args.get("query", "")).strip()
            self._log(log_callback, f"Agent 正在使用【联网查资料】技能搜索：{query}")
            content = self._mock_search_info(query)
            self._log(log_callback, "【联网查资料】技能完成。")
            return ToolResult(content, {"tool": name})

        if name == "generate_image":
            prompt = str(args.get("prompt", "")).strip() or "本章高光场景"
            self._log(log_callback, f"Agent 正在使用【AI 插图】技能生成：{prompt[:80]}")
            image_path = generate_image(prompt)
            self._log(log_callback, f"【AI 插图】技能完成：{image_path}")
            return ToolResult(f"已生成本章插图：{image_path}", {"tool": name, "image_path": image_path})

        if name == "search_local_knowledge":
            query = str(args.get("query", "")).strip()
            self._log(log_callback, f"Agent 正在使用【本地设定集检索】技能：{query}")
            content = search_local_knowledge(query)
            self._log(log_callback, "【本地设定集检索】技能完成。")
            return ToolResult(content, {"tool": name})

        if name == "local_text_audit":
            text = str(args.get("text", ""))
            self._log(log_callback, "Agent 正在使用【本地错字检查】技能。")
            content = local_text_audit(text)
            self._log(log_callback, "【本地错字检查】技能完成。")
            return ToolResult(content, {"tool": name})

        if name == "local_archive_and_stats":
            self._log(log_callback, "Agent 正在使用【本地归档统计】技能。")
            content = local_archive_and_stats()
            self._log(log_callback, "【本地归档统计】技能完成。")
            return ToolResult(content, {"tool": name})

        if name in self.local_skills:
            skill = self.local_skills[name]
            self._log(log_callback, f"Agent 正在使用【导入 Skill：{name}】。")
            content = run_local_skill(skill, args)
            self._log(log_callback, f"【导入 Skill：{name}】执行完成。")
            return ToolResult(content, {"tool": name, "local_skill": True, "skill_file": str(skill.file_path)})

        return ToolResult(f"未知工具：{name}", {"tool": name, "error": True})

    def _openai_tool_specs(self) -> list[dict[str, Any]]:
        return OPENAI_TOOL_SPECS + local_skills_to_openai_specs(self.local_skills)

    def _anthropic_tool_specs(self) -> list[dict[str, Any]]:
        return ANTHROPIC_TOOL_SPECS + local_skills_to_anthropic_specs(self.local_skills)

    def _auto_local_context(
        self,
        prompt_text: str,
        current_text: str,
        log_callback: LogCallback | None,
    ) -> str:
        """根据用户输入中的关键词自动运行本地技能，并把结果注入上下文。"""
        combined = f"{prompt_text}\n{current_text}".strip()
        blocks: list[str] = []

        if any(keyword in combined for keyword in ("查看我的设定", "检索本地设定", "设定集", "知识库")):
            query = prompt_text.strip()[:120] or "主角 设定 世界观"
            self._log(log_callback, f"自动触发【本地设定集检索】：{query}")
            blocks.append("【自动本地设定检索结果】\n" + search_local_knowledge(query))

        if any(keyword in combined for keyword in ("查资料", "联网", "历史背景", "硬科幻", "专业知识", "考据")):
            query = prompt_text.strip()[:120] or "小说专业资料"
            self._log(log_callback, f"自动触发【联网查资料】：{query}")
            blocks.append("【自动联网资料结果】\n" + self._mock_search_info(query))

        if any(keyword in combined for keyword in ("检查一下错字", "检查错字", "错别字", "敏感词")):
            audit_target = current_text.strip() or prompt_text.strip()
            self._log(log_callback, "自动触发【本地错字检查】。")
            blocks.append("【自动本地审校报告】\n" + local_text_audit(audit_target))

        if not blocks:
            return ""
        return "\n\n".join(blocks)

    def _build_memory_context(
        self,
        novel_setting: str,
        outline: str,
        chapters: list[dict[str, str]],
    ) -> str:
        """简单 Context Window：设定 + 大纲 + 最近三章节选。"""
        lines = ["上下文记忆：", "", "【小说设定】", novel_setting.strip() or "暂无。"]

        if outline.strip():
            lines.extend(["", "【全书/分卷大纲】", self._clip(outline.strip(), 4000)])

        if chapters:
            lines.extend(["", "【前文概要/最近章节】"])
            for chapter in chapters[-3:]:
                title = chapter.get("title", "未命名章节")
                content = chapter.get("content", "")
                lines.append(f"\n{title}")
                lines.append(self._clip(content, 1800, keep_tail=True))
        else:
            lines.extend(["", "【前文概要】", "尚未生成正文，这是第一章。"])

        return self._clip("\n".join(lines), 11000, keep_tail=True)

    def _mock_search_info(self, query: str) -> str:
        """搜索技能占位实现；以后可在这里接 SerpAPI、Bing Search 或自建 RAG。"""
        query = query or "未指定主题"
        time.sleep(0.2)
        return (
            f"【Mock 联网资料】主题：{query}\n"
            "1. 请把该资料视为写作参考，而非权威事实来源。\n"
            "2. 可用于补充时代氛围、专业术语、生活细节和冲突素材。\n"
            "3. 若要接入真实搜索 API，可在 NovelAgent._mock_search_info 中替换为 httpx 请求。"
        )

    def _raise_if_cancelled(self) -> None:
        if self._cancel_requested:
            raise RuntimeError("用户已中止生成。")

    def _raise_for_status(self, response: httpx.Response) -> None:
        if response.is_success:
            return
        body = response.text[:800]
        raise RuntimeError(f"API 请求失败：HTTP {response.status_code}\n{body}")

    def _endpoint(self, base_url: str, suffix: str, known_tail: str) -> str:
        base = (base_url or "").strip().rstrip("/")
        if not base:
            base = "https://api.deepseek.com/v1"
        if base.endswith(known_tail):
            return base
        if suffix.startswith("/v1/") and base.endswith("/v1"):
            return base + suffix[len("/v1") :]
        return base + suffix

    def _max_tokens(self, max_words: int) -> int:
        # 中文“字数”和 token 不等价，这里留足空间，同时避免部分网关拒绝过大值。
        return max(1200, min(16000, int(max_words * 2.4)))

    def _loads_json(self, text: str) -> dict[str, Any]:
        try:
            data = json.loads(text or "{}")
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            return {}

    def _extract_text(self, content: Any) -> str:
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    parts.append(str(item.get("text") or item.get("content") or ""))
                else:
                    parts.append(str(item))
            return "\n".join(part for part in parts if part).strip()
        return str(content).strip()

    def _to_anthropic_messages(self, messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
        system_parts: list[str] = []
        converted: list[dict[str, Any]] = []

        for message in messages:
            role = message.get("role")
            content = message.get("content", "")
            if role == "system":
                system_parts.append(str(content))
            elif role in {"user", "assistant"}:
                converted.append({"role": role, "content": str(content)})

        return "\n\n".join(system_parts), converted

    def _clip(self, text: str, limit: int, keep_tail: bool = False) -> str:
        if len(text) <= limit:
            return text
        if keep_tail:
            return "……（前文已截断）\n" + text[-limit:]
        return text[:limit] + "\n……（后文已截断）"

    def _log(self, callback: LogCallback | None, message: str) -> None:
        if callback:
            callback(message)
