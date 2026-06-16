from __future__ import annotations

from typing import Any

from local_tools import (
    generate_image,
    local_archive_and_stats,
    local_text_audit,
    search_local_knowledge,
)
from prompt_templates import SYSTEM_PROMPT


def _load_langchain_core() -> tuple[Any, Any]:
    """Load LangChain only when this optional adapter is used."""
    try:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.tools import StructuredTool
    except ImportError as exc:
        raise RuntimeError(
            "LangChain 适配层需要 langchain-core。请先运行：pip install -r requirements.txt"
        ) from exc
    return ChatPromptTemplate, StructuredTool


def describe_langchain_integration() -> str:
    return (
        "LangChain 适配层提供两类能力：\n"
        "1. 把当前小说 Agent 的大纲、章节、润色 Prompt 封装为 ChatPromptTemplate。\n"
        "2. 把本地知识库检索、错字审稿、归档统计、Mock 插图封装为 LangChain Tool。\n"
        "它不会替换现有 PyQt6 主流程，主要用于后续接入 LangGraph、API 服务和自动化链路。"
    )


def build_prompt_templates() -> dict[str, Any]:
    """Return reusable LangChain ChatPromptTemplate objects."""
    ChatPromptTemplate, _ = _load_langchain_core()

    outline_user = f"""请基于用户设定，生成一份可连载执行的全书/分卷大纲。

小说类型：{{style}}

用户设定：
{{novel_setting}}

{{auto_context}}

输出要求：
1. 给出书名候选、核心卖点、主线矛盾、主要人物弧光。
2. 至少规划 3 卷，每卷包含目标、冲突升级、关键反转和结尾钩子。
3. 给出前 10 章章节标题和每章一句话剧情。
4. 保留后续扩展空间，不要把所有谜底一次性揭开。
"""

    chapter_user = """请生成第 {chapter_index} 章正文。

小说类型：{style}
目标篇幅：约 {max_words} 字

{memory_context}

{auto_context}

写作要求：
1. 章节标题放在第一行。
2. 承接前文，不要重启故事，不要重复大段设定说明。
3. 至少安排一个推进主线的行动、一个人物选择、一个结尾钩子。
4. 正文可直接发布，避免解释“我将如何写”。
"""

    polish_user = f"""请润色以下小说文本。

小说类型：{{style}}
润色目标：{{instruction}}

{{auto_context}}

原文：
{{selected_text}}

输出要求：
1. 只输出润色后的文本。
2. 不改变核心人物关系、剧情事实和叙事视角。
3. 可适度扩写动作、环境和心理，但不要灌水。
"""

    return {
        "outline": ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("user", outline_user)]),
        "chapter": ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("user", chapter_user)]),
        "polish": ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("user", polish_user)]),
    }


def _search_knowledge_tool(query: str) -> str:
    """本地设定集语义检索。查找世界观、境界体系、人物能力、势力组织等设定。"""
    return search_local_knowledge(query)


def _audit_text_tool(text: str) -> str:
    """本地敏感词与错别字检测。检查章节正文中的错别字、标点问题和敏感词。"""
    return local_text_audit(text)


def _archive_stats_tool() -> str:
    """本地章节归档统计。扫描 history 目录并生成全书进度报表。"""
    return local_archive_and_stats()


def _illustration_tool(prompt: str) -> str:
    """Mock 插图生成。根据章节高光场景描述生成本地 PNG 图片并返回路径。"""
    return generate_image(prompt)


def build_langchain_tools() -> list[Any]:
    """Return built-in local capabilities as LangChain StructuredTool objects."""
    _, StructuredTool = _load_langchain_core()

    return [
        StructuredTool.from_function(
            func=_search_knowledge_tool,
            name="search_local_knowledge",
            description=_search_knowledge_tool.__doc__ or "",
        ),
        StructuredTool.from_function(
            func=_audit_text_tool,
            name="local_text_audit",
            description=_audit_text_tool.__doc__ or "",
        ),
        StructuredTool.from_function(
            func=_archive_stats_tool,
            name="local_archive_and_stats",
            description=_archive_stats_tool.__doc__ or "",
        ),
        StructuredTool.from_function(
            func=_illustration_tool,
            name="generate_image",
            description=_illustration_tool.__doc__ or "",
        ),
    ]
