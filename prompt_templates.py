from __future__ import annotations

from dataclasses import dataclass


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

DEFAULT_NOVEL_SETTING = "用户暂未填写详细设定，请先给出一个可扩展的原创框架。"
DEFAULT_POLISH_INSTRUCTION = "增强画面感、节奏和情绪张力，同时保持原剧情事实不变。"


@dataclass(frozen=True)
class PromptPair:
    """A model-ready system/user prompt pair."""

    system: str
    user: str

    def to_messages(self) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.system},
            {"role": "user", "content": self.user},
        ]


def outline_prompt(
    *,
    novel_setting: str,
    style: str,
    auto_context: str,
) -> PromptPair:
    user_prompt = f"""请基于用户设定，生成一份可连载执行的全书/分卷大纲。

小说类型：{style}

用户设定：
{novel_setting or DEFAULT_NOVEL_SETTING}

{auto_context}

输出要求：
1. 给出书名候选、核心卖点、主线矛盾、主要人物弧光。
2. 至少规划 3 卷，每卷包含目标、冲突升级、关键反转和结尾钩子。
3. 给出前 10 章章节标题和每章一句话剧情。
4. 保留后续扩展空间，不要把所有谜底一次性揭开。
"""
    return PromptPair(system=SYSTEM_PROMPT, user=user_prompt)


def chapter_prompt(
    *,
    chapter_index: int,
    style: str,
    max_words: int,
    memory_context: str,
    auto_context: str,
) -> PromptPair:
    user_prompt = f"""请生成第 {chapter_index} 章正文。

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
    return PromptPair(system=SYSTEM_PROMPT, user=user_prompt)


def polish_prompt(
    *,
    selected_text: str,
    style: str,
    instruction: str,
    auto_context: str,
) -> PromptPair:
    user_prompt = f"""请润色以下小说文本。

小说类型：{style}
润色目标：{instruction or DEFAULT_POLISH_INSTRUCTION}

{auto_context}

原文：
{selected_text}

输出要求：
1. 只输出润色后的文本。
2. 不改变核心人物关系、剧情事实和叙事视角。
3. 可适度扩写动作、环境和心理，但不要灌水。
"""
    return PromptPair(system=SYSTEM_PROMPT, user=user_prompt)
