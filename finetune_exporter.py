from __future__ import annotations

import argparse
import json
from pathlib import Path

from local_tools import APP_DIR, HISTORY_DIR, ensure_app_dirs
from prompt_templates import SYSTEM_PROMPT


DEFAULT_OUTPUT_DIR = APP_DIR / "finetune_datasets"
DEFAULT_OUTPUT_FILE = DEFAULT_OUTPUT_DIR / "novel_agent_sft.jsonl"


def _history_files(source_dir: Path) -> list[Path]:
    if not source_dir.exists():
        return []
    return [
        path
        for path in sorted(source_dir.glob("*"))
        if path.is_file()
        and path.suffix.lower() in {".md", ".txt"}
        and path.name != "书籍大纲进度报表.txt"
    ]


def _extract_title(content: str, fallback: str) -> str:
    for line in content.splitlines():
        title = line.strip().lstrip("#").strip()
        if title:
            return title[:80]
    return fallback


def _build_example(content: str, title: str, style: str) -> dict[str, list[dict[str, str]]]:
    user_prompt = (
        f"请以「{style}」网文风格生成标题为「{title}」的一章正文。"
        "要求承接设定、节奏清晰、包含行动、选择和结尾钩子。"
    )
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": content.strip()},
        ]
    }


def export_openai_chat_jsonl(
    *,
    source_dir: Path | None = None,
    output_path: Path | None = None,
    min_chars: int = 200,
    limit: int | None = None,
    style: str = "中文类型小说",
) -> tuple[Path, int]:
    """Export archived chapters to OpenAI-style chat fine-tuning JSONL."""
    ensure_app_dirs()
    source_dir = source_dir or HISTORY_DIR
    output_path = output_path or DEFAULT_OUTPUT_FILE
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with output_path.open("w", encoding="utf-8") as file:
        for path in _history_files(source_dir):
            content = path.read_text(encoding="utf-8", errors="ignore").strip()
            if len(content) < min_chars:
                continue
            title = _extract_title(content, path.stem)
            example = _build_example(content, title, style)
            file.write(json.dumps(example, ensure_ascii=False) + "\n")
            count += 1
            if limit is not None and count >= limit:
                break

    return output_path, count


def main() -> None:
    parser = argparse.ArgumentParser(description="Export history chapters as chat fine-tuning JSONL.")
    parser.add_argument("--source-dir", type=Path, default=HISTORY_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--min-chars", type=int, default=200)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--style", default="中文类型小说")
    args = parser.parse_args()

    output_path, count = export_openai_chat_jsonl(
        source_dir=args.source_dir,
        output_path=args.output,
        min_chars=args.min_chars,
        limit=args.limit,
        style=args.style,
    )
    print(f"已导出 {count} 条微调样本：{output_path}")


if __name__ == "__main__":
    main()
