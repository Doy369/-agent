from __future__ import annotations

import math
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Iterable

import jieba


def get_app_dir() -> Path:
    """返回可写的应用目录；打包为 exe 后使用 exe 所在目录。"""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def get_resource_path(name: str) -> Path:
    """读取 PyInstaller onefile 解包目录中的资源，开发环境则读取源码目录。"""
    bundle_dir = getattr(sys, "_MEIPASS", None)
    if bundle_dir:
        bundled_path = Path(bundle_dir) / name
        if bundled_path.exists():
            return bundled_path
    return Path(__file__).resolve().parent / name


APP_DIR = get_app_dir()
HISTORY_DIR = APP_DIR / "history"
KNOWLEDGE_FILE = APP_DIR / "knowledge.txt"
KNOWLEDGE_DIR = APP_DIR / "knowledge"
IMAGE_DIR = APP_DIR / "generated_images"
SKILL_DIR = APP_DIR / "local_skills"


# Common Chinese stop words that carry little semantic meaning for search
_STOP_WORDS: set[str] = {
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一",
    "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着",
    "没有", "看", "好", "自己", "这", "他", "她", "它", "们", "那", "些",
    "什么", "怎么", "如何", "哪", "吗", "呢", "吧", "啊", "哦", "嗯",
    "可以", "这个", "那个", "哪个", "为什么", "因为", "所以", "但是",
    "如果", "虽然", "而且", "或者", "还是", "已经", "正在", "将", "把",
    "被", "让", "从", "对", "向", "与", "以", "及", "等", "其", "或",
    "该", "则", "而", "且", "但", "只", "可", "能", "会", "应", "需",
    "请", "如", "按", "照", "根据", "关于", "通过", "经过", "为了",
    "出来", "起来", "过来", "过去", "下来", "下去", "进行", "使用",
}


DEFAULT_KNOWLEDGE = """# 示例设定集

【境界划分】
炼气、筑基、金丹、元婴、化神。每个大境界分为初期、中期、后期、圆满。

【主角能力】
主角林照夜拥有"星砂推演"能力，可通过星光痕迹还原过去三日内发生的关键事件。

【世界背景】
故事发生在云衡大陆。灵气潮汐每三十年涨落一次，潮汐最低时古老遗迹会露出入口。
"""


def ensure_app_dirs() -> None:
    """创建运行所需目录，并在首次运行时准备一个示例 knowledge.txt。"""
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    SKILL_DIR.mkdir(parents=True, exist_ok=True)

    if not KNOWLEDGE_FILE.exists():
        bundled = get_resource_path("knowledge.txt")
        if bundled.exists() and bundled != KNOWLEDGE_FILE:
            KNOWLEDGE_FILE.write_text(bundled.read_text(encoding="utf-8"), encoding="utf-8")
        else:
            KNOWLEDGE_FILE.write_text(DEFAULT_KNOWLEDGE, encoding="utf-8")


def safe_filename(name: str, fallback: str = "未命名章节") -> str:
    """将章节标题转换为 Windows 可用文件名。"""
    cleaned = re.sub(r'[\\/:*?"<>|\r\n]+', "_", name).strip(" .")
    return cleaned[:80] or fallback


def _read_knowledge_sources() -> list[tuple[str, str]]:
    ensure_app_dirs()
    sources: list[tuple[str, str]] = []

    if KNOWLEDGE_FILE.exists():
        sources.append((KNOWLEDGE_FILE.name, KNOWLEDGE_FILE.read_text(encoding="utf-8", errors="ignore")))

    knowledge_paths = list(KNOWLEDGE_DIR.glob("*.txt")) + list(KNOWLEDGE_DIR.glob("*.md"))
    for path in sorted(knowledge_paths):
        sources.append((path.name, path.read_text(encoding="utf-8", errors="ignore")))

    return sources


def _unique_destination(directory: Path, filename: str) -> Path:
    """Return a non-conflicting destination path inside directory."""
    target = directory / safe_filename(Path(filename).stem, "imported")
    suffix = Path(filename).suffix.lower() or ".txt"
    candidate = target.with_suffix(suffix)
    index = 2
    while candidate.exists():
        candidate = directory / f"{target.stem}_{index}{suffix}"
        index += 1
    return candidate


def import_knowledge_files(paths: list[str]) -> list[Path]:
    """Copy user-selected knowledge files into knowledge/ for later retrieval."""
    ensure_app_dirs()
    imported: list[Path] = []
    allowed_suffixes = {".txt", ".md"}

    for raw_path in paths:
        source = Path(raw_path)
        if not source.exists() or not source.is_file():
            continue
        if source.suffix.lower() not in allowed_suffixes:
            continue
        destination = _unique_destination(KNOWLEDGE_DIR, source.name)
        shutil.copy2(source, destination)
        imported.append(destination)

    return imported


def _split_sections(text: str) -> list[str]:
    """Split knowledge text on blank lines, markdown headers, 【】 blocks, and --- separators."""
    parts = re.split(
        r"\n\s*\n|(?=【[^】]{1,50}】)|(?=^#{1,3}\s+)|(?=^---\s*$)",
        text,
        flags=re.M,
    )
    return [
        part.strip()
        for part in parts
        if part.strip() and not re.match(r"^---\s*$", part.strip())
    ]


def _query_terms(query: str) -> list[str]:
    """Extract meaningful Chinese/English search terms using jieba segmentation."""
    query = query.strip()
    if not query:
        return []

    # Use jieba search mode for better recall (splits compounds into sub-words)
    words = jieba.cut_for_search(query)
    terms: list[str] = []
    for word in words:
        word = word.strip()
        if not word:
            continue
        if word in _STOP_WORDS:
            continue
        if len(word) >= 2 or word.isascii():
            terms.append(word)
        elif len(word) == 1 and "一" <= word <= "鿿":
            # Single Chinese char that survived stop-word filter (e.g., rare chars in names)
            terms.append(word)

    # Deduplicate while preserving order (case-insensitive)
    seen: set[str] = set()
    result: list[str] = []
    for term in terms:
        if term.lower() not in seen:
            seen.add(term.lower())
            result.append(term)
    return result


def _tokenize_for_tfidf(text: str) -> dict[str, float]:
    """Tokenize text and return term frequency dict (raw counts)."""
    words = jieba.cut_for_search(text)
    tf: dict[str, float] = {}
    for word in words:
        word = word.strip().lower()
        if not word or word in _STOP_WORDS:
            continue
        if len(word) < 2 and not word.isascii():
            continue
        tf[word] = tf.get(word, 0.0) + 1.0
    return tf


def _cosine_similarity(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
    """Compute cosine similarity between two sparse vectors represented as dicts."""
    if not vec_a or not vec_b:
        return 0.0

    dot_product = sum(vec_a.get(k, 0.0) * vec_b.get(k, 0.0) for k in set(vec_a) | set(vec_b))
    mag_a = math.sqrt(sum(v * v for v in vec_a.values()))
    mag_b = math.sqrt(sum(v * v for v in vec_b.values()))

    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot_product / (mag_a * mag_b)


def search_local_knowledge(query: str) -> str:
    """本地设定集语义检索：使用 jieba 分词 + TF-IDF 余弦相似度进行搜索。"""
    ensure_app_dirs()
    query = query.strip()
    if not query:
        return "请先输入要检索的关键词，例如：境界划分、主角能力、唐朝夜市。"

    query_terms = _query_terms(query)
    if not query_terms:
        return f"无法从查询中提取有效搜索词：{query}"

    # Build corpus: collect all sections from all knowledge sources
    sources = _read_knowledge_sources()
    sections: list[tuple[str, str, str]] = []  # (source_name, section_text, section_text_lower)
    for source_name, text in sources:
        for section in _split_sections(text):
            if section.strip():
                sections.append((source_name, section, section.lower()))

    if not sections:
        return f"知识库为空。你可以编辑 {KNOWLEDGE_FILE} 或导入知识库文件添加设定。"
    # ... rest unchanged


    # Pre-compute TF vectors for all sections
    section_vectors: list[dict[str, float]] = [_tokenize_for_tfidf(s[2]) for s in sections]

    # Build IDF from corpus
    N = len(sections)
    df: dict[str, int] = {}  # document frequency
    for vec in section_vectors:
        for term in vec:
            df[term] = df.get(term, 0) + 1

    # Compute TF-IDF vectors for sections
    section_tfidf: list[dict[str, float]] = []
    for vec in section_vectors:
        tfidf_vec: dict[str, float] = {}
        for term, tf in vec.items():
            idf = math.log((N + 1) / (df.get(term, 1) + 1)) + 1.0
            tfidf_vec[term] = tf * idf
        section_tfidf.append(tfidf_vec)

    # Build query TF-IDF vector
    query_tf: dict[str, float] = {}
    for term in query_terms:
        query_tf[term.lower()] = query_tf.get(term.lower(), 0.0) + 1.0
    query_tfidf: dict[str, float] = {}
    for term, tf in query_tf.items():
        idf = math.log((N + 1) / (df.get(term, 1) + 1)) + 1.0
        query_tfidf[term] = tf * idf

    # Score sections by cosine similarity
    scored: list[tuple[float, str, str]] = []
    query_str_lower = query.lower()
    for i, (source_name, section, _) in enumerate(sections):
        sim = _cosine_similarity(query_tfidf, section_tfidf[i])
        # Bonus for exact query substring match (helps with proper names)
        if query_str_lower in section.lower():
            sim += 0.15
        if sim > 0.0:
            scored.append((sim, source_name, section))

    if not scored:
        # Fallback: try individual query terms against sections as substring
        for term in query_terms:
            for source_name, section, _ in sections:
                if term.lower() in section.lower():
                    scored.append((0.05, source_name, section))
        if not scored:
            return (
                f'未在本地设定集中找到与"{query}"语义相关的内容。\n'
                f'提取的搜索词：{", ".join(query_terms)}\n'
                f'你可以编辑 {KNOWLEDGE_FILE} 添加设定，或导入更多知识库文件。'
            )

    scored.sort(key=lambda item: item[0], reverse=True)
    lines = [f'本地设定语义检索结果：{query}', f'搜索词：{", ".join(query_terms)}', ""]
    for index, (score, source_name, section) in enumerate(scored[:5], start=1):
        lines.append(f'{index}. 来源：{source_name}（相关度 {score:.2f}）')
        lines.append(section)
        lines.append("")
    return "\n".join(lines).strip()


def _find_all(pattern: str, text: str) -> Iterable[re.Match[str]]:
    return re.finditer(pattern, text, flags=re.M)


def local_text_audit(text: str) -> str:
    """本地敏感词与错别字检测：用轻量规则给出可人工确认的修改建议。"""
    text = text or ""
    if not text.strip():
        return "当前没有可检查的文本。"

    suggestions: list[str] = []

    typo_map = {
        "在次": "再次",
        "以经": "已经",
        "因该": "应该",
        "必竞": "毕竟",
        "既使": "即使",
        "帐号": "账号",
        "登陆": "登录（如果指进入系统）",
        "其它": "其他（正式正文中更常用）",
    }
    for wrong, right in typo_map.items():
        for match in _find_all(re.escape(wrong), text):
            suggestions.append(f'疑似错别字：第 {match.start() + 1} 字附近，"{wrong}"可考虑改为"{right}"。')

    punctuation_rules = [
        (r"[，。！？、；：]{2,}", "连续中文标点过多，建议压缩为一个标点。"),
        (r"\.{4,}", '英文省略号过长，中文正文建议使用"......"。'),
        (r"[!?]{2,}", "连续英文感叹/问号较多，建议统一为中文标点。"),
        (r"\s+[，。！？；：、]", "标点前存在多余空格。"),
    ]
    for pattern, message in punctuation_rules:
        for match in _find_all(pattern, text):
            snippet = text[max(0, match.start() - 10) : match.end() + 10].replace("\n", " ")
            suggestions.append(f'标点/格式：第 {match.start() + 1} 字附近，{message} 片段：{snippet}')

    sensitive_words = {
        "身份证号": "可能涉及真实个人信息，请确认是否为虚构内容。",
        "银行卡号": "可能涉及真实个人信息，请确认是否为虚构内容。",
        "自杀方法": "涉及高风险细节，建议改为心理状态描写或求助情节。",
        "未成年露骨": "涉及敏感描写，建议删除或改为非露骨表达。",
        "血腥细节": "如描写过度，建议弱化为氛围或结果描写。",
    }
    for word, advice in sensitive_words.items():
        for match in _find_all(re.escape(word), text):
            suggestions.append(f'敏感词提示：第 {match.start() + 1} 字附近出现"{word}"。{advice}')

    long_sentence_pattern = r"[^。！？\n]{160,}[。！？]"
    for match in _find_all(long_sentence_pattern, text):
        suggestions.append(f'可读性提示：第 {match.start() + 1} 字附近句子较长，建议拆分以增强节奏。')

    if not suggestions:
        return "本地检查完成：未发现明显错别字、异常标点或预设敏感词。"

    lines = ["本地错字/敏感词检查报告", ""]
    lines.extend(f"- {item}" for item in suggestions[:80])
    if len(suggestions) > 80:
        lines.append(f"- 其余 {len(suggestions) - 80} 条已省略，建议分段检查。")
    return "\n".join(lines)


def archive_chapter(title: str, content: str, suffix: str = ".md") -> Path:
    """自动备份章节到 history/，防止界面关闭或误操作导致丢稿。"""
    ensure_app_dirs()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    extension = suffix if suffix in {".md", ".txt"} else ".md"
    filename = f"{timestamp}_{safe_filename(title)}{extension}"
    path = HISTORY_DIR / filename

    if extension == ".md":
        path.write_text(f"# {title}\n\n{content.strip()}\n", encoding="utf-8")
    else:
        path.write_text(content.strip() + "\n", encoding="utf-8")
    return path


def _count_cjk_and_words(text: str) -> int:
    """粗略统计正文长度：中文按字计，英文按词计。"""
    cjk_count = len(re.findall(r"[\u4e00-\u9fff]", text))
    latin_words = len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", text))
    return cjk_count + latin_words


def local_archive_and_stats() -> str:
    """遍历 history/ 目录，统计章节数和总字数，并生成进度报表。"""
    ensure_app_dirs()
    chapter_files = [
        path
        for path in sorted(HISTORY_DIR.glob("*"))
        if path.suffix.lower() in {".txt", ".md"} and path.name != "书籍大纲进度报表.txt"
    ]

    total_words = 0
    report_lines = [
        "书籍大纲进度报表",
        f"生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"历史目录：{HISTORY_DIR}",
        "",
        "章节明细：",
    ]

    for index, path in enumerate(chapter_files, start=1):
        text = path.read_text(encoding="utf-8", errors="ignore")
        count = _count_cjk_and_words(text)
        total_words += count
        title = path.stem
        first_line = next((line.strip("# ").strip() for line in text.splitlines() if line.strip()), "")
        if first_line:
            title = first_line[:60]
        report_lines.append(f"{index:03d}. {title} | 约 {count} 字 | {path.name}")

    report_lines.extend(
        [
            "",
            f"总章节数：{len(chapter_files)}",
            f"总字数：约 {total_words} 字",
            "",
            "建议：如果章节重复或包含大纲备份，可手动整理 history/ 后重新生成统计。",
        ]
    )

    report = "\n".join(report_lines)
    report_path = HISTORY_DIR / "书籍大纲进度报表.txt"
    report_path.write_text(report, encoding="utf-8")
    return report + f"\n\n报表已写入：{report_path}"


def generate_image(prompt: str) -> str:
    """AI 插图技能 Mock：生成一张本地 PNG，占位模拟 Stable Diffusion / DALL-E 返回图。"""
    ensure_app_dirs()
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise RuntimeError("缺少 Pillow，请先执行：pip install -r requirements.txt") from exc

    prompt = (prompt or "本章精彩场景").strip()
    width, height = 1024, 768
    image = Image.new("RGB", (width, height), "#111827")
    draw = ImageDraw.Draw(image)

    # 绘制简单渐变背景，让 Mock 插图在弹窗里有明确视觉反馈。
    for y in range(height):
        ratio = y / max(1, height - 1)
        r = int(17 + 26 * ratio)
        g = int(24 + 70 * ratio)
        b = int(39 + 95 * ratio)
        draw.line([(0, y), (width, y)], fill=(r, g, b))

    for radius, color in [(360, "#2563eb"), (250, "#f59e0b"), (160, "#10b981")]:
        x = int(width * (0.25 + radius / 1600))
        y = int(height * (0.22 + radius / 2200))
        draw.ellipse((x, y, x + radius, y + radius), outline=color, width=4)

    font_paths = [
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
    ]
    title_font = None
    body_font = None
    for font_path in font_paths:
        if font_path.exists():
            title_font = ImageFont.truetype(str(font_path), 44)
            body_font = ImageFont.truetype(str(font_path), 24)
            break
    title_font = title_font or ImageFont.load_default()
    body_font = body_font or ImageFont.load_default()

    draw.rounded_rectangle((70, 500, 954, 700), radius=24, fill=(15, 23, 42), outline="#94a3b8", width=2)
    draw.text((96, 530), "小说 AI 插图 Mock", font=title_font, fill="#f8fafc")

    wrapped = []
    line = ""
    for char in prompt[:120]:
        line += char
        if len(line) >= 28:
            wrapped.append(line)
            line = ""
    if line:
        wrapped.append(line)
    draw.text((100, 595), "\n".join(wrapped[:3]), font=body_font, fill="#dbeafe", spacing=8)

    filename = f"{time.strftime('%Y%m%d_%H%M%S')}_{safe_filename(prompt[:28], 'chapter_image')}.png"
    path = IMAGE_DIR / filename
    image.save(path)
    return str(path)
