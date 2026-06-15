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
    source_dir = Path(__file__).resolve().parent
    # 开发模式下，如果 dist/ 目录存在，优先使用（与打包后的路径保持一致）
    dist_dir = source_dir / "dist"
    if dist_dir.is_dir():
        return dist_dir
    return source_dir


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


# ---------------------------------------------------------------------------
# Cross-lingual keyword mapping for Chinese → English knowledge base search
# Each Chinese key maps to a list of English terms/phrases used in knowledge/*.md
# ---------------------------------------------------------------------------
_CROSSLINGUAL_MAP: dict[str, list[str]] = {
    # ── 修炼 / 境界 / 功法 ──────────────────────────────────────────
    "修炼": ["cultivation", "practice", "cultivator", "training", "advance"],
    "境界": ["cultivation realm", "stage", "breakthrough tier", "level"],
    "功法": ["technique", "method", "manual", "art", "path", "school"],
    "灵力": ["qi", "spirit energy", "spiritual power", "energy source"],
    "灵气": ["qi", "clear qi", "turbid qi", "spirit qi", "energy"],
    "突破": ["breakthrough", "advancement", "ascend", "realm crossing"],
    "瓶颈": ["bottleneck", "plateau", "wall", "block"],
    "金丹": ["golden core", "golden pill"],
    "元婴": ["nascent soul", "yuan ying", "origin spirit"],
    "化神": ["transformation", "spirit transformation", "deity"],
    "筑基": ["foundation building", "foundation establishment"],
    "炼气": ["qi refinement", "breath cultivation"],
    "渡劫": ["tribulation", "heavenly trial", "catastrophe"],
    "大乘": ["great vehicle", "mahayana"],
    "飞升": ["ascension", "ascend", "rise to upper realm"],
    "体系": ["system", "framework", "structure", "scaffolding"],
    "赋能": ["intake", "energy source", "power origin"],
    "承载": ["bearing", "load", "capacity", "tolerance"],

    # ── 世界 / 层次 / 空间 ──────────────────────────────────────────
    "世界": ["world", "realm", "layer", "dimension", "sphere", "universe"],
    "世界框架": ["world axes", "world structure", "cosmology", "pillars of world"],
    "世界观": ["worldbuilding", "setting", "cosmology", "world logic"],
    "层次": ["layer", "tier", "level", "stratification", "fold"],
    "分层": ["layered", "stacked", "nested", "seven layers", "folded"],
    "上界": ["upper realm", "immortal court", "celestial", "higher layer"],
    "下界": ["lower realm", "mortal world", "bottom layer", "hongchenyuan"],
    "中间层": ["middle layer", "intermediate realm", "between layers"],
    "空间": ["space", "dimension", "realm boundary", "sphere"],
    "维度": ["dimension", "fold", "plane"],
    "壁垒": ["barrier", "wall", "seal", "blockade"],
    "通道": ["route", "passage", "gate", "path", "travel route"],

    # ── 势力 / 宗门 / 家族 ──────────────────────────────────────────
    "势力": ["faction", "power", "force", "bloc", "camp"],
    "宗门": ["sect", "clan", "school", "order", "house"],
    "家族": ["clan", "family", "house", "great family", "lineage"],
    "门派": ["sect", "school", "faction", "martial order"],
    "朝廷": ["court", "dynasty", "imperial", "official power"],
    "世家": ["great clan", "noble house", "aristocratic house", "great family"],
    "六院": ["six houses", "six-house court", "six clans"],
    "仙廷": ["immortal court", "hidden court", "celestial court", "upper court"],
    "官方": ["official power", "court", "dynasty", "empire"],
    "行会": ["guild", "trade", "alliance", "workshop"],
    "祭祀": ["ritual", "folk power", "temple", "sacrifice"],

    # ── 人物 / 关系 / 主角 ──────────────────────────────────────────
    "人物": ["character", "cast", "person", "figure"],
    "角色": ["character", "role", "cast member"],
    "关系": ["relationship", "arc", "tension", "bond", "conflict"],
    "主角": ["protagonist", "hero", "main character", "lead"],
    "反派": ["antagonist", "villain", "enemy", "opponent", "rival"],
    "配角": ["supporting character", "side character", "ensemble"],
    "人物弧光": ["character arc", "arc", "transformation", "growth"],
    "身份": ["identity", "status", "origin", "register"],
    "来历": ["origin", "background", "birth", "lineage"],

    # ── 金手指 / 机缘 / 主角优势 ────────────────────────────────────
    "金手指": ["golden finger", "cheat", "hidden inheritance", "protagonist advantage"],
    "机缘": ["opportunity", "fortune", "chance", "encounter", "inheritance"],
    "奇遇": ["encounter", "fortune", "lucky chance", "discovery"],
    "传承": ["inheritance", "legacy", "transmission", "heritage"],
    "古宝": ["ancient artifact", "artifact", "relic", "treasure"],
    "残魂": ["remnant soul", "residual spirit", "fragment soul"],
    "系统": ["system", "interface", "mechanism"],
    "骨片": ["bone shard", "bone fragment", "skull shard"],
    "创始人": ["founder", "murdered founder", "ancestor", "originator"],

    # ── 剧情 / 结构 / 节奏 ──────────────────────────────────────────
    "分卷": ["volume", "arc", "book", "series structure"],
    "结构": ["structure", "framework", "architecture", "scaffold"],
    "大纲": ["outline", "structure", "plan", "blueprint"],
    "节奏": ["pacing", "rhythm", "tempo", "progression"],
    "高潮": ["climax", "peak", "revelation", "payoff"],
    "反转": ["twist", "reversal", "turn", "subversion"],
    "伏笔": ["foreshadowing", "setup", "clue", "hint", "payoff"],
    "呼应": ["payoff", "callback", "resolution", "reclassify"],
    "钩子": ["hook", "cliffhanger", "tension", "pull"],
    "揭露": ["revelation", "reveal", "unveiling", "expose"],
    "谜底": ["truth", "secret", "mystery", "hidden truth"],
    "线索": ["clue", "hint", "trail", "thread", "fragment"],

    # ── 命名 / 称号 / 语言 ──────────────────────────────────────────
    "命名": ["naming", "name", "title", "term", "register"],
    "称号": ["title", "epithet", "honorific", "rank name"],
    "语言": ["language", "register", "naming system", "terminology"],
    "标签": ["label", "tag", "category", "classifier"],

    # ── 场景 / 氛围 / 感官 ──────────────────────────────────────────
    "场景": ["scene", "location", "setting", "place", "realm description"],
    "氛围": ["atmosphere", "flavor", "ambiance", "mood", "texture"],
    "感官": ["sensory", "smell", "sound", "color", "texture", "light"],
    "描写": ["description", "depiction", "rendering", "language"],
    "环境": ["environment", "surroundings", "ecology", "terrain"],

    # ── 万兽 / 妖族 / 兽域 ──────────────────────────────────────────
    "妖兽": ["beast", "monster", "creature", "spirit beast"],
    "兽域": ["beast realm", "beast cavern", "wild realm"],
    "血脉": ["bloodline", "ancestry", "lineage", "blood", "inheritance"],
    "妖族": ["beast", "creature", "non-human", "monster civilization"],
    "万兽": ["ten thousand beast", "beast cavern", "beast realm"],

    # ── 冥界 / 鬼域 / 死亡 ──────────────────────────────────────────
    "冥界": ["underworld", "ghost realm", "blood realm", "afterlife"],
    "鬼域": ["ghost realm", "underworld", "spirit world", "dead realm"],
    "灵魂": ["soul", "spirit", "ghost", "memory", "remnant"],
    "死亡": ["death", "dead", "underworld", "grave", "corpse"],
    "轮回": ["reincarnation", "cycle", "rebirth", "return"],
    "转世": ["reincarnation", "rebirth", "new life"],
    "记忆": ["memory", "recollection", "remembrance", "trace"],

    # ── 外来者 / 侵蚀 / 外域 ──────────────────────────────────────
    "外来者": ["outsider", "invader", "foreign", "cosmic outsider"],
    "侵蚀": ["corruption", "erosion", "decay", "rot", "contamination"],
    "外域": ["outside realm", "outer domain", "beyond", "void"],
    "入侵": ["invasion", "incursion", "intrusion", "breach"],
    "污染": ["corruption", "contamination", "taint", "blight"],

    # ── 实例 / 案例 ────────────────────────────────────────────────
    "实例": ["worked example", "example", "case", "demonstration"],
    "案例": ["worked example", "example", "case study", "template"],
    "红尘院": ["hongchenyuan", "red dust courtyard"],
    "七重": ["sevenfold", "seven layers", "seven fold"],
    "星图": ["star atlas", "star chart", "fracture star"],
    "碎骨": ["bone shard", "bone fragment"],

    # ── 通用 / 杂项 ────────────────────────────────────────────────
    "设定": ["setting", "worldbuilding", "configuration", "definition"],
    "规则": ["rule", "law", "principle", "axiom", "logic"],
    "设计": ["design", "build", "construct", "architect"],
    "一致性": ["consistency", "coherence", "contradiction", "checklist"],
    "矛盾": ["contradiction", "conflict", "tension", "inconsistency"],
    "代价": ["cost", "price", "burden", "sacrifice", "consequence"],
    "压迫": ["exploitation", "oppression", "extraction", "burden"],
    "秩序": ["order", "orthodoxy", "legitimacy", "system"],
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
    """Extract meaningful Chinese/English search terms using jieba segmentation.

    After jieba tokenization, Chinese terms are expanded via _CROSSLINGUAL_MAP
    to include their English equivalents, enabling cross-language retrieval from
    English-language knowledge base documents.
    """
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

    # Expand Chinese terms with their English equivalents from the cross-lingual map.
    # Also try combining adjacent Chinese terms as a compound key for multi-word mappings.
    chinese_only = [t for t in terms if any("一" <= ch <= "鿿" for ch in t)]
    for term in chinese_only:
        if term in _CROSSLINGUAL_MAP:
            terms.extend(_CROSSLINGUAL_MAP[term])

    # Also try compound keys: e.g. jieba splits "世界框架" → "世界" + "框架";
    # try the original compound from the raw query to catch "世界框架" as a whole.
    for compound_key, english_terms in _CROSSLINGUAL_MAP.items():
        if len(compound_key) >= 3 and compound_key in query:
            terms.extend(english_terms)

    # Deduplicate while preserving order (case-insensitive)
    seen: set[str] = set()
    result: list[str] = []
    for term in terms:
        key = term.lower()
        if key not in seen:
            seen.add(key)
            result.append(term)
    return result


def _tokenize_for_tfidf(text: str) -> dict[str, float]:
    """Tokenize text and return term frequency dict (raw counts).

    After standard jieba segmentation, also scans the text for multi-word English
    phrases from _CROSSLINGUAL_MAP so that queries expanded with compound terms
    (e.g. "golden finger") can match document contents directly.
    """
    words = jieba.cut_for_search(text)
    tf: dict[str, float] = {}
    for word in words:
        word = word.strip().lower()
        if not word or word in _STOP_WORDS:
            continue
        if len(word) < 2 and not word.isascii():
            continue
        tf[word] = tf.get(word, 0.0) + 1.0

    # Preserve multi-word English phrases so cross-lingual compound queries
    # (e.g. "golden finger", "world axes", "bone shard") match the document
    text_lower = text.lower()
    for english_terms in _CROSSLINGUAL_MAP.values():
        for phrase in english_terms:
            if " " not in phrase:
                continue  # single-word terms already handled by jieba
            count = text_lower.count(phrase)
            if count > 0:
                tf[phrase] = tf.get(phrase, 0.0) + float(count)

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
    """本地设定集语义检索：优先使用 Embedding 语义搜索，降级到 TF-IDF。

    === 搜索策略 ===

    第一方案（首选）—— Embedding 语义搜索：
      - 使用 BGE-small-zh-v1.5 模型把查询文本转成向量
      - 在 ChromaDB 向量库中搜索最相似的文档块
      - 优势：理解语义，支持中英跨语言检索
      - 依赖：sentence-transformers 库

    第二方案（降级）—— TF-IDF 关键词搜索：
      - 使用 jieba 分词 + TF-IDF 余弦相似度
      - 优势：无额外依赖，始终可用
      - 局限：只能匹配同一种语言的关键词

    选择逻辑：
      - 如果知识库中有向量索引数据 → 直接用语义搜索
      - 如果 sentence-transformers 没装 → 自动降级到 TF-IDF
      - 如果语义搜索出错 → 自动降级到 TF-IDF（但会提示错误原因）
    """
    ensure_app_dirs()
    query = query.strip()
    if not query:
        return "请先输入要检索的关键词，例如：境界划分、主角能力、唐朝夜市。"

    # ---- 尝试语义搜索 ----
    # 用 try/except 保护：就算新代码出了问题，也不影响现有功能
    try:
        from knowledge_base import EmbeddingKnowledgeBase, get_knowledge_base

        kb = get_knowledge_base()

        # check_and_refresh: 如果知识文件有变动，自动重建索引
        # 这样修改 knowledge.txt 后不需要手动点"重建索引"按钮
        chunk_count = kb.check_and_refresh()

        if chunk_count > 0:
            # 语义搜索可用的标志
            return kb.search(query, top_k=5)

    except ImportError:
        # sentence-transformers 没装 → 静默降级
        pass
    except Exception as exc:
        # 其他错误 → 在结果中提示，然后降级
        fallback_header = (
            f"[注意] 语义搜索遇到问题，已降级到关键词搜索。\n"
            f"原因：{exc}\n"
            f'可尝试：重新导入知识库文件，或在设置中点击“重建索引”。\n\n'
        )
        return fallback_header + _search_local_knowledge_tfidf(query)

    # ---- 降级：使用原来的 TF-IDF 搜索 ----
    return _search_local_knowledge_tfidf(query)


def _search_local_knowledge_tfidf(query: str) -> str:
    """原始的 TF-IDF 关键词搜索（作为降级方案保留）。"""
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
