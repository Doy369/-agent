"""
===============================================================================
知识库语义检索模块 -- 基于 Embedding + Numpy 向量存储
===============================================================================

=== 核心概念 ===

1. Embedding（文本嵌入 / 向量化）
   把一段文字变成一串数字（向量）。意思相近的文本，向量也相近。
   "主角突破金丹境界" -> [0.12, -0.34, 0.67, ..., 0.03]  (512 维)
   "golden core breakthrough" -> [0.11, -0.31, 0.65, ..., 0.05]
         这两个向量非常接近！因为语义相似。

2. 向量存储（Numpy 数组）
   把所有向量存在一个大矩阵里：(N 个文本块) × (512 维)。
   搜索时用矩阵乘法一次算出查询向量和所有文档向量的相似度，然后取 Top-K。

   对 ~1000 个块的规模，这是最快最可靠的方式，不依赖任何外部数据库。

3. RAG（检索增强生成）
   检索(Retrieval) + 增强(Augmentation) + 生成(Generation)
   流程：用户提问 -> 在知识库中检索相关内容 -> 拼入 Prompt -> LLM 生成回答

=== 和原来 TF-IDF 的区别 ===

           TF-IDF（旧）              Embedding（新）
   ------  ---------------------     ---------------------
   原理     数词频，词匹配              理解语义，意思匹配
   跨语言   需要人工维护映射表           模型自带跨语言理解
   "金丹"   只能找到含"金丹"的文本       也能找到 "golden core"
   "奇遇"   只能找到含"奇遇"的文本       也能找到 "机缘" "bone shard"
===============================================================================
"""

from __future__ import annotations

import hashlib
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from local_tools import (
    _read_knowledge_sources,
    _split_sections,
    ensure_app_dirs,
)

# ---------------------------------------------------------------------------
# Embedding 模型
# ---------------------------------------------------------------------------

EMBEDDING_MODEL_NAME = "BAAI/bge-small-zh-v1.5"

# BGE 模型查询前缀 -- 告诉模型"这是一条检索查询"，提升准确率 ~5-10%
BGE_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："

# ---------------------------------------------------------------------------
# 依赖检测
# ---------------------------------------------------------------------------

try:
    from sentence_transformers import SentenceTransformer

    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False
    SentenceTransformer = None

try:
    import numpy as np

    _NP_AVAILABLE = True
except ImportError:
    _NP_AVAILABLE = False
    np = None


# ===========================================================================
# EmbeddingKnowledgeBase -- 语义知识库
# ===========================================================================

class EmbeddingKnowledgeBase:
    """基于 Embedding + Numpy 的轻量语义知识库。

    为什么不用 ChromaDB？
    --------------------
    ChromaDB 在 Windows 上的 Rust HNSW 后端不够稳定，反复出现索引文件损坏。
    对于 1000-5000 个文本块的规模，用 numpy 矩阵乘法做暴力搜索反而更简单可靠：
      - 零外部依赖（numpy 已被 sentence-transformers 带入）
      - 存储透明（一个 .npz 文件保存所有向量和元数据）
      - 搜索快（矩阵乘法在底层用 BLAS 优化，~1100×512 不到 1ms）
    """

    def __init__(self, persist_dir: Optional[Path] = None):
        ensure_app_dirs()

        if persist_dir is None:
            from local_tools import APP_DIR
            persist_dir = APP_DIR / "vector_db"

        self._persist_dir = persist_dir
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        self._model = None  # SentenceTransformer，延迟加载

        # 内存中的索引数据
        self._embeddings: Optional[np.ndarray] = None  # (N, 512)
        self._documents: list[str] = []
        self._metadatas: list[dict] = []
        self._source_hash: str = ""

        # 启动时尝试加载已有索引
        self._load_from_disk()

    # ------------------------------------------------------------------
    # 属性
    # ------------------------------------------------------------------

    @property
    def st_available(self) -> bool:
        return _ST_AVAILABLE and _NP_AVAILABLE

    # ------------------------------------------------------------------
    # 模型加载
    # ------------------------------------------------------------------

    def _init_model(self):
        """加载 SentenceTransformer embedding 模型。

        首次调用从 HuggingFace 下载（~100MB），之后缓存到本地。
        国内慢可设：set HF_ENDPOINT=https://hf-mirror.com
        """
        if not _ST_AVAILABLE or not _NP_AVAILABLE:
            raise RuntimeError(
                "语义搜索功能需要安装 sentence-transformers 和 numpy。\n"
                "请在终端运行：pip install -r requirements.txt"
            )
        if self._model is None:
            self._model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        return self._model

    # ------------------------------------------------------------------
    # 磁盘持久化（替代 ChromaDB）
    # ------------------------------------------------------------------

    @property
    def _index_path(self) -> Path:
        return self._persist_dir / "vector_index.npz"

    @property
    def _meta_path(self) -> Path:
        return self._persist_dir / "vector_meta.pkl"

    def _save_to_disk(self) -> None:
        """把向量和元数据写入磁盘。

        两个文件：
          vector_index.npz  -- numpy 压缩格式，存向量矩阵
          vector_meta.pkl   -- pickle 格式，存文档文本、来源、哈希
        """
        if self._embeddings is None:
            return
        if not _NP_AVAILABLE:
            raise RuntimeError("保存向量索引需要 numpy。请先运行：pip install -r requirements.txt")

        np.savez_compressed(
            self._index_path,
            embeddings=self._embeddings,
        )

        with open(self._meta_path, "wb") as f:
            pickle.dump(
                {
                    "documents": self._documents,
                    "metadatas": self._metadatas,
                    "source_hash": self._source_hash,
                    "model": EMBEDDING_MODEL_NAME,
                    "built_at": datetime.now(timezone.utc).isoformat(),
                },
                f,
            )

    def _load_from_disk(self) -> bool:
        """从磁盘加载已有索引。

        Returns:
            True 如果成功加载，False 如果文件不存在或损坏。
        """
        if not self._index_path.exists() or not self._meta_path.exists():
            return False
        if not _NP_AVAILABLE:
            return False

        try:
            data = np.load(self._index_path)
            self._embeddings = data["embeddings"]

            with open(self._meta_path, "rb") as f:
                meta = pickle.load(f)

            self._documents = meta["documents"]
            self._metadatas = meta["metadatas"]
            self._source_hash = meta.get("source_hash", "")

            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # 构建 / 重建向量索引
    # ------------------------------------------------------------------

    def build_or_rebuild(self) -> int:
        """扫描 knowledge.txt 和 knowledge/*.md，全量重建向量索引。

        步骤：
          1. 读取所有知识文件
          2. 切分成段落块
          3. 用 SentenceTransformer 把每个块转成向量
          4. 把向量和文档一起存入 .npz + .pkl 文件

        Returns:
            成功存入的文本块数量。
        """
        sources = _read_knowledge_sources()
        if not sources:
            return 0

        # 切块
        chunks: list[dict] = []
        for source_name, text in sources:
            sections = _split_sections(text)
            for i, section in enumerate(sections):
                clean = section.strip()
                if clean and len(clean) > 20:
                    chunks.append({
                        "text": clean,
                        "source": source_name,
                        "chunk_index": i,
                    })

        if not chunks:
            return 0

        # 向量化
        model = self._init_model()
        texts = [c["text"] for c in chunks]
        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        # 存入内存
        self._embeddings = embeddings  # numpy 数组 (N, 512)
        self._documents = texts
        self._metadatas = [
            {"source": c["source"], "chunk_index": c["chunk_index"]}
            for c in chunks
        ]
        self._source_hash = self._compute_source_hash(sources)

        # 持久化到磁盘
        self._save_to_disk()

        return len(chunks)

    # ------------------------------------------------------------------
    # 语义搜索
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 5) -> str:
        """在知识库中执行语义搜索。

        原理：
          1. 把查询文本转成 512 维向量
          2. 用矩阵乘法一次算出查询向量和所有文档向量的余弦相似度
             ── 归一化后，余弦相似度 = 向量点积
                 score = query_vec @ doc_vecs.T   (1, 512) @ (512, N) = (1, N)
          3. 取出相似度最高的 K 个，格式化返回

        Args:
            query: 搜索内容。支持中文、英文、中英混合。
            top_k: 返回前 K 个最相似的结果。

        Returns:
            格式化的搜索报告文本，可直接拼入 LLM 上下文。
        """
        if not _ST_AVAILABLE or not _NP_AVAILABLE:
            return (
                "[语义搜索不可用] 需要安装 sentence-transformers 和 numpy。\n"
                "请在终端运行：pip install -r requirements.txt"
            )

        if self._embeddings is None or len(self._documents) == 0:
            count = self.build_or_rebuild()
            if count == 0:
                return "知识库为空，请先通过 GUI 导入设定文件或编辑 knowledge.txt。"

        model = self._init_model()

        # 查询向量化（BGE 需要加查询前缀）
        query_vec = model.encode(
            BGE_QUERY_INSTRUCTION + query,
            normalize_embeddings=True,
        )  # shape: (512,)

        # ---- 核心：矩阵乘法一次算出所有相似度 ----
        # self._embeddings: (N, 512)
        # query_vec: (512,)
        # scores: (N,) -- 每个文档和查询的余弦相似度
        scores = self._embeddings @ query_vec  # 归一化后点积 = 余弦相似度

        # 取 Top-K
        k = min(top_k, len(scores))
        top_indices = np.argpartition(scores, -k)[-k:]  # 部分排序，比 argsort 快
        top_indices = top_indices[np.argsort(scores[top_indices])][::-1]  # 降序

        # 格式化输出
        lines = [f"语义检索结果：{query}", ""]
        for rank, idx in enumerate(top_indices, start=1):
            sim = float(scores[idx])
            source = (self._metadatas[idx] or {}).get("source", "未知来源")
            lines.append(f"{rank}. 来源：{source}（相关度 {sim:.2f}）")
            lines.append(self._documents[idx].strip())
            lines.append("")

        return "\n".join(lines).strip()

    # ------------------------------------------------------------------
    # 状态检查
    # ------------------------------------------------------------------

    def is_ready(self) -> bool:
        """检查向量索引是否已加载。"""
        return self._embeddings is not None and len(self._documents) > 0

    def get_stats(self) -> dict:
        """返回知识库统计信息。"""
        return {
            "total_chunks": len(self._documents),
            "model": EMBEDDING_MODEL_NAME,
            "persist_dir": str(self._persist_dir),
            "metadata": {"source_hash": self._source_hash},
            "st_available": _ST_AVAILABLE and _NP_AVAILABLE,
        }

    def check_and_refresh(self) -> int:
        """检查知识文件是否有变化，如有变化则自动重建索引。

        每次搜索前调用，确保搜索结果反映最新的知识文件。
        """
        sources = _read_knowledge_sources()
        if not sources:
            return 0

        current_hash = self._compute_source_hash(sources)

        # 哈希不同 -> 文件有变动 -> 重建
        if current_hash != self._source_hash:
            return self.build_or_rebuild()

        return len(self._documents)

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    def _compute_source_hash(self, sources: list[tuple[str, str]]) -> str:
        h = hashlib.md5()
        for name, text in sorted(sources):
            h.update(name.encode("utf-8"))
            h.update(text.encode("utf-8"))
        return h.hexdigest()


# ===========================================================================
# 模块级单例
# ===========================================================================

_kb_instance: Optional[EmbeddingKnowledgeBase] = None


def get_knowledge_base() -> EmbeddingKnowledgeBase:
    """获取全局知识库单例。（模型只加载一次，省内存）"""
    global _kb_instance
    if _kb_instance is None:
        _kb_instance = EmbeddingKnowledgeBase()
    return _kb_instance


def rebuild_knowledge_base() -> int:
    """强制重建知识库索引（供 GUI 按钮调用）。"""
    kb = get_knowledge_base()
    return kb.build_or_rebuild()
