"""
================================================================================
RAG 深度教程 -- 逐步拆解你刚搭建的语义搜索系统
================================================================================

运行方式：
  在终端执行：python rag_tutorial.py

每执行一步会暂停，让你看清楚中间结果。
输入 "q" 可随时退出。

================================================================================
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


# ===========================================================================
# 第 1 步：理解「文本 -> 数字」(Embedding)
# ===========================================================================

def step1_what_is_embedding():
    """演示 embedding 如何把文字变成向量，语义相近的向量也相近。"""
    from sentence_transformers import SentenceTransformer

    print("=" * 65)
    print("第 1 步：什么是 Embedding（文本嵌入）？")
    print("=" * 65)
    print()
    print("一句话解释：把一段文字变成一串数字（向量）。")
    print("意思相近的文字，它们的数字也会很相近。")
    print()

    print("[加载模型 BAAI/bge-small-zh-v1.5 ...]")
    model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
    print("模型加载完成。")
    print()

    texts = [
        "主角林照夜拥有星砂推演能力，可通过星光痕迹还原过去三日内发生的关键事件。",
        "The protagonist has a star-sand deduction ability to trace past events.",
        "今天天气很好，适合出去散步。",
        "灵气潮汐每三十年涨落一次，潮汐最低时古老遗迹会露出入口。",
    ]

    print("对以下 4 段文字做 embedding：")
    for i, t in enumerate(texts):
        print(f"  [{i + 1}] {t[:80]}...")
    print()

    embeddings = model.encode(texts, normalize_embeddings=True)

    print("每段文字被转成了 512 个浮点数的向量：")
    for i in range(4):
        vec = embeddings[i]
        print(f"  文本[{i + 1}] 前 8 维: {vec[:8].round(4)}")
        print(f"            后 8 维: {vec[-8:].round(4)}")
        print()

    # 计算相似度
    print("--- 文本间语义相似度（-1 到 1，越高越相似）---")
    print()

    pairs = [
        (0, 1, "中文「星砂推演」", "英文「star-sand deduction」"),
        (0, 3, "中文「星砂推演」", "中文「灵气潮汐」"),
        (0, 2, "中文「星砂推演」", "中文「今天天气很好」"),
    ]

    for a, b, label_a, label_b in pairs:
        sim = float(embeddings[a] @ embeddings[b])
        print(f"  {label_a}  vs  {label_b}")
        print(f"  相似度: {sim:+.3f}")

        if sim > 0.6:
            print(f"  -> 非常相似！跨语言、理解语义")
        elif sim > 0.2:
            print(f"  -> 有些关联（都是修真相关话题）")
        else:
            print(f"  -> 几乎无关")
        print()

    input("按 Enter 继续下一步...")


# ===========================================================================
# 第 2 步：向量数据库 (ChromaDB) 怎么工作
# ===========================================================================

def step2_chromadb_basics():
    """手把手演示 ChromaDB 的存储和检索过程。"""
    import chromadb
    from sentence_transformers import SentenceTransformer

    print()
    print("=" * 65)
    print("第 2 步：向量数据库（ChromaDB）是怎么工作的？")
    print("=" * 65)
    print()
    print("一句概括：把向量存进去，查询时找「距离最近」的。")
    print()
    print("对比普通数据库：")
    print("  SQL:    SELECT * FROM table WHERE name = '金丹'")
    print("  Chroma: 找到和查询向量最接近的 K 个文档")
    print()

    # 创建内存数据库
    print("--- 2.1 创建临时数据库 ---")
    client = chromadb.Client()
    collection = client.create_collection(name="tutorial_demo")
    print("创建了 collection: tutorial_demo（内存模式，关闭后消失）")
    print()

    # 准备数据
    print("--- 2.2 存入文档 ---")
    model = SentenceTransformer("BAAI/bge-small-zh-v1.5")

    documents = [
        "金丹期是修真第三个大境界，修炼者需将体内灵力凝聚成固态金丹。",
        "筑基期是修真第二个大境界，修炼者打牢根基、淬炼经脉。",
        "元婴期是修真第四个大境界，金丹破裂后化出元婴，可离体遨游。",
        "炼气期是修真第一个大境界，修炼者引灵气入体、打通经脉。",
        "化神期是修真第五个大境界，元婴与肉身合一，开始感悟天地法则。",
    ]

    embeddings = model.encode(documents, normalize_embeddings=True)

    collection.add(
        ids=["chunk_1", "chunk_2", "chunk_3", "chunk_4", "chunk_5"],
        embeddings=embeddings.tolist(),
        documents=documents,
    )

    print(f"已存入 {len(documents)} 篇修真境界说明")
    for d in documents:
        print(f"  . {d}")
    print()

    # 搜索
    print("--- 2.3 语义搜索 ---")
    print()

    queries = ["结丹的方法是什么", "最开始怎么修炼", "身体和灵魂融合"]

    for query in queries:
        query_emb = model.encode(
            "为这个句子生成表示以用于检索相关文章：" + query,
            normalize_embeddings=True,
        )

        results = collection.query(
            query_embeddings=[query_emb.tolist()],
            n_results=2,
            include=["documents", "distances"],
        )

        print(f'查询: "{query}"')
        for i, (doc, dist) in enumerate(
            zip(results["documents"][0], results["distances"][0])
        ):
            sim = 1.0 - dist
            print(f"  Top{i + 1}（相似度 {sim:.3f}）: {doc}")
        print()

    print(">>> 关键发现：")
    print("  「结丹的方法」-> 找到了「金丹期」而不是「筑基期」")
    print("  「最开始怎么修炼」-> 找到了「炼气期」（理解了'最开始'='第一个'）")
    print("  「身体和灵魂融合」-> 找到了「化神期」（理解了'融合'='合一'）")
    print()

    input("按 Enter 继续下一步...")


# ===========================================================================
# 第 3 步：完整 RAG 流水线
# ===========================================================================

def step3_full_rag_pipeline():
    """完整的 RAG 流程：切块 -> 向量化 -> 检索 -> 拼入 Prompt"""
    import chromadb
    from sentence_transformers import SentenceTransformer

    print()
    print("=" * 65)
    print("第 3 步：完整 RAG 流水线")
    print("=" * 65)
    print()
    print("RAG = Retrieval（检索）+ Augmentation（增强）+ Generation（生成）")
    print()
    print("+-------------------------------------------------------+")
    print("|                                                       |")
    print("|  离线阶段（只执行一次）                                  |")
    print("|    知识文件 -> 切块 -> Embedding -> 存入 ChromaDB        |")
    print("|                                                       |")
    print("|  在线阶段（每次搜索执行）                                 |")
    print("|    用户查询 -> Embedding -> ChromaDB 检索 -> 拼入 Prompt |")
    print("|                                                       |")
    print("+-------------------------------------------------------+")
    print()

    # ---- 离线阶段 ----
    print("--- 离线阶段：构建知识库 ---")
    model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
    client = chromadb.Client()

    knowledge_sources = [
        ("knowledge.txt", """[境界划分]
炼气、筑基、金丹、元婴、化神。每个大境界分为初期、中期、后期、圆满。

[主角能力]
主角林照夜拥有"星砂推演"能力，可通过星光痕迹还原过去三日内发生的关键事件。

[世界背景]
故事发生在云衡大陆。灵气潮汐每三十年涨落一次，潮汐最低时古老遗迹会露出入口。
"""),
        ("golden-finger-design.md", """# Golden Finger Design

A strong golden finger is a key, a witness, and a source of pressure.

The protagonist's golden finger should:
1. Be unique and not replicable by others
2. Have clear limitations and costs
3. Grow in capability as the story progresses
"""),
        ("factions.md", """# 六院体系
六院是统治修真界的六大机构：
- 天机院：天文、预言、星象
- 丹霞院：炼丹、医药、毒理
- 剑澜院：剑道、战斗、杀伐
- 灵枢院：阵法、符箓、禁制
- 万象院：妖兽、灵植、驯养
- 红尘院：世俗、权谋、情报
"""),
    ]

    # 切块
    from local_tools import _split_sections

    chunks = []
    for source_name, text in knowledge_sources:
        sections = _split_sections(text)
        for i, section in enumerate(sections):
            clean = section.strip()
            if clean and len(clean) > 20:
                chunks.append({
                    "text": clean, "source": source_name, "index": i
                })

    print(f"从 {len(knowledge_sources)} 个文件中切出 {len(chunks)} 个文本块")
    print()

    # 向量化并存入
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, normalize_embeddings=True)

    try:
        client.delete_collection("rag_pipeline_demo")
    except Exception:
        pass
    collection = client.create_collection(name="rag_pipeline_demo")
    collection.add(
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        embeddings=embeddings.tolist(),
        documents=texts,
        metadatas=[{"source": c["source"]} for c in chunks],
    )
    print(f"已存入 {collection.count()} 个向量到 ChromaDB")
    print()

    # ---- 在线阶段 ----
    print("--- 在线阶段：用户查询 -> 检索 -> 拼入 Prompt ---")
    print()

    test_queries = [
        "主角有什么特殊能力",
        "golden finger的设计原则",
        "六院分别负责什么",
    ]

    for query in test_queries:
        query_emb = model.encode(
            "为这个句子生成表示以用于检索相关文章：" + query,
            normalize_embeddings=True,
        )

        results = collection.query(
            query_embeddings=[query_emb.tolist()],
            n_results=3,
            include=["documents", "metadatas", "distances"],
        )

        # 组装检索结果
        retrieval_text = ""
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            sim = 1.0 - dist
            retrieval_text += f"\n[来源：{meta['source']} (相关度 {sim:.2f})]\n{doc}\n"

        # 模拟最终发给 LLM 的 Prompt
        print(f'查询: "{query}"')
        print(f"检索到的上下文片段: {retrieval_text[:250]}...")
        print()
        print("+-- 如果调用 LLM，完整的 Prompt 将是 --")
        prompt = f"""你是一个小说创作助手。请基于以下知识库内容回答用户问题。

知识库内容：
{retrieval_text}

用户问题：{query}

请基于知识库回答，如果知识库中没有相关信息，请说明。"""
        print(prompt[:400])
        print("+-----------------------------------------")
        print()

    print(">>> 这就是 RAG 的完整流程：")
    print("  1. 用户问题 -> Embedding -> 向量")
    print("  2. 向量 -> ChromaDB 搜索 -> 最相关的文档")
    print("  3. 文档 + 用户问题 -> 拼入 Prompt -> 发给 LLM")
    print("  4. LLM '看到'了私有知识，回答就更准确")
    print()

    input("按 Enter 继续下一步...")


# ===========================================================================
# 第 4 步：关键参数调优
# ===========================================================================

def step4_key_parameters():
    """理解 top_k、切块大小等对检索效果的影响。"""
    import chromadb
    from sentence_transformers import SentenceTransformer

    print()
    print("=" * 65)
    print("第 4 步：关键参数调优")
    print("=" * 65)
    print()

    model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
    client = chromadb.Client()

    documents = [
        "金丹期：修炼者将灵力凝聚为固态金丹，需要极高的控制力和大量灵气。",
        "金丹期突破过程极为凶险，稍有不慎就会丹碎人亡。",
        "金丹品质分为九转，一转最次、九转最高。",
        "筑基期是为金丹期打基础的阶段，筑基越扎实，结丹成功率越高。",
        "炼丹需要将各种灵药按比例投入丹炉，以真火炼化数日方可成丹。",
        "金丹修士寿元可达五百年，远超筑基期的两百岁。",
        "元婴期修士可将元婴寄托于法宝中，即使肉身毁灭也能存活。",
        "化神期是修真第五个境界，需要将元婴与肉身完全融合。",
    ]

    embeddings = model.encode(documents, normalize_embeddings=True)
    try:
        client.delete_collection("params_demo")
    except Exception:
        pass
    collection = client.create_collection(name="params_demo")
    collection.add(
        ids=[f"d_{i}" for i in range(len(documents))],
        embeddings=embeddings.tolist(),
        documents=documents,
    )

    # ---- 参数 1：top_k ----
    print("--- 参数 1：top_k（返回几个结果）---")
    print()

    query = "金丹期的修炼要点"
    query_emb = model.encode(
        "为这个句子生成表示以用于检索相关文章：" + query,
        normalize_embeddings=True,
    )

    for k in [1, 3, 5]:
        results = collection.query(
            query_embeddings=[query_emb.tolist()],
            n_results=k,
            include=["documents", "distances"],
        )
        print(f"top_k={k}:")
        for doc, dist in zip(results["documents"][0], results["distances"][0]):
            print(f"  [{1.0 - dist:.3f}] {doc[:80]}")
        print()

    print(">>> top_k 越大，给 LLM 的上下文越多，但噪音也越多。")
    print("   建议 top_k=5，太多会干扰写作方向。")
    print()

    # ---- 参数 2：切块大小 ----
    print("--- 参数 2：文本切块（chunking）的影响 ---")
    print()

    print("大块方式：整个境界体系存为一个 chunk")
    print("  内容: '炼气期是第一个境界...化神期是第五个境界...'（共 5 个境界混在一起）")
    print("  问题: 搜索「筑基」时整段返回，LLM 要自己找到筑基部分")
    print()

    print("小块方式：每个境界一个 chunk")
    for c in [
        "炼气期：修真第一个境界，吸收灵气入体、打通经脉。",
        "筑基期：修真第二个境界，淬炼经脉、打牢根基。",
        "金丹期：修真第三个境界，灵力凝聚为固态金丹。",
        "元婴期：修真第四个境界，金丹破裂化出元婴，可离体遨游。",
        "化神期：修真第五个境界，元婴与肉身合一，感悟天地法则。",
    ]:
        print(f"  . {c}")
    print("  优势: 搜索「筑基」时精准命中筑基期那一个块，不被其他境界干扰")
    print()

    input("按 Enter 继续下一步...")


# ===========================================================================
# 第 5 步：在真实知识库上运行
# ===========================================================================

def step5_your_real_knowledge_base():
    """在你的 knowledge.txt + knowledge/*.md 上运行语义搜索。"""
    from knowledge_base import get_knowledge_base

    print()
    print("=" * 65)
    print("第 5 步：在你的真实知识库上运行")
    print("=" * 65)
    print()

    kb = get_knowledge_base()

    print("正在检查知识库索引...")
    count = kb.check_and_refresh()
    print(f"当前索引中有 {count} 个文本块。")
    print()

    stats = kb.get_stats()
    print(f"Embedding 模型: {stats['model']}")
    print(f"向量维度: 512")
    print(f"存储位置: {stats['persist_dir']}")
    print()

    test_queries = [
        "主角的特殊能力是什么",
        "金丹期怎么突破",
        "六院有哪些",
        "golden finger should have limitations",
        "反派怎么设计",
        "境界划分体系",
    ]

    for query in test_queries:
        result = kb.search(query, top_k=3)
        lines = result.split("\n")
        print(f'搜索: "{query}"')
        for line in lines:
            if line.startswith(("1.", "2.", "3.")):
                idx = line.index("：") if "：" in line else 80
                print(f"  -> {line[idx:] if idx < 80 else line[:120]}")
        print()

    print(">>> 以上是你真实知识库的搜索结果。")
    print("   注意看英文查询如何匹配到英文文档。")
    print()

    input("按 Enter 继续最后一步...")


# ===========================================================================
# 第 6 步：在 agent 中的调用路径
# ===========================================================================

def step6_in_your_agent():
    """演示 knowledge_base 如何被 agent.py 调用。"""
    from local_tools import search_local_knowledge

    print()
    print("=" * 65)
    print("第 6 步：在真实 agent 中的调用路径")
    print("=" * 65)
    print()

    print("当 LLM 调用 search_local_knowledge 工具时：")
    print()
    print("  agent.py 第 470 行：")
    print("  +--------------------------------------------------")
    print("  | if name == 'search_local_knowledge':")
    print("  |     content = search_local_knowledge(query)")
    print("  |     # 返回的文本会拼入 messages，LLM 就能看到知识库")
    print("  +--------------------------------------------------")
    print()
    print("  local_tools.py search_local_knowledge() 第 391 行：")
    print("    1. 尝试 get_knowledge_base().search(query)")
    print("       -> Embedding 语义搜索（安装了 sentence-transformers）")
    print("    2. 如果失败，降级到 _search_local_knowledge_tfidf()")
    print("       -> jieba + TF-IDF 关键词搜索（始终可用）")
    print()

    print("--- 实际调用演示 ---")
    print()
    result = search_local_knowledge("金手指怎么设计")
    print(result[:800])
    print()

    if "语义检索结果" in result:
        print(">>> 当前使用：Embedding 语义搜索（新方案）")
    else:
        print(">>> 当前使用：TF-IDF 关键词搜索（旧方案降级）")
    print()

    print("=" * 65)
    print("教程完成！")
    print()
    print("你应该已经理解了：")
    print("  1. Embedding: 文字 -> 数字向量，向量相近 = 语义相近")
    print("  2. ChromaDB: 存向量、搜向量，毫秒级返回最相似的文档")
    print("  3. RAG 流水线: 切块 -> 向量化 -> 存入 -> 查询时检索 -> 拼入 Prompt")
    print("  4. 你的项目: search_local_knowledge -> knowledge_base -> ChromaDB")
    print()
    print("下一步学习方向：")
    print("  - LangChain: 把上述流程用框架封装")
    print("  - LangGraph: 多步骤 agent 流程")
    print("  - 调优: 尝试不同的 embedding 模型、切块策略、top_k 值")
    print("=" * 65)


# ===========================================================================
# 主程序
# ===========================================================================

def main():
    steps = [
        ("1", "什么是 Embedding", step1_what_is_embedding),
        ("2", "ChromaDB 怎么工作", step2_chromadb_basics),
        ("3", "完整 RAG 流水线", step3_full_rag_pipeline),
        ("4", "关键参数调优", step4_key_parameters),
        ("5", "在你的真实知识库上运行", step5_your_real_knowledge_base),
        ("6", "在 agent 中的调用路径", step6_in_your_agent),
    ]

    print()
    print("+==============================================================+")
    print("|          RAG 深度教程 -- 从零吃透语义搜索                      |")
    print("+==============================================================+")
    print("|  共 6 步，每步运行完会暂停，给你时间消化中间结果                 |")
    print("|  随时输入 q + Enter 可退出                                    |")
    print("+==============================================================+")
    print()

    print("教程步骤：")
    for num, title, _ in steps:
        print(f"  第 {num} 步: {title}")
    print()

    for num, title, func in steps:
        print(f"\n{'=' * 65}")
        choice = input(f"第 {num} 步「{title}」-- 按 Enter 执行, s 跳过, q 退出: ").strip().lower()
        if choice == "q":
            print("已退出教程。")
            return
        if choice == "s":
            print(f"已跳过第 {num} 步。")
            continue
        func()

    print()
    print("全部 6 步完成！你可以随时重新运行 python rag_tutorial.py 复习。")


if __name__ == "__main__":
    main()
