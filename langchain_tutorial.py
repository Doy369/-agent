"""
================================================================================
LangChain 深度教程 -- 从你的手写代码到框架封装
================================================================================

核心思路：
  你不是"从头学 LangChain"，你是"把手写代码翻译成 LangChain"。
  每一步都是：先看你现在的代码，再看 LangChain 等价写法，最后解释为什么。

运行方式：python langchain_tutorial.py
每次暂停可看清中间结果，输入 q 可随时退出。

================================================================================
"""

from __future__ import annotations
import sys, os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


# ═══════════════════════════════════════════════════════════════════════════
# 第 1 步：Embedding 模型 -- 你手写 vs LangChain
# ═══════════════════════════════════════════════════════════════════════════

def step1_embedding_comparison():
    print("=" * 65)
    print("第 1 步：Embedding 模型加载")
    print("=" * 65)
    print()
    print("你的手写代码（knowledge_base.py 约第 160 行）：")
    print("  from sentence_transformers import SentenceTransformer")
    print('  model = SentenceTransformer("BAAI/bge-small-zh-v1.5")')
    print('  vec = model.encode("主角突破金丹境界", normalize_embeddings=True)')
    print()
    print("LangChain 等价写法：")
    print('  from langchain_community.embeddings import HuggingFaceEmbeddings')
    print('  embeddings = HuggingFaceEmbeddings(')
    print('      model_name="BAAI/bge-small-zh-v1.5",')
    print('      model_kwargs={"device": "cpu"},')
    print('      encode_kwargs={"normalize_embeddings": True},')
    print('  )')
    print('  vec = embeddings.embed_query("主角突破金丹境界")  # 查询用这个')
    print('  vecs = embeddings.embed_documents(["文本1", "文本2"])  # 文档用这个')
    print()

    # 运行对比
    input("按 Enter 运行两种方式对比...")

    from sentence_transformers import SentenceTransformer
    from langchain_community.embeddings import HuggingFaceEmbeddings

    text = "主角林照夜拥有星砂推演能力"
    query_text = "BGE 查询前缀：" + text

    # 手写方式
    st_model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
    st_vec = st_model.encode(text, normalize_embeddings=True)

    # LangChain 方式
    lc_embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    lc_vec = lc_embeddings.embed_query(text)

    print(f"手写方式 shape: {st_vec.shape} 前5维: {st_vec[:5].round(4)}")
    print(f"LangChain 方式 shape: {len(lc_vec)} 前5维: {[round(x,4) for x in lc_vec[:5]]}")
    print()

    print(">>> 为什么 LangChain 要封装这一步？")
    print()
    print("  1. 统一接口：不管底层是 BGE、OpenAI、还是 Cohere，")
    print("     调用方式都是 embed_query() / embed_documents()")
    print()
    print("  2. 自动处理细节：比如 BGE 的查询前缀，LangChain 内部帮你加了，")
    print("     你不用记住「BGE 要加前缀，OpenAI 不要」这种差异。")
    print()
    print("  3. 和下游组件无缝对接：VectorStore 只认 embed_query 这个接口，")
    print("     你换 embedding 模型时不用改其他代码。")
    print()

    input("按 Enter 继续下一步...")


# ═══════════════════════════════════════════════════════════════════════════
# 第 2 步：文本切块 -- 你手写 vs LangChain
# ═══════════════════════════════════════════════════════════════════════════

def step2_splitting_comparison():
    print()
    print("=" * 65)
    print("第 2 步：文本切块（Text Splitting）")
    print("=" * 65)
    print()

    print("你的手写代码（local_tools.py _split_sections 约第 282 行）：")
    print('  re.split(r"\\n\\s*\\n|(?=【)|(?=^#{1,3})|(?=^---)", text)')
    print("  逻辑：按空行、标题、分隔符切分")
    print()

    print("LangChain 等价写法：")
    print("  RecursiveCharacterTextSplitter(")
    print("      chunk_size=300,")
    print("      chunk_overlap=50,")
    print("      separators=['\\n\\n', '\\n', '。', '！', '？', ' '],")
    print("  )")
    print()

    print(">>> 核心区别：你的切块是'自然段落切分'，LangChain 多了两个概念：")
    print()
    print("  chunk_size（块大小）：每个块最多多少个字符")
    print("    太小(50字)  → 信息碎片化，搜「金丹期突破条件」可能只命中「金丹期」")
    print("    太大(2000字) → 噪音多，LLM 要在长篇里自己找相关信息")
    print("    建议：300-500 字，和一个小段落差不多")
    print()
    print("  chunk_overlap（重叠量）：相邻两块之间重叠多少字符")
    print("    不重叠 → 「金丹需要极高控制力」被切断在块边界，搜不到")
    print("    重叠50字 → 关键句子在两个块里都有，搜索不会漏")
    print("    建议：chunk_size 的 10-15%")
    print()

    # 演示
    input("按 Enter 看切块效果对比...")

    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from local_tools import _split_sections, _read_knowledge_sources

    # 拿一段真实知识文本
    sample = """【境界划分】
炼气、筑基、金丹、元婴、化神。每个大境界分为初期、中期、后期、圆满。

【金丹期详解】
金丹期是修真第三个大境界。修炼者需将体内灵力凝聚成固态金丹。
金丹品质分为九转，一转最次，九转最高。突破过程极为凶险，
稍有不慎就会丹碎人亡。金丹修士寿元可达五百年，远超筑基期的两百岁。

【元婴期详解】
金丹破裂后化出元婴，可离体遨游。元婴期修士可将元婴寄托于法宝中，
即使肉身毁灭也能存活。元婴期是修真者的重要转折点。"""

    # 你的手写方式
    old_chunks = _split_sections(sample)
    print(f"手写 _split_sections：共 {len(old_chunks)} 块")
    for i, c in enumerate(old_chunks):
        print(f"  块{i+1} ({len(c)}字): {c[:80]}...")

    print()

    # LangChain 方式
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=30,
        separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""],
    )
    lc_chunks = splitter.split_text(sample)
    print(f"LangChain RecursiveCharacterTextSplitter：共 {len(lc_chunks)} 块")
    for i, c in enumerate(lc_chunks):
        print(f"  块{i+1} ({len(c)}字): {c[:80]}...")

    print()
    print(">>> 注意 LangChain 切得更均匀，而且有重叠。")
    print("    当你换一个知识库（比如英文文档），只需改 chunk_size 和 separators，")
    print("    不用重写正则表达式。")
    print()

    input("按 Enter 继续下一步...")


# ═══════════════════════════════════════════════════════════════════════════
# 第 3 步：向量存储 -- 你手写 Numpy 矩阵 vs LangChain VectorStore
# ═══════════════════════════════════════════════════════════════════════════

def step3_vectorstore_comparison():
    print()
    print("=" * 65)
    print("第 3 步：向量存储和搜索")
    print("=" * 65)
    print()

    print("你的手写代码（knowledge_base.py）：")
    print("  存储：np.savez_compressed('vector_index.npz', embeddings=embeddings)")
    print("  搜索：scores = embeddings @ query_vec    # 矩阵乘法")
    print("       top_indices = np.argpartition(scores, -k)[-k:]")
    print()
    print("LangChain 等价写法：")
    print("  from langchain_community.vectorstores import FAISS")
    print("  vectorstore = FAISS.from_documents(docs, embeddings)")
    print("  results = vectorstore.similarity_search('金丹突破', k=5)")
    print()
    print(">>> 为什么用 FAISS 替代手写 Numpy 矩阵？")
    print()
    print("  1. FAISS 是 Meta 开源的向量搜索库，专门优化过大量向量的搜索速度。")
    print("     1000 个块：numpy 矩阵乘法 ~0.5ms，FAISS ~0.2ms（差不多）")
    print("     10 万个块：numpy ~200ms，FAISS ~2ms（100 倍差距）")
    print()
    print("  2. LangChain 的 VectorStore 统一了接口：")
    print("     vectorstore.similarity_search(query, k=5)")
    print("     不管底层是 FAISS、ChromaDB、Pinecone 还是 Weaviate，")
    print("     调用方式完全一样。你可以从本地 FAISS 无缝迁移到云端的 Pinecone。")
    print()
    print("  3. 自带 disk 持久化：FAISS.save_local() / FAISS.load_local()")
    print("     比你自己管理 .npz + .pkl 更省心。")
    print()

    input("按 Enter 用 FAISS 在你的真实知识库上运行...")

    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from local_tools import _read_knowledge_sources

    # 复用你现有的知识库读取
    sources = _read_knowledge_sources()
    print(f"读取了 {len(sources)} 个知识文件")

    # LangChain 切块（你要手动调 chunk_size）
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "！", "？", "，", "  ", " ", ""],
    )

    all_texts = []
    all_metadatas = []
    for source_name, text in sources:
        chunks = splitter.split_text(text)
        for chunk in chunks:
            if len(chunk.strip()) > 20:
                all_texts.append(chunk.strip())
                all_metadatas.append({"source": source_name})

    print(f"切出 {len(all_texts)} 个文本块")

    # LangChain 向量化 + 存储
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    print("正在构建 FAISS 向量库...")
    vectorstore = FAISS.from_texts(
        all_texts,
        embeddings,
        metadatas=all_metadatas,
    )
    print(f"FAISS 向量库构建完成：{vectorstore.index.ntotal} 个向量")

    # 搜索演示
    print()
    print("--- 搜索演示 ---")
    queries = ["金丹境界突破", "golden finger设计", "六院有哪些"]
    for q in queries:
        results = vectorstore.similarity_search(q, k=3)
        print(f'搜索: "{q}"')
        for i, doc in enumerate(results):
            # FAISS 返回的是 L2 距离，越小越相似
            print(f"  {i+1}. [{doc.metadata['source']}] {doc.page_content[:100]}...")
        print()

    # 持久化演示：FAISS 写入需要纯 ASCII 路径
    save_dir = Path(__file__).resolve().parent / "dist" / "faiss_index"
    save_dir.mkdir(parents=True, exist_ok=True)
    try:
        save_path = str(save_dir).encode('ascii', 'ignore').decode()
        vectorstore.save_local(save_path)
        print(f"已保存到磁盘: {save_path}")
        print(f"  文件: {', '.join(f.name for f in save_dir.iterdir())}")
    except Exception as e:
        print(f"(保存跳过: {e})")
        print(f"实际项目用你已有的 vector_db 方案持久化即可。")

    print()
    print(">>> FAISS.save_local() 和你的 np.savez() 做的事一样——")
    print("    把向量和元数据持久化到磁盘，下次启动直接 load。")
    print("    区别是 FAISS 还存了搜索索引结构，搜索更快。")
    print()

    input("按 Enter 继续下一步...")


# ═══════════════════════════════════════════════════════════════════════════
# 第 4 步：Prompt 模板 -- 你手写 vs LangChain ChatPromptTemplate
# ═══════════════════════════════════════════════════════════════════════════

def step4_prompt_template():
    print()
    print("=" * 65)
    print("第 4 步：Prompt 模板")
    print("=" * 65)
    print()

    print("你的手写代码（agent.py generate_next_chapter 约第 230 行）：")
    print('  user_prompt = f"""请生成第 {next_index} 章正文。')
    print()
    print('  小说类型：{style}')
    print('  目标篇幅：约 {config.max_words} 字')
    print()
    print('  {memory_context}')
    print()
    print('  {auto_context}')
    print()
    print('  写作要求：...')
    print('  """')
    print()

    print("问题：f-string 拼 Prompt 有四个缺陷：")
    print("  1. 嵌套复杂时容易漏括号、变量名拼错")
    print("  2. DeepSeek 和 OpenAI 和 Anthropic 的消息格式不同，")
    print("     你需要手动写两套（你确实写了 _call_openai_compatible 和 _call_anthropic）")
    print("  3. 多条消息（system + user + assistant）混在一起难以管理")
    print("  4. 没法复用 —— 大纲生成和章节生成各写了各的 f-string")
    print()

    print("LangChain ChatPromptTemplate 等价写法：")
    print()
    print("  from langchain_core.prompts import ChatPromptTemplate")
    print()
    print("  chapter_prompt = ChatPromptTemplate.from_messages([")
    print("      ('system', system_template),   # 角色提示，固定不变")
    print("      ('user', user_template),       # 具体请求，每次不同")
    print("  ])")
    print()
    print("  messages = chapter_prompt.invoke({")
    print("      'style': '玄幻',")
    print("      'chapter_index': 3,")
    print("      'outline': '...',")
    print("      'recent_chapters': '...',")
    print("      'knowledge_context': '...',")
    print("  })")
    print("  # 返回：ChatPromptValue，.to_messages() 得到标准消息列表")
    print()

    # 实际演示
    input("按 Enter 看实际效果...")

    from langchain_core.prompts import ChatPromptTemplate

    # 你的 system prompt
    system_template = """你是一名金牌网络小说作家。你的任务是帮助用户创作高完成度的小说。

写作要求：
1. 严格尊重用户给定的世界观、人物设定、章节大纲和前文事实。
2. 每章要有清晰的目标、冲突、反转或钩子。
3. 文风要贴合用户选择的类型。
"""

    # 你的章节生成 prompt（原来是 f-string）
    user_template = """请生成第 {chapter_index} 章正文。

小说类型：{style}
目标篇幅：约 {word_count} 字

{outline_section}

{previous_chapters}

{knowledge_context}

写作要求：
1. 章节标题放在第一行。
2. 承接前文，不要重启故事。
3. 至少安排一个推进主线的行动、一个人物选择、一个结尾钩子。
"""

    chapter_prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("user", user_template),
    ])

    # 模拟调用
    messages = chapter_prompt.invoke({
        "style": "玄幻",
        "chapter_index": 3,
        "word_count": 2500,
        "outline_section": "【全书大纲】第三卷：主角首次渡劫，揭露幕后黑手...",
        "previous_chapters": "【前文概要】\n第1章：林照夜觉醒了星砂推演能力...\n第2章：被逐出家族，独自前往云衡城...",
        "knowledge_context": "【知识库检索结果】\n1. 金丹期：修炼者需凝聚灵力...",
    })

    print("invoke() 返回的消息列表：")
    print(f"  类型: {type(messages).__name__}")
    print(f"  消息数: {len(messages.to_messages())}")
    print()

    for i, msg in enumerate(messages.to_messages()):
        print(f"  --- 消息 {i+1} ---")
        print(f"  角色: {msg.type}")
        print(f"  内容(前120字): {repr(msg.content)[:120]}...")
        print()

    print(">>> 关键优势：")
    print()
    print("  1. 模板和变量分离：Prompt 结构写一次，变量从外面传进来")
    print("  2. 自动适配 API 格式：")
    print("     to_messages() 返回标准格式 -> LangChain 的 ChatModel 自动转成")
    print("     OpenAI 的 {role, content} / Anthropic 的 system+content blocks")
    print("     你不需要再写 _call_openai_compatible 和 _call_anthropic 两套代码！")
    print("  3. 模板复用：大纲生成、章节生成、润色可以共用 system 模板，")
    print("     只换 user 模板")
    print()

    input("按 Enter 继续下一步...")


# ═══════════════════════════════════════════════════════════════════════════
# 第 5 步：完整的 RAG 链 -- 用 LCEL 把组件串起来
# ═══════════════════════════════════════════════════════════════════════════

def step5_rag_chain():
    print()
    print("=" * 65)
    print("第 5 步：RAG 链 -- 用 LCEL 串联所有组件")
    print("=" * 65)
    print()

    print("你现在的手写流程（agent.py）：")
    print("  search_local_knowledge(query)  → 返回文本")
    print('  user_prompt = f"...{search_result}..."   → 拼字符串')
    print("  httpx.post(url, json=payload)  → 调 API")
    print("  全部手动串联")
    print()

    print("LangChain LCEL（LangChain Expression Language）：")
    print()
    print("  chain = (")
    print("      {")
    print('          "context": retriever,            # 自动检索')
    print('          "question": RunnablePassthrough(), # 透传用户输入')
    print("      }")
    print("      | prompt          # 自动拼 Prompt")
    print("      | llm             # 自动调 API")
    print("      | StrOutputParser() # 自动提取文本")
    print("  )")
    print()
    print("  result = chain.invoke('金丹期怎么突破')")
    print()
    print("  | 是管道操作符，把上一个组件的输出传给下一个组件。")
    print("  这和 Unix 的 ls | grep | sort 是完全一样的思路。")
    print()

    print(">>> 为什么叫 LCEL？")
    print("  很多框架用 YAML/JSON 配置文件来编排流程。")
    print("  LangChain 选的是 Python 表达式：用 | 连接组件。")
    print("  好处：有 IDE 自动补全、类型检查、调试方便。")
    print()

    input("按 Enter 构建一个完整的 RAG 链...")

    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser

    # 1. 加载之前保存的 FAISS 索引
    save_dir = Path(__file__).resolve().parent / "dist" / "faiss_index"
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    if save_dir.exists():
        vectorstore = FAISS.load_local(
            str(save_dir),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        print(f"从磁盘加载 FAISS 索引：{vectorstore.index.ntotal} 个向量")
    else:
        print("未找到已保存的索引，请先运行第3步。")
        return

    # 2. 创建 retriever
    #    as_retriever() 把 vectorstore 包装成一个可调用对象
    #    传给它 query，它返回相关文档列表
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 5},  # 每次检索返回 5 个文档
    )

    print(f"Retriever 创建完成（k=5）。")
    print()

    # 3. 定义 Prompt 模板
    #    注意 {context} 和 {question} 是两个占位符
    #    context 由 retriever 自动填充
    #    question 由用户输入填充
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一名金牌网络小说作家。请基于以下知识库内容回答创作相关问题。"),
        ("user", """知识库内容：
{context}

用户问题：{question}

请基于知识库回答。如果知识库中没有相关信息，请如实说明。"""),
    ])

    # 4. 用 LCEL 串联成链
    #    这一步不调用 LLM（避免消耗 API Token），
    #    只演示"链的结构"——看数据怎么流
    rag_chain = (
        {
            "context": retriever,           # ① 检索
            "question": RunnablePassthrough(),  # ② 原样传递用户输入
        }
        | prompt                             # ③ 拼 Prompt
    )

    # 5. 测试（只到 prompt 这步，不实际调 LLM）
    print("--- 测试链的前半段（检索 + 拼 Prompt）---")
    print()

    test_queries = ["主角的特殊能力", "六院体系包含哪些"]

    for query in test_queries:
        result = rag_chain.invoke(query)
        print(f'用户输入: "{query}"')
        print(f"输出类型: {type(result).__name__}")
        print(f"消息角色: {result.messages[0].type}")
        print(f"消息角色: {result.messages[1].type}")
        # 显示拼接后的完整 prompt 的前半部分
        user_content = result.messages[1].content
        print(f"拼接后的 user prompt (前 400 字):")
        print(user_content[:400])
        print("...")
        print()

    print(">>> LCEL 管道的本质：")
    print()
    print("  {" + '"context": retriever, "question": RunnablePassthrough()' + "}")
    print("  这是一个字典，每个值是一个可调用对象。")
    print("  invoke时，每个值都收到同一个输入，并行执行：")
    print("    retriever.invoke('主角的特殊能力') → [Document1, Document2, ...]")
    print("    RunnablePassthrough().invoke('主角的特殊能力') → '主角的特殊能力'")
    print("  然后把结果作为 context 和 question 传给 prompt。")
    print()
    print("  | prompt：收到 {context, question} → 填充模板 → 返回消息列表")
    print("  | llm：收到消息列表 → 调 API → 返回 AIMessage")
    print("  | StrOutputParser：收到 AIMessage → 提取 .content")
    print()

    print(">>> 对比你的手写代码：")
    print("  你的: search → f-string拼 → httpx发送 → 手动解析")
    print("  LCEL: retriever → prompt → llm → StrOutputParser")
    print("  同样 4 步，但 LCEL 是声明式的，改顺序/换组件只需要调整 | 符号")
    print()

    input("按 Enter 继续最后一步...")


# ═══════════════════════════════════════════════════════════════════════════
# 第 6 步：Agent 工具 -- 你手写 vs LangChain Tool
# ═══════════════════════════════════════════════════════════════════════════

def step6_agent_tools():
    print()
    print("=" * 65)
    print("第 6 步：Agent 工具封装")
    print("=" * 65)
    print()

    print("你的手写代码（agent.py）：")
    print("  OPENAI_TOOL_SPECS = [{")
    print('      "type": "function",')
    print('      "function": {')
    print('          "name": "search_local_knowledge",')
    print('          "description": "...",')
    print('          "parameters": { ... }')
    print("      }")
    print("  }]")
    print()
    print("  然后 _execute_tool() 写了一个大 if/elif 链来分发：")
    print('    if name == "search_info": ...')
    print('    elif name == "generate_image": ...')
    print('    elif name == "search_local_knowledge": ...')
    print("    ...")
    print()

    print("LangChain 等价写法：")
    print()
    print("  from langchain_core.tools import tool")
    print()
    print("  @tool")
    print("  def search_local_knowledge(query: str) -> str:")
    print('      """本地设定集检索。查找世界观、境界、人物能力等设定。"""')
    print("      from local_tools import search_local_knowledge")
    print("      return search_local_knowledge(query)")
    print()
    print(">>> 装饰器 @tool 自动做了什么：")
    print("  1. 函数名 → tool.name")
    print("  2. 文档字符串 → tool.description")
    print("  3. 函数的参数类型注解 → tool.args_schema（参数 schema）")
    print("  4. 函数体 → tool._run()（执行逻辑）")
    print()
    print("  你不需要手写 JSON Schema 了。加了新工具只需多写一个 @tool 函数，")
    print("  不用改 if/elif 链。")
    print()

    input("按 Enter 看实际效果...")

    from langchain_core.tools import tool
    from local_tools import search_local_knowledge as _search_knowledge
    from local_tools import local_text_audit as _audit
    from local_tools import generate_image as _gen_image

    # 定义工具 —— 只需装饰器 + 类型注解 + 文档字符串
    @tool
    def search_knowledge(query: str) -> str:
        """本地设定集语义检索。查找世界观、境界体系、人物能力、势力组织等设定。支持中文和英文搜索。"""
        return _search_knowledge(query)

    @tool
    def audit_text(text: str) -> str:
        """本地敏感词与错别字检测。检查章节正文中的错别字、标点问题和敏感词。"""
        return _audit(text)

    @tool
    def create_illustration(prompt: str) -> str:
        """AI 插图生成。根据章节高光场景描述生成插图。参数 prompt 为画面描述。"""
        return _gen_image(prompt)

    tools = [search_knowledge, audit_text, create_illustration]

    print("已定义的 LangChain 工具：")
    print()
    for t in tools:
        print(f"  名称: {t.name}")
        print(f"  描述: {t.description[:80]}...")
        if hasattr(t, 'args_schema'):
            schema = t.args_schema.model_json_schema()
            props = schema.get('properties', {})
            print(f"  参数: {list(props.keys())}")
        print()

    # 演示：工具可以像普通函数一样调用
    print("--- 直接调用工具 ---")
    result = search_knowledge.invoke({"query": "金丹期"})
    print(f"search_knowledge('金丹期') 返回 (前 200 字):")
    print(result[:200])
    print()

    print(">>> 下一步：把这些工具传给 LLM，由 LLM 决定什么时候调用哪个。")
    print("    LangChain 的 create_tool_calling_agent() + AgentExecutor 会自动")
    print("    处理调用循环，你不需要写那个 for _ in range(4) 的 while 循环了。")
    print()

    print("=" * 65)
    print("教程完成！你应该已经理解：")
    print()
    print("  1. HuggingFaceEmbeddings -- 统一的 embedding 接口")
    print("  2. RecursiveCharacterTextSplitter -- 配置化的文本切块")
    print("  3. FAISS VectorStore -- 向量存储 + 搜索 + 持久化")
    print("  4. ChatPromptTemplate -- 模板化 prompt 管理")
    print("  5. LCEL -- 用 | 管道符串联组件形成 chain")
    print("  6. @tool 装饰器 -- 自动把函数变成 LLM 可调用的工具")
    print()
    print("下一步：langchain_agent.py 用以上所有组件重写 NovelAgent，")
    print("实现完整的小说写作 Agent（能实际调用 DeepSeek）。")
    print("=" * 65)


# ═══════════════════════════════════════════════════════════════════════════
# 主程序
# ═══════════════════════════════════════════════════════════════════════════

def main():
    steps = [
        ("1", "Embedding 模型 -- 手写 vs LangChain", step1_embedding_comparison),
        ("2", "文本切块 -- _split_sections vs TextSplitter", step2_splitting_comparison),
        ("3", "向量存储 -- Numpy 矩阵 vs FAISS", step3_vectorstore_comparison),
        ("4", "Prompt 模板 -- f-string vs ChatPromptTemplate", step4_prompt_template),
        ("5", "RAG 链 -- 用 LCEL 串联所有组件", step5_rag_chain),
        ("6", "Agent 工具 -- if/elif 链 vs @tool 装饰器", step6_agent_tools),
    ]

    print()
    print("+==============================================================+")
    print("|     LangChain 深度教程 -- 从手写代码到框架封装                |")
    print("+==============================================================+")
    print("|  每步对比「你的手写代码」和「LangChain 等价写法」              |")
    print("|  随时输入 q + Enter 可退出                                    |")
    print("+==============================================================+")
    print()

    for num, title, func in steps:
        print(f"\n{'=' * 65}")
        choice = input(f"第 {num} 步「{title}」-- Enter 执行, s 跳过, q 退出: ").strip().lower()
        if choice == "q":
            print("已退出。")
            return
        if choice == "s":
            print(f"已跳过第 {num} 步。")
            continue
        func()

    print()
    print("全部完成！用 python langchain_tutorial.py 随时复习。")


if __name__ == "__main__":
    main()
