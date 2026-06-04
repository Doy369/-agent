# 小说 AI Agent 桌面应用

这是一个 PyQt6 桌面应用代码框架，包含：

- API 配置区：API Key、Base URL、模型、接口格式。
- 小说设定区：设定输入、风格选择、生成字数滑动条。
- 核心按钮：生成大纲、生成下一章、润色文本、中止生成、生成本章插图。
- 中央富文本编辑器：展示与手动编辑生成内容。
- 历史与状态：历史章节列表、保存当前章、状态栏。
- Skill 系统：OpenAI/Anthropic Function Calling、本地知识库、错字检查、全书统计、Mock 插图。
- 前端导入：可在【本地工具箱】中导入知识库文件与本地 Python Skill。

## 运行

```powershell
cd C:\Users\13432\Desktop\agent开发项目\novel_ai_agent
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
```

默认 DeepSeek OpenAI 兼容配置：

- Base URL: `https://api.deepseek.com/v1`
- Model: `deepseek-chat`
- 接口格式：`OpenAI 兼容`

Anthropic 配置：

- Base URL: `https://api.anthropic.com`
- Model: `claude-3-5-sonnet-latest`
- 接口格式：`Anthropic`

## 打包为 exe

```powershell
cd C:\Users\13432\Desktop\agent开发项目\novel_ai_agent
.\.venv\Scripts\Activate.ps1
pyinstaller --noconfirm --clean --windowed --onefile --name NovelAIAgent --add-data "knowledge.txt;." main.py
```

打包完成后，exe 位于：

```text
dist\NovelAIAgent.exe
```

注意事项：

- Windows 的 `--add-data` 分隔符是英文分号 `;`，macOS/Linux 是冒号 `:`。
- 不要把 exe 放到 `Program Files` 等无写入权限目录，否则 `history/`、`generated_images/` 可能无法创建。
- 如果 PyInstaller 漏收 PyQt6 插件，可追加：`--collect-all PyQt6 --collect-all PIL`。
- API Key 不会自动保存到磁盘；如需持久保存，可后续增加本地加密配置文件。

## 导入知识库

点击左侧【本地工具箱】里的【导入知识库】，选择 `.txt` 或 `.md` 文件。
应用会把文件复制到：

```text
knowledge\
```

检索时会读取：

```text
knowledge.txt
knowledge\*.txt
knowledge\*.md
```

## 导入本地 Skill

点击【导入 Skill】，选择你写好的 `.py` 文件。应用会把它复制到：

```text
local_skills\
```

Skill 文件需要提供：

```python
SKILL_NAME = "my_skill"
SKILL_DESCRIPTION = "这个 Skill 的用途说明"

PARAMETERS = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "查询内容"}
    },
    "required": ["query"],
}

def run(query: str) -> str:
    return "Skill 返回给 Agent 的内容"
```

可参考项目里的 `skill_template.py`。

安全提醒：本地 Skill 是 Python 代码，导入后会在本机执行，请只导入可信文件。
