# 小说 AI Agent

这是一个在本地电脑运行的小说写作助手。它可以帮你生成小说大纲、续写章节、润色文本、检查错别字、检索本地设定集，并生成一张本地 Mock 插图。

所有程序文件、历史章节、知识库和生成图片默认都保存在你自己的电脑里。本项目不会自动把 API Key 写入磁盘。

## 普通用户怎么用

### 方式一：直接运行本地安装包

如果你拿到的是已经打包好的版本，请双击运行：

```text
dist\NovelAIAgent.exe
```

这是本地桌面程序，没有网页运行地址。打开后在顶部填写你的模型 API Key，然后点击“生成大纲”或“生成下一章”即可使用。

常用默认配置：

```text
Base URL: https://api.deepseek.com/v1
Model: deepseek-chat
接口格式: OpenAI 兼容
```

### 方式二：从源码一键安装并运行

适合从 GitHub 下载 ZIP 或 clone 仓库的用户。

1. 安装 Python 3.11 或更高版本。
2. 进入项目目录。
3. 双击 `install_windows.bat` 安装依赖。
4. 双击 `run_desktop.bat` 启动桌面版。

也可以在 PowerShell 里运行：

```powershell
.\install_windows.bat
.\run_desktop.bat
```

## 本地运行地址

桌面版是本地窗口程序，不需要浏览器地址。

如果你想把它作为本地 API 服务使用，请双击：

```text
run_api.bat
```

启动后浏览器打开：

```text
http://127.0.0.1:8000/docs
```

常用本地地址：

```text
API 文档: http://127.0.0.1:8000/docs
健康检查: http://127.0.0.1:8000/health
```

这个 API 只监听本机 `127.0.0.1`，默认不会开放到公网。

## 生成本地安装包

如果你想自己生成 exe，请先安装依赖，然后双击：

```text
package_windows.bat
```

打包完成后，本地安装包会生成在：

```text
dist\NovelAIAgent.exe
```

建议把整个 `dist` 文件夹放在有写入权限的位置，例如桌面或文档目录。不要放进 `Program Files`，否则历史章节和生成图片可能无法保存。

## 数据保存在哪里

程序会在本地目录里保存这些内容：

```text
history\            自动备份的章节和统计报表
knowledge\          导入的设定集文件
generated_images\   生成的本地插图
local_skills\       导入的本地 Skill
dist\               打包后的本地运行目录
```

## 导入知识库

在左侧“本地工具箱”点击“导入知识库”，选择 `.txt` 或 `.md` 文件。

程序会把文件复制到：

```text
knowledge\
```

检索时会读取：

```text
knowledge.txt
knowledge\*.txt
knowledge\*.md
```

导入后可以点击“重建语义索引”，让本地检索效果更好。

## 导入本地 Skill

在左侧“本地工具箱”点击“导入 Skill”，可以导入 `.py`、`.json` 或 `.md` 文件。

Python Skill 会在你的电脑上执行，请只导入可信文件。

Python Skill 示例：

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

也可以参考项目里的 `skill_template.py` 和 `json_skill_template.json`。

## 开发者运行方式

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
```

启动本地 API：

```powershell
uvicorn api_server:app --host 127.0.0.1 --port 8000
```

Docker 本地运行：

```powershell
docker build -t novel-ai-agent .
docker run --rm -p 8000:8000 novel-ai-agent
```

Docker 运行后访问：

```text
http://127.0.0.1:8000/docs
```
