"""本地 Skill 模板。

使用方法：
1. 复制本文件并改名，例如 my_research_skill.py。
2. 修改 SKILL_NAME、SKILL_DESCRIPTION、PARAMETERS 和 run()。
3. 在应用左侧【本地工具箱】点击【导入 Skill】，选择你的 .py 文件。
"""

SKILL_NAME = "my_research_skill"
SKILL_DESCRIPTION = "示例 Skill：根据 query 返回一段自定义资料。"

PARAMETERS = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "查询关键词或任务描述。"}
    },
    "required": ["query"],
}


def run(query: str) -> str:
    """这里写你的本地技能逻辑，可以读文件、查本地数据库或调用内网服务。"""
    return f"示例 Skill 收到查询：{query}"
