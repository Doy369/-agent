from __future__ import annotations

from typing import Any, TypedDict

from local_tools import local_text_audit, search_local_knowledge


class ReviewGraphState(TypedDict, total=False):
    query: str
    novel_setting: str
    current_text: str
    knowledge_context: str
    audit_report: str


def _load_langgraph() -> tuple[Any, Any]:
    try:
        from langgraph.graph import END, StateGraph
    except ImportError as exc:
        raise RuntimeError(
            "LangGraph 工作流需要 langgraph。请先运行：pip install -r requirements.txt"
        ) from exc
    return END, StateGraph


def build_review_graph() -> Any:
    """Build a small local workflow: retrieve knowledge, then audit text."""
    END, StateGraph = _load_langgraph()

    def retrieve_context(state: ReviewGraphState) -> dict[str, str]:
        query = (
            state.get("query")
            or state.get("novel_setting")
            or state.get("current_text")
            or "主角 设定 世界观"
        )
        return {"knowledge_context": search_local_knowledge(query[:200])}

    def audit_text(state: ReviewGraphState) -> dict[str, str]:
        text = state.get("current_text", "")
        return {"audit_report": local_text_audit(text)}

    graph = StateGraph(ReviewGraphState)
    graph.add_node("retrieve_context", retrieve_context)
    graph.add_node("audit_text", audit_text)
    graph.set_entry_point("retrieve_context")
    graph.add_edge("retrieve_context", "audit_text")
    graph.add_edge("audit_text", END)
    return graph.compile()


def run_review_graph(
    *,
    query: str = "",
    novel_setting: str = "",
    current_text: str = "",
) -> ReviewGraphState:
    """Run the local review graph without calling an LLM."""
    graph = build_review_graph()
    return graph.invoke(
        {
            "query": query,
            "novel_setting": novel_setting,
            "current_text": current_text,
        }
    )
