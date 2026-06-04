from __future__ import annotations

from typing import Any

from PyQt6.QtCore import QThread, pyqtSignal

from agent import AgentConfig, NovelAgent


class GenerationWorker(QThread):
    """耗时任务线程：所有 API 请求和可能变慢的工具调用都放在这里执行。"""

    result_ready = pyqtSignal(str, object)
    error = pyqtSignal(str)
    status = pyqtSignal(str)
    skill_log = pyqtSignal(str)
    image_ready = pyqtSignal(str)

    def __init__(
        self,
        agent: NovelAgent,
        action: str,
        config: AgentConfig | None,
        payload: dict[str, Any],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.agent = agent
        self.action = action
        self.config = config
        self.payload = payload
        self._stopped = False

    def stop(self) -> None:
        """由“中止生成”按钮调用。"""
        self._stopped = True
        self.requestInterruption()
        self.agent.cancel_current_request()

    def run(self) -> None:
        try:
            self.status.emit("AI 正在思考中...")

            if self.action == "outline":
                assert self.config is not None
                result = self.agent.generate_outline(
                    self.config,
                    novel_setting=self.payload.get("novel_setting", ""),
                    style=self.payload.get("style", "玄幻"),
                    current_text=self.payload.get("current_text", ""),
                    log_callback=self.skill_log.emit,
                )

            elif self.action == "chapter":
                assert self.config is not None
                result = self.agent.generate_next_chapter(
                    self.config,
                    novel_setting=self.payload.get("novel_setting", ""),
                    style=self.payload.get("style", "玄幻"),
                    outline=self.payload.get("outline", ""),
                    chapters=self.payload.get("chapters", []),
                    current_text=self.payload.get("current_text", ""),
                    log_callback=self.skill_log.emit,
                )

            elif self.action == "polish":
                assert self.config is not None
                result = self.agent.polish_text(
                    self.config,
                    selected_text=self.payload.get("selected_text", ""),
                    style=self.payload.get("style", "玄幻"),
                    instruction=self.payload.get("instruction", ""),
                    log_callback=self.skill_log.emit,
                )

            elif self.action == "image":
                prompt = self.payload.get("prompt", "")
                result = self.agent.generate_image_for_chapter(prompt, log_callback=self.skill_log.emit)

            else:
                raise ValueError(f"未知任务类型：{self.action}")

            if self._stopped or self.isInterruptionRequested():
                raise RuntimeError("用户已中止生成。")

            for image_path in result.metadata.get("image_paths", []):
                self.image_ready.emit(str(image_path))

            if image_path := result.metadata.get("image_path"):
                self.image_ready.emit(str(image_path))

            self.result_ready.emit(result.content, result.metadata)
            self.status.emit("生成完毕")

        except Exception as exc:
            if self._stopped or self.isInterruptionRequested():
                self.error.emit("已中止生成。")
            else:
                self.error.emit(str(exc))
