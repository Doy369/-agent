from __future__ import annotations

import html
import importlib.util
import re
import sys
import time
from pathlib import Path
from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QCloseEvent, QFont, QPixmap, QTextCursor
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from agent import AgentConfig, NovelAgent
from finetune_exporter import DEFAULT_OUTPUT_FILE, export_openai_chat_jsonl
from langchain_integration import describe_langchain_integration
from langgraph_workflow import run_review_graph
from local_tools import (
    KNOWLEDGE_DIR,
    SKILL_DIR,
    archive_chapter,
    ensure_app_dirs,
    import_knowledge_files,
    local_archive_and_stats,
    local_text_audit,
    safe_filename,
    search_local_knowledge,
)
from knowledge_base import get_knowledge_base, rebuild_knowledge_base
from prompt_templates import chapter_prompt, outline_prompt, polish_prompt
from skill_loader import import_skill_files, run_local_skill
from workers import GenerationWorker


class NovelAIAgentWindow(QMainWindow):
    """小说 AI Agent 桌面应用主窗口。"""

    def __init__(self) -> None:
        super().__init__()
        ensure_app_dirs()

        self.agent = NovelAgent()
        self.worker: GenerationWorker | None = None
        self.active_action = ""
        self.pending_polish_range: tuple[int, int] | None = None

        # history_entries 用于右侧列表回看；chapters 用于 Agent 记忆上下文。
        self.history_entries: list[dict[str, Any]] = []
        self.chapters: list[dict[str, str]] = []
        self.current_outline = ""
        self.current_history_entry_index: int | None = None

        self.setWindowTitle("小说 AI Agent")
        self.resize(1480, 920)
        self._build_ui()
        self._connect_signals()
        self._refresh_workspace_status()
        self.statusBar().showMessage("就绪")
        self.append_skill_log("本地工具箱已就绪。")
        self.append_skill_log(self.agent.get_local_skill_summary())
        for error in self.agent.local_skill_errors:
            self.append_skill_log(f"Skill 加载警告：{error}")

    def _build_ui(self) -> None:
        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(14, 14, 14, 10)
        root_layout.setSpacing(12)

        root_layout.addWidget(self._build_config_panel())

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_center_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setSizes([360, 840, 320])
        root_layout.addWidget(splitter, 1)

        self.setCentralWidget(root)
        self._apply_styles()

    def _build_config_panel(self) -> QGroupBox:
        group = QGroupBox("API 配置")
        layout = QGridLayout(group)
        layout.setHorizontalSpacing(10)
        layout.setVerticalSpacing(8)

        self.api_key_edit = QLineEdit()
        self.api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key_edit.setPlaceholderText("sk-... / DeepSeek API Key / Anthropic API Key")

        self.base_url_edit = QLineEdit("https://api.deepseek.com/v1")
        self.base_url_edit.setPlaceholderText("例如：https://api.deepseek.com/v1")

        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        self.model_combo.addItems(
            [
                "deepseek-chat",
                "deepseek-reasoner",
                "gpt-4o-mini",
                "gpt-4.1-mini",
                "claude-3-5-sonnet-latest",
            ]
        )

        self.provider_combo = QComboBox()
        self.provider_combo.addItem("OpenAI 兼容", "openai")
        self.provider_combo.addItem("Anthropic", "anthropic")

        layout.addWidget(QLabel("API Key"), 0, 0)
        layout.addWidget(self.api_key_edit, 0, 1, 1, 3)
        layout.addWidget(QLabel("Base URL"), 1, 0)
        layout.addWidget(self.base_url_edit, 1, 1)
        layout.addWidget(QLabel("模型"), 1, 2)
        layout.addWidget(self.model_combo, 1, 3)
        layout.addWidget(QLabel("接口格式"), 1, 4)
        layout.addWidget(self.provider_combo, 1, 5)

        layout.setColumnStretch(1, 3)
        layout.setColumnStretch(3, 2)
        return group

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        panel.setObjectName("SidePanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        tabs = QTabWidget()
        tabs.setObjectName("LeftTabs")
        tabs.addTab(self._build_creation_tab(), "创作")
        tabs.addTab(self._build_assets_tab(), "资料")
        tabs.addTab(self._build_architecture_tab(), "增强")
        layout.addWidget(tabs, 1)
        return panel

    def _build_creation_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 10, 0, 0)
        layout.setSpacing(12)

        setting_group = QGroupBox("小说设定")
        setting_layout = QVBoxLayout(setting_group)

        self.setting_edit = QTextEdit()
        self.setting_edit.setAcceptRichText(False)
        self.setting_edit.setPlaceholderText("输入小说大纲、背景设定、主角人设；也可写：查看我的设定、检查一下错字。")
        self.setting_edit.setMinimumHeight(360)

        form_row = QHBoxLayout()
        self.style_combo = QComboBox()
        self.style_combo.addItems(["玄幻", "科幻", "悬疑", "都市", "历史", "仙侠", "奇幻", "轻小说"])
        form_row.addWidget(QLabel("风格"))
        form_row.addWidget(self.style_combo, 1)

        self.word_label = QLabel("2500 字")
        self.word_slider = QSlider(Qt.Orientation.Horizontal)
        self.word_slider.setRange(1000, 5000)
        self.word_slider.setSingleStep(250)
        self.word_slider.setPageStep(500)
        self.word_slider.setValue(2500)

        setting_layout.addWidget(self.setting_edit)
        setting_layout.addLayout(form_row)
        setting_layout.addWidget(QLabel("生成字数"))
        setting_layout.addWidget(self.word_slider)
        setting_layout.addWidget(self.word_label)

        layout.addWidget(setting_group, 1)
        layout.addStretch(1)
        return tab

    def _build_assets_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 10, 0, 0)
        layout.setSpacing(12)

        toolbox_group = QGroupBox("本地工具箱")
        toolbox_layout = QVBoxLayout(toolbox_group)
        self.import_knowledge_btn = QPushButton("导入知识库")
        self.import_skill_btn = QPushButton("导入 Skill")
        self.refresh_skill_btn = QPushButton("刷新 Skill")
        self.run_skill_btn = QPushButton("运行本地 Skill")
        self.local_search_btn = QPushButton("检索本地设定")
        self.rebuild_index_btn = QPushButton("重建语义索引")
        self.local_audit_btn = QPushButton("本地错字检查")
        self.local_stats_btn = QPushButton("生成全书统计")
        toolbox_layout.addWidget(self.import_knowledge_btn)
        toolbox_layout.addWidget(self.import_skill_btn)
        toolbox_layout.addWidget(self.refresh_skill_btn)
        toolbox_layout.addWidget(self.run_skill_btn)
        toolbox_layout.addWidget(self.local_search_btn)
        toolbox_layout.addWidget(self.rebuild_index_btn)
        toolbox_layout.addWidget(self.local_audit_btn)
        toolbox_layout.addWidget(self.local_stats_btn)

        layout.addWidget(toolbox_group)
        layout.addStretch(1)
        return tab

    def _build_architecture_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 10, 0, 0)
        layout.setSpacing(12)

        architecture_group = QGroupBox("架构增强")
        architecture_layout = QVBoxLayout(architecture_group)
        self.arch_status_btn = QPushButton("查看能力状态")
        self.prompt_preview_btn = QPushButton("预览 Prompt 模板")
        self.langchain_status_btn = QPushButton("LangChain 适配状态")
        self.langgraph_review_btn = QPushButton("运行 LangGraph 审稿")
        self.finetune_export_btn = QPushButton("导出微调数据")
        self.api_docker_btn = QPushButton("API / Docker 启动说明")
        architecture_layout.addWidget(self.arch_status_btn)
        architecture_layout.addWidget(self.prompt_preview_btn)
        architecture_layout.addWidget(self.langchain_status_btn)
        architecture_layout.addWidget(self.langgraph_review_btn)
        architecture_layout.addWidget(self.finetune_export_btn)
        architecture_layout.addWidget(self.api_docker_btn)

        layout.addWidget(architecture_group)
        layout.addStretch(1)
        return tab

    def _build_center_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        status_bar = QFrame()
        status_bar.setObjectName("WorkspaceStatusBar")
        status_layout = QHBoxLayout(status_bar)
        status_layout.setContentsMargins(10, 8, 10, 8)
        status_layout.setSpacing(8)
        self.outline_state_label = self._build_status_chip("大纲：未生成")
        self.chapter_state_label = self._build_status_chip("章节：0")
        self.body_count_label = self._build_status_chip("正文：0 字")
        self.selection_count_label = self._build_status_chip("选中：0 字")
        status_layout.addWidget(self.outline_state_label)
        status_layout.addWidget(self.chapter_state_label)
        status_layout.addWidget(self.body_count_label)
        status_layout.addWidget(self.selection_count_label)
        status_layout.addStretch(1)

        controls = QFrame()
        controls.setObjectName("ControlBar")
        controls_layout = QHBoxLayout(controls)
        controls_layout.setContentsMargins(10, 10, 10, 10)
        controls_layout.setSpacing(10)

        self.outline_btn = QPushButton("生成大纲")
        self.outline_btn.setObjectName("PrimaryButton")
        self.chapter_btn = QPushButton("生成下一章")
        self.chapter_btn.setObjectName("PrimaryButton")
        self.polish_btn = QPushButton("润色文本")
        self.image_btn = QPushButton("生成本章插图")
        self.stop_btn = QPushButton("中止生成")
        self.stop_btn.setObjectName("DangerButton")
        self.stop_btn.setEnabled(False)

        controls_layout.addWidget(self.outline_btn)
        controls_layout.addWidget(self.chapter_btn)
        controls_layout.addWidget(self.polish_btn)
        controls_layout.addWidget(self.image_btn)
        controls_layout.addStretch(1)
        controls_layout.addWidget(self.stop_btn)

        self.editor = QTextEdit()
        self.editor.setAcceptRichText(True)
        self.editor.setPlaceholderText("生成的大纲和章节会显示在这里，也可以直接手动修改。")
        self.editor.setFont(QFont("Microsoft YaHei UI", 11))

        layout.addWidget(status_bar)
        layout.addWidget(controls)
        layout.addWidget(self.editor, 1)
        return panel

    def _build_status_chip(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("StatusChip")
        label.setMinimumHeight(30)
        return label

    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        action_group = QGroupBox("章节操作")
        action_layout = QVBoxLayout(action_group)
        self.save_btn = QPushButton("保存当前章")
        action_layout.addWidget(self.save_btn)

        right_tabs = QTabWidget()
        right_tabs.setObjectName("RightTabs")

        history_tab = QWidget()
        history_layout = QVBoxLayout(history_tab)
        history_layout.setContentsMargins(0, 10, 0, 0)
        self.history_list = QListWidget()
        history_layout.addWidget(self.history_list, 1)

        log_tab = QWidget()
        console_layout = QVBoxLayout(log_tab)
        console_layout.setContentsMargins(0, 10, 0, 0)
        self.skill_console = QTextEdit()
        self.skill_console.setReadOnly(True)
        console_layout.addWidget(self.skill_console, 1)

        right_tabs.addTab(history_tab, "历史")
        right_tabs.addTab(log_tab, "日志")

        layout.addWidget(action_group)
        layout.addWidget(right_tabs, 1)
        return panel

    def _connect_signals(self) -> None:
        self.word_slider.valueChanged.connect(lambda value: self.word_label.setText(f"{value} 字"))
        self.word_slider.valueChanged.connect(lambda _value: self._refresh_workspace_status())
        self.provider_combo.currentIndexChanged.connect(self._on_provider_changed)
        self.editor.textChanged.connect(self._refresh_workspace_status)
        self.editor.selectionChanged.connect(self._refresh_workspace_status)

        self.outline_btn.clicked.connect(self.generate_outline)
        self.chapter_btn.clicked.connect(self.generate_next_chapter)
        self.polish_btn.clicked.connect(self.polish_text)
        self.image_btn.clicked.connect(self.generate_chapter_image)
        self.stop_btn.clicked.connect(self.stop_generation)
        self.save_btn.clicked.connect(self.save_current_chapter)
        self.history_list.itemClicked.connect(self.load_history_item)

        self.import_knowledge_btn.clicked.connect(self.import_knowledge_from_dialog)
        self.import_skill_btn.clicked.connect(self.import_skill_from_dialog)
        self.refresh_skill_btn.clicked.connect(lambda: self.refresh_local_skills(show_dialog=True))
        self.run_skill_btn.clicked.connect(self.run_selected_local_skill)
        self.local_search_btn.clicked.connect(self.run_local_knowledge_search)
        self.rebuild_index_btn.clicked.connect(self.rebuild_knowledge_index)
        self.local_audit_btn.clicked.connect(self.run_local_text_audit)
        self.local_stats_btn.clicked.connect(self.run_local_stats)
        self.arch_status_btn.clicked.connect(self.show_architecture_status)
        self.prompt_preview_btn.clicked.connect(self.preview_prompt_templates)
        self.langchain_status_btn.clicked.connect(self.show_langchain_status)
        self.langgraph_review_btn.clicked.connect(self.run_langgraph_review)
        self.finetune_export_btn.clicked.connect(self.export_finetune_dataset_from_dialog)
        self.api_docker_btn.clicked.connect(self.show_api_docker_guide)

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background: #f5f7fb;
                color: #1f2937;
                font-family: "Microsoft YaHei UI", "Segoe UI", sans-serif;
                font-size: 13px;
            }
            QGroupBox {
                background: #ffffff;
                border: 1px solid #d8dee9;
                border-radius: 8px;
                margin-top: 18px;
                padding: 12px;
                font-weight: 600;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
                color: #334155;
            }
            QTextEdit, QLineEdit, QComboBox, QListWidget {
                background: #ffffff;
                border: 1px solid #cbd5e1;
                border-radius: 6px;
                padding: 7px;
                selection-background-color: #bfdbfe;
            }
            QTextEdit:focus, QLineEdit:focus, QComboBox:focus, QListWidget:focus {
                border: 1px solid #2563eb;
            }
            QPushButton {
                background: #e2e8f0;
                border: 1px solid #cbd5e1;
                border-radius: 6px;
                padding: 9px 13px;
                font-weight: 600;
            }
            QPushButton:hover {
                background: #dbeafe;
                border-color: #93c5fd;
            }
            QPushButton:disabled {
                background: #edf2f7;
                color: #94a3b8;
                border-color: #e2e8f0;
            }
            QPushButton#PrimaryButton {
                background: #2563eb;
                color: #ffffff;
                border-color: #1d4ed8;
            }
            QPushButton#PrimaryButton:hover {
                background: #1d4ed8;
            }
            QPushButton#DangerButton {
                background: #fee2e2;
                color: #991b1b;
                border-color: #fecaca;
            }
            QPushButton#DangerButton:hover {
                background: #fecaca;
            }
            QFrame#ControlBar {
                background: #ffffff;
                border: 1px solid #d8dee9;
                border-radius: 8px;
            }
            QFrame#WorkspaceStatusBar {
                background: #ffffff;
                border: 1px solid #d8dee9;
                border-radius: 8px;
            }
            QLabel#StatusChip {
                background: #f1f5f9;
                border: 1px solid #d8dee9;
                border-radius: 6px;
                padding: 5px 9px;
                color: #334155;
                font-weight: 600;
            }
            QTabWidget::pane {
                border: 0;
                background: transparent;
            }
            QTabBar::tab {
                background: #e2e8f0;
                color: #334155;
                border: 1px solid #cbd5e1;
                border-bottom: none;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                padding: 8px 14px;
                margin-right: 4px;
                font-weight: 600;
            }
            QTabBar::tab:selected {
                background: #ffffff;
                color: #1d4ed8;
                border-color: #93c5fd;
            }
            QTabBar::tab:hover {
                background: #dbeafe;
            }
            QSlider::groove:horizontal {
                height: 6px;
                background: #dbe3ef;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                width: 18px;
                margin: -6px 0;
                background: #2563eb;
                border-radius: 9px;
            }
            QStatusBar {
                background: #eef2f7;
                color: #334155;
            }
            """
        )

    def _on_provider_changed(self) -> None:
        provider = self.provider_combo.currentData()
        if provider == "anthropic":
            if "deepseek" in self.base_url_edit.text().lower():
                self.base_url_edit.setText("https://api.anthropic.com")
            self.model_combo.setCurrentText("claude-3-5-sonnet-latest")
        else:
            if "anthropic" in self.base_url_edit.text().lower():
                self.base_url_edit.setText("https://api.deepseek.com/v1")
            self.model_combo.setCurrentText("deepseek-chat")

    def _config_from_ui(self) -> AgentConfig:
        return AgentConfig(
            api_key=self.api_key_edit.text().strip(),
            base_url=self.base_url_edit.text().strip(),
            model=self.model_combo.currentText().strip(),
            provider=self.provider_combo.currentData() or "openai",
            max_words=self.word_slider.value(),
        )

    def _base_payload(self) -> dict[str, Any]:
        return {
            "novel_setting": self.setting_edit.toPlainText().strip(),
            "style": self.style_combo.currentText(),
            "outline": self.current_outline,
            "chapters": [dict(item) for item in self.chapters],
            "current_text": self.editor.toPlainText(),
        }

    def generate_outline(self) -> None:
        self._start_worker("outline", self._base_payload())

    def generate_next_chapter(self) -> None:
        self.sync_current_editor_to_history()
        payload = self._base_payload()
        if not payload["outline"] and self.history_entries:
            payload["outline"] = self.current_outline
        self._start_worker("chapter", payload)

    def polish_text(self) -> None:
        cursor = self.editor.textCursor()
        selected_text = cursor.selectedText().replace("\u2029", "\n")
        self.pending_polish_range = None

        if selected_text.strip():
            start = min(cursor.selectionStart(), cursor.selectionEnd())
            end = max(cursor.selectionStart(), cursor.selectionEnd())
            self.pending_polish_range = (start, end)
        else:
            selected_text = self.editor.toPlainText()

        instruction, ok = QInputDialog.getText(
            self,
            "润色文本",
            "输入润色目标：",
            text="增强画面感、节奏和情绪张力",
        )
        if not ok:
            return

        payload = self._base_payload()
        payload["selected_text"] = selected_text
        payload["instruction"] = instruction
        self._start_worker("polish", payload)

    def generate_chapter_image(self) -> None:
        selected = self.editor.textCursor().selectedText().replace("\u2029", "\n").strip()
        source = selected or self.editor.toPlainText().strip() or self.setting_edit.toPlainText().strip()
        prompt = source[:500] or "小说章节高光场景，电影感构图，细节丰富"
        self._start_worker("image", {"prompt": prompt}, requires_api=False)

    def _start_worker(self, action: str, payload: dict[str, Any], requires_api: bool = True) -> None:
        if self.worker and self.worker.isRunning():
            QMessageBox.information(self, "任务进行中", "当前已有生成任务在运行，请先中止或等待完成。")
            return

        config = self._config_from_ui() if requires_api else None
        self.active_action = action
        self.set_busy(True)
        self.append_skill_log(f"启动任务：{self._action_label(action)}")

        self.worker = GenerationWorker(self.agent, action, config, payload, parent=self)
        self.worker.status.connect(self.statusBar().showMessage)
        self.worker.skill_log.connect(self.append_skill_log)
        self.worker.image_ready.connect(self.show_image_dialog)
        self.worker.result_ready.connect(self.handle_worker_result)
        self.worker.error.connect(self.handle_worker_error)
        self.worker.finished.connect(self.handle_worker_finished)
        self.worker.start()

    def stop_generation(self) -> None:
        if self.worker and self.worker.isRunning():
            self.statusBar().showMessage("正在中止生成...")
            self.append_skill_log("用户请求中止当前任务。")
            self.worker.stop()

    def handle_worker_result(self, text: str, metadata: object) -> None:
        action = self.active_action
        content = text.strip()

        if action == "outline":
            self.current_outline = content
            self.editor.setPlainText(content)
            self._add_history_entry("全书大纲", content, kind="outline")
            path = archive_chapter("全书大纲", content, suffix=".md")
            self.append_skill_log(f"已自动备份大纲：{path}")

        elif action == "chapter":
            title = self._extract_title(content, len(self.chapters) + 1)
            self.editor.setPlainText(content)
            self._add_history_entry(title, content, kind="chapter")
            path = archive_chapter(title, content, suffix=".md")
            self.append_skill_log(f"已自动备份章节：{path}")

        elif action == "polish":
            if self.pending_polish_range:
                start, end = self.pending_polish_range
                cursor = self.editor.textCursor()
                cursor.setPosition(start)
                cursor.setPosition(end, QTextCursor.MoveMode.KeepAnchor)
                cursor.insertText(content)
                self.editor.setTextCursor(cursor)
            else:
                self.editor.setPlainText(content)
            self.sync_current_editor_to_history()
            path = archive_chapter(f"润色稿_{time.strftime('%H%M%S')}", self.editor.toPlainText(), suffix=".md")
            self.append_skill_log(f"已自动备份润色稿：{path}")

        elif action == "image":
            self.append_skill_log(content)

        if isinstance(metadata, dict):
            for tool_result in metadata.get("tool_results", []):
                if isinstance(tool_result, dict) and tool_result.get("name") in {
                    "search_local_knowledge",
                    "local_text_audit",
                    "local_archive_and_stats",
                }:
                    self.show_text_dialog(f"工具结果：{tool_result.get('name')}", tool_result.get("content", ""))

        self.statusBar().showMessage("生成完毕")

    def handle_worker_error(self, message: str) -> None:
        self.statusBar().showMessage(message)
        self.append_skill_log(f"错误：{message}")
        if message != "已中止生成。":
            QMessageBox.warning(self, "任务失败", message)

    def handle_worker_finished(self) -> None:
        self.set_busy(False)
        self.worker = None
        self.active_action = ""
        self.pending_polish_range = None

    def set_busy(self, busy: bool) -> None:
        for button in [
            self.outline_btn,
            self.chapter_btn,
            self.polish_btn,
            self.image_btn,
            self.save_btn,
            self.import_knowledge_btn,
            self.import_skill_btn,
            self.refresh_skill_btn,
            self.run_skill_btn,
            self.local_search_btn,
            self.rebuild_index_btn,
            self.local_audit_btn,
            self.local_stats_btn,
            self.arch_status_btn,
            self.prompt_preview_btn,
            self.langchain_status_btn,
            self.langgraph_review_btn,
            self.finetune_export_btn,
            self.api_docker_btn,
        ]:
            button.setEnabled(not busy)
        self.stop_btn.setEnabled(busy)

    def _refresh_workspace_status(self) -> None:
        if not hasattr(self, "outline_state_label"):
            return

        text = self.editor.toPlainText()
        selected_text = self.editor.textCursor().selectedText().replace("\u2029", "\n")
        self.outline_state_label.setText("大纲：已生成" if self.current_outline.strip() else "大纲：未生成")
        self.chapter_state_label.setText(f"章节：{len(self.chapters)}")
        self.body_count_label.setText(f"正文：约 {self._count_text_units(text)} / {self.word_slider.value()} 字")
        self.selection_count_label.setText(f"选中：约 {self._count_text_units(selected_text)} 字")

    def _count_text_units(self, text: str) -> int:
        cjk_count = len(re.findall(r"[\u4e00-\u9fff]", text or ""))
        latin_words = len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", text or ""))
        return cjk_count + latin_words

    def save_current_chapter(self) -> None:
        content = self.editor.toPlainText().strip()
        if not content:
            QMessageBox.information(self, "没有内容", "当前编辑区没有可保存的内容。")
            return

        title = self._extract_title(content, len(self.chapters) + 1)
        default_path = str(Path.cwd() / f"{safe_filename(title)}.md")
        path, _ = QFileDialog.getSaveFileName(
            self,
            "保存当前章",
            default_path,
            "Markdown 文件 (*.md);;文本文件 (*.txt)",
        )
        if not path:
            return

        file_path = Path(path)
        file_path.write_text(content + "\n", encoding="utf-8")
        self.statusBar().showMessage(f"已保存：{file_path}")
        self.append_skill_log(f"用户手动保存：{file_path}")

    def _add_history_entry(self, title: str, content: str, kind: str) -> None:
        entry: dict[str, Any] = {"title": title, "content": content, "kind": kind}
        if kind == "chapter":
            chapter_index = len(self.chapters)
            self.chapters.append({"title": title, "content": content})
            entry["chapter_index"] = chapter_index
            label = f"{chapter_index + 1:02d}. {title}"
        elif kind == "outline":
            label = f"大纲：{title}"
        else:
            label = title

        self.history_entries.append(entry)
        item = QListWidgetItem(label)
        item.setData(Qt.ItemDataRole.UserRole, len(self.history_entries) - 1)
        self.history_list.addItem(item)
        self.history_list.setCurrentItem(item)
        self.current_history_entry_index = len(self.history_entries) - 1
        self._refresh_workspace_status()

    def load_history_item(self, item: QListWidgetItem) -> None:
        self.sync_current_editor_to_history()
        entry_index = item.data(Qt.ItemDataRole.UserRole)
        if entry_index is None:
            return
        entry = self.history_entries[int(entry_index)]
        self.current_history_entry_index = int(entry_index)
        self.editor.setPlainText(entry.get("content", ""))
        if entry.get("kind") == "outline":
            self.current_outline = entry.get("content", "")
        self.statusBar().showMessage(f"已切换：{entry.get('title', '未命名')}")
        self._refresh_workspace_status()

    def sync_current_editor_to_history(self) -> None:
        """把用户在富文本编辑区的手动修改同步回 Agent 记忆。"""
        if self.current_history_entry_index is None:
            return
        if not (0 <= self.current_history_entry_index < len(self.history_entries)):
            return

        content = self.editor.toPlainText()
        entry = self.history_entries[self.current_history_entry_index]
        entry["content"] = content

        if entry.get("kind") == "outline":
            self.current_outline = content
        elif entry.get("kind") == "chapter":
            chapter_index = entry.get("chapter_index")
            if isinstance(chapter_index, int) and 0 <= chapter_index < len(self.chapters):
                self.chapters[chapter_index]["content"] = content
        self._refresh_workspace_status()

    def import_knowledge_from_dialog(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "导入知识库文件",
            str(KNOWLEDGE_DIR),
            "知识库文件 (*.txt *.md);;所有文件 (*)",
        )
        if not paths:
            return

        imported = import_knowledge_files(paths)
        if not imported:
            QMessageBox.information(self, "未导入文件", "请选择 .txt 或 .md 格式的知识库文件。")
            return

        lines = ["已导入知识库文件：", ""]
        lines.extend(str(path) for path in imported)
        lines.extend(["", "这些文件已进入本地知识库，后续可通过【检索本地设定】或 search_local_knowledge 工具读取。"])
        message = "\n".join(lines)
        self.append_skill_log(f"已导入 {len(imported)} 个知识库文件到：{KNOWLEDGE_DIR}")
        # 导入后自动重建语义索引
        self.append_skill_log("正在重建语义索引...")
        try:
            count = rebuild_knowledge_base()
            self.append_skill_log(f"语义索引已更新（{count} 个文本块）。")
        except Exception as exc:
            self.append_skill_log(f"语义索引更新失败：{exc}（下次检索时会自动重试）")
        self.show_text_dialog("导入知识库完成", message)

    def import_skill_from_dialog(self) -> None:
        reply = QMessageBox.warning(
            self,
            "导入 Skill 安全提示",
            "Python Skill 会在本机执行，请只导入你信任的 .py 文件。JSON Skill 和 Markdown Skill 可安全导入。请只导入你信任的文件。",
            QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
        )
        if reply != QMessageBox.StandardButton.Ok:
            return

        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "导入 Skill 文件",
            str(SKILL_DIR),
            "Skill 文件 (*.py *.json *.md);;Python Skill (*.py);;JSON Skill (*.json);;Markdown Skill (*.md);;所有文件 (*)",
        )
        if not paths:
            return

        imported = import_skill_files(paths)
        if not imported:
            QMessageBox.information(self, "未导入 Skill", "请选择有效的 .py Skill 文件。")
            return

        self.append_skill_log(f"已导入 {len(imported)} 个 Skill 文件到：{SKILL_DIR}")
        self.refresh_local_skills(show_dialog=True, imported_paths=imported)

    def refresh_local_skills(self, show_dialog: bool = True, imported_paths: list[Path] | None = None) -> None:
        errors = self.agent.reload_local_skills()
        summary = self.agent.get_local_skill_summary()

        lines: list[str] = []
        if imported_paths:
            lines.append("本次导入文件：")
            lines.extend(str(path) for path in imported_paths)
            lines.append("")

        lines.append(summary)
        if errors:
            lines.extend(["", "加载警告："])
            lines.extend(errors)

        self.append_skill_log(summary.replace("\n", " | "))
        for error in errors:
            self.append_skill_log(f"Skill 加载警告：{error}")

        if show_dialog:
            self.show_text_dialog("本地 Skill 状态", "\n".join(lines))

    def run_selected_local_skill(self) -> None:
        errors = self.agent.reload_local_skills()
        for error in errors:
            self.append_skill_log(f"Skill 加载警告：{error}")

        skills = list(self.agent.local_skills.values())
        if not skills:
            QMessageBox.information(self, "没有可运行的 Skill", "当前未加载本地 Skill。")
            return

        labels = [f"{skill.name} - {skill.description[:48]}" for skill in skills]
        choice, ok = QInputDialog.getItem(
            self,
            "运行本地 Skill",
            "选择要运行的 Skill：",
            labels,
            0,
            False,
        )
        if not ok:
            return

        skill = skills[labels.index(choice)]
        args = self._build_local_skill_args(skill)
        if args is None:
            return

        self.append_skill_log(f"手动运行本地 Skill：{skill.name}")
        try:
            result = run_local_skill(skill, args)
        except Exception as exc:
            self.append_skill_log(f"Skill 执行失败：{skill.name} | {exc}")
            QMessageBox.warning(self, "Skill 执行失败", str(exc))
            return

        self.show_text_dialog(f"Skill 结果：{skill.name}", result)

    def _build_local_skill_args(self, skill: Any) -> dict[str, Any] | None:
        parameters = skill.parameters if isinstance(skill.parameters, dict) else {}
        properties = parameters.get("properties") if isinstance(parameters.get("properties"), dict) else {}
        required = set(parameters.get("required") or [])

        selected_text = self.editor.textCursor().selectedText().replace("\u2029", "\n").strip()
        current_text = self.editor.toPlainText().strip()
        setting = self.setting_edit.toPlainText().strip()
        source_text = selected_text or current_text or setting
        memory_context = self.agent._build_memory_context(setting, self.current_outline, self.chapters)

        args: dict[str, Any] = {}
        for name in properties:
            lowered = name.lower()
            if lowered in {"text", "selected_text", "chapter_text", "content"}:
                args[name] = source_text
            elif lowered in {"current_text", "current_chapter"}:
                args[name] = current_text or source_text
            elif lowered in {"previous_context", "context", "memory_context"}:
                args[name] = memory_context
            elif lowered in {"query", "topic", "instruction"}:
                args[name] = selected_text or setting or current_text[:200]
            elif lowered in {"novel_setting", "setting"}:
                args[name] = setting
            elif lowered == "style":
                args[name] = self.style_combo.currentText()
            elif name in required:
                value, ok = QInputDialog.getText(self, "Skill 参数", f"请输入参数 {name}：")
                if not ok:
                    return None
                args[name] = value

        return args

    def run_local_knowledge_search(self) -> None:
        default = self.editor.textCursor().selectedText().replace("\u2029", "\n").strip()
        query, ok = QInputDialog.getText(self, "检索本地设定", "输入关键词：", text=default or "主角能力")
        if not ok:
            return
        self.append_skill_log(f"手动触发【本地设定集检索】：{query}")
        result = search_local_knowledge(query)
        self.show_text_dialog("本地设定检索结果", result)

    def run_local_text_audit(self) -> None:
        selected = self.editor.textCursor().selectedText().replace("\u2029", "\n").strip()
        target = selected or self.editor.toPlainText()
        self.append_skill_log("手动触发【本地错字检查】。")
        result = local_text_audit(target)
        self.show_text_dialog("本地错字检查报告", result)

    def run_local_stats(self) -> None:
        self.append_skill_log("手动触发【本地归档统计】。")
        result = local_archive_and_stats()
        self.show_text_dialog("全书统计", result)

    def rebuild_knowledge_index(self) -> None:
        """手动重建语义索引。"""
        self.append_skill_log("手动触发【重建语义索引】...")
        try:
            count = rebuild_knowledge_base()
            kb = get_knowledge_base()
            stats = kb.get_stats()
            msg = (
                f"语义索引重建完成！\n\n"
                f"文本块数: {count}\n"
                f"存储位置: {stats['persist_dir']}\n"
                f"模型: {stats['model']}\n\n"
                f"现在搜索「检索本地设定」将使用语义搜索。"
            )
            self.append_skill_log(f"索引重建完成：{count} 个块")
            self.show_text_dialog("重建语义索引", msg)
        except Exception as exc:
            self.append_skill_log(f"索引重建失败：{exc}")
            QMessageBox.warning(self, "重建失败", str(exc))

    def show_architecture_status(self) -> None:
        kb = get_knowledge_base()
        stats = kb.get_stats()
        lines = [
            "当前架构能力状态",
            "",
            "已接入：",
            "- Prompt 工程：已拆分到 prompt_templates.py",
            "- RAG / 向量检索：已接入 knowledge_base.py",
            "- LangChain：已提供 langchain_integration.py 适配层",
            "- LangGraph：已提供 langgraph_workflow.py 本地审稿流程",
            "- FastAPI：已提供 api_server.py HTTP 服务",
            "- Docker：已提供 Dockerfile 和 .dockerignore",
            "- 微调数据：已提供 finetune_exporter.py JSONL 导出器",
            "",
            "依赖状态：",
            f"- langchain_core: {self._dependency_label('langchain_core')}",
            f"- langgraph: {self._dependency_label('langgraph')}",
            f"- fastapi: {self._dependency_label('fastapi')}",
            f"- uvicorn: {self._dependency_label('uvicorn')}",
            "",
            "知识库状态：",
            f"- 文本块数：{stats['total_chunks']}",
            f"- Embedding 模型：{stats['model']}",
            f"- 语义检索依赖：{'可用' if stats['st_available'] else '未安装 sentence-transformers'}",
            f"- 向量存储目录：{stats['persist_dir']}",
        ]
        self.append_skill_log("已查看架构能力状态。")
        self.show_text_dialog("架构增强状态", "\n".join(lines))

    def preview_prompt_templates(self) -> None:
        payload = self._base_payload()
        selected_text = self.editor.textCursor().selectedText().replace("\u2029", "\n").strip()
        selected_text = selected_text or payload["current_text"] or "这里会放入当前选中的文本或编辑区全文。"

        outline = outline_prompt(
            novel_setting=payload["novel_setting"],
            style=payload["style"],
            auto_context="【自动上下文会在实际生成时注入到这里】",
        )
        memory_context = self.agent._build_memory_context(
            payload["novel_setting"],
            payload["outline"],
            payload["chapters"],
        )
        chapter = chapter_prompt(
            chapter_index=len(self.chapters) + 1,
            style=payload["style"],
            max_words=self.word_slider.value(),
            memory_context=memory_context,
            auto_context="【自动上下文会在实际生成时注入到这里】",
        )
        polish = polish_prompt(
            selected_text=selected_text,
            style=payload["style"],
            instruction="增强画面感、节奏和情绪张力",
            auto_context="【自动上下文会在实际生成时注入到这里】",
        )

        text = "\n\n".join(
            [
                "【System Prompt】",
                outline.system,
                "【生成大纲 Prompt】",
                outline.user,
                "【生成下一章 Prompt】",
                chapter.user,
                "【润色 Prompt】",
                polish.user,
            ]
        )
        self.append_skill_log("已预览 Prompt 模板。")
        self.show_text_dialog("Prompt 模板预览", text)

    def show_langchain_status(self) -> None:
        lines = [
            describe_langchain_integration(),
            "",
            "依赖状态：",
            f"- langchain_core: {self._dependency_label('langchain_core')}",
            f"- langchain: {self._dependency_label('langchain')}",
            f"- langchain_community: {self._dependency_label('langchain_community')}",
            f"- langchain_openai: {self._dependency_label('langchain_openai')}",
            "",
            "使用方式：",
            "安装依赖后，可在代码中调用 build_prompt_templates() 获取 ChatPromptTemplate，",
            "调用 build_langchain_tools() 获取可交给 LangChain Agent 的本地工具列表。",
        ]
        self.append_skill_log("已查看 LangChain 适配状态。")
        self.show_text_dialog("LangChain 适配状态", "\n".join(lines))

    def run_langgraph_review(self) -> None:
        current_text = self.editor.toPlainText().strip()
        setting = self.setting_edit.toPlainText().strip()
        if not current_text and not setting:
            QMessageBox.information(self, "缺少内容", "请先在小说设定或正文编辑区输入内容。")
            return

        self.append_skill_log("手动触发【LangGraph 审稿流程】。")
        try:
            result = run_review_graph(
                query=setting[:200] or current_text[:200],
                novel_setting=setting,
                current_text=current_text,
            )
        except Exception as exc:
            self.append_skill_log(f"LangGraph 审稿流程不可用：{exc}")
            QMessageBox.warning(self, "LangGraph 不可用", str(exc))
            return

        lines = [
            "LangGraph 审稿流程结果",
            "",
            "【检索上下文】",
            result.get("knowledge_context", ""),
            "",
            "【本地审稿报告】",
            result.get("audit_report", ""),
        ]
        self.show_text_dialog("LangGraph 审稿", "\n".join(lines))

    def export_finetune_dataset_from_dialog(self) -> None:
        default_path = str(DEFAULT_OUTPUT_FILE)
        path, _ = QFileDialog.getSaveFileName(
            self,
            "导出微调数据集",
            default_path,
            "JSONL 文件 (*.jsonl);;所有文件 (*)",
        )
        if not path:
            return

        try:
            output_path, count = export_openai_chat_jsonl(output_path=Path(path))
        except Exception as exc:
            self.append_skill_log(f"微调数据导出失败：{exc}")
            QMessageBox.warning(self, "导出失败", str(exc))
            return

        msg = (
            f"已导出 {count} 条微调样本。\n\n"
            f"输出文件：{output_path}\n\n"
            "格式：OpenAI chat fine-tuning JSONL，每行包含 system/user/assistant messages。"
        )
        self.append_skill_log(f"已导出微调数据集：{output_path}（{count} 条）")
        self.show_text_dialog("微调数据导出完成", msg)

    def show_api_docker_guide(self) -> None:
        lines = [
            "FastAPI 本地启动",
            "",
            "1. 安装服务依赖：",
            r".venv\Scripts\python.exe -m pip install -r requirements-api.txt",
            "",
            "2. 启动 API 服务：",
            r".venv\Scripts\uvicorn.exe api_server:app --reload --host 127.0.0.1 --port 8000",
            "",
            "3. 打开接口文档：",
            "http://127.0.0.1:8000/docs",
            "",
            "Docker 启动",
            "",
            "docker build -t novel-ai-agent-api .",
            "docker run --rm -p 8000:8000 novel-ai-agent-api",
            "",
            "已暴露的主要接口：",
            "- GET /health",
            "- POST /generate-outline",
            "- POST /generate-chapter",
            "- POST /polish",
            "- POST /search-knowledge",
            "- POST /audit-text",
            "- POST /rebuild-knowledge-index",
            "- POST /generate-image",
            "- POST /export-finetune-dataset",
        ]
        self.append_skill_log("已查看 API / Docker 启动说明。")
        self.show_text_dialog("API / Docker 启动说明", "\n".join(lines))

    def _dependency_label(self, module_name: str) -> str:
        return "已安装" if importlib.util.find_spec(module_name) else "未安装"

    def show_text_dialog(self, title: str, text: str) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        dialog.resize(760, 560)
        layout = QVBoxLayout(dialog)

        viewer = QTextEdit()
        viewer.setReadOnly(True)
        viewer.setPlainText(text)
        layout.addWidget(viewer, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.exec()

    def show_image_dialog(self, image_path: str) -> None:
        path = Path(image_path)
        if not path.exists():
            QMessageBox.warning(self, "插图不存在", f"未找到图片：{path}")
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("本章插图")
        dialog.resize(920, 720)
        layout = QVBoxLayout(dialog)

        label = QLabel()
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            label.setText(f"图片加载失败：{path}")
        else:
            label.setPixmap(
                pixmap.scaled(
                    860,
                    620,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(label)
        layout.addWidget(scroll, 1)

        path_label = QLabel(str(path))
        path_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(path_label)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.exec()

    def append_skill_log(self, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        safe_message = html.escape(message)
        html_line = (
            f'<p style="margin:6px 0; padding:6px; background:#fff7ed; '
            f'border-left:4px solid #f97316;">'
            f'<b style="color:#9a3412;">{timestamp}</b> {safe_message}</p>'
        )
        self.skill_console.append(html_line)
        bar = self.skill_console.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _extract_title(self, content: str, index: int) -> str:
        for line in content.splitlines():
            title = line.strip().lstrip("#").strip()
            if title:
                return title[:60]
        return f"第 {index} 章"

    def _action_label(self, action: str) -> str:
        return {
            "outline": "生成大纲",
            "chapter": "生成下一章",
            "polish": "润色文本",
            "image": "生成本章插图",
        }.get(action, action)

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.worker and self.worker.isRunning():
            reply = QMessageBox.question(
                self,
                "任务仍在运行",
                "当前还有生成任务在运行，是否中止并退出？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self.worker.stop()
            self.worker.wait(1500)
        event.accept()


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("小说 AI Agent")
    window = NovelAIAgentWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
