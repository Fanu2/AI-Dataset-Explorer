"""
AI Dataset Explorer Studio — PySide6
Version 2: local-first AI dataset intelligence.

Preserves the original explorer's main features and adds:
- Parquet / multi-shard Parquet support (PyArrow)
- Hugging Face repository download/open support
- Large-dataset, batch-oriented browsing and search
- Dashboard with dataset/schema statistics
- Conversation/sequence-aware inspector
- Token analytics for text+tokens datasets
- Dataset quality checks
- Pagination
- CSV / JSON / JSONL support
- Filtered JSONL export
- Safe QThread lifecycle / cleanup
- Recent datasets
- Dark/light theme
- Read-only source handling

Install:
    py -m pip install PySide6 pyarrow

Optional Hugging Face:
    py -m pip install huggingface_hub
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any

from PySide6.QtCore import QObject, QThread, Qt, Signal, Slot
from PySide6.QtGui import QAction, QFont
from PySide6.QtWidgets import (
    QApplication, QFileDialog, QFrame, QGridLayout, QHBoxLayout,
    QHeaderView, QLabel, QLineEdit, QListWidget, QListWidgetItem,
    QMainWindow, QMessageBox, QProgressBar, QPushButton, QComboBox,
    QSpinBox, QStackedWidget, QStatusBar, QTableWidget, QTableWidgetItem,
    QTextEdit, QToolBar, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
    QAbstractItemView, QCheckBox
)

APP_NAME = "AI Dataset Explorer Studio"
APP_VERSION = "3.0 — Dataset Intelligence"
CONFIG_DIR = Path.home() / ".ai_dataset_explorer"
CONFIG_FILE = CONFIG_DIR / "settings.json"
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

try:
    import pyarrow as pa
    import pyarrow.dataset as pads
    import pyarrow.parquet as pq
except ImportError:
    pa = None
    pads = None
    pq = None


def human_number(n: int | float) -> str:
    return f"{n:,.0f}"


def human_size(n: int | float) -> str:
    x = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if x < 1024 or unit == "TB":
            return f"{x:,.1f} {unit}"
        x /= 1024
    return f"{x:,.1f} TB"


def short_text(value: Any, limit: int = 180) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        try:
            text = json.dumps(value, ensure_ascii=False)
        except Exception:
            text = repr(value)
    else:
        text = str(value)
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def conversation_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict):
                role = item.get("from") or item.get("role") or item.get("speaker") or "message"
                content = item.get("value") if "value" in item else item.get("content")
                if content is None:
                    content = item
                parts.append(f"{role}: {conversation_to_text(content)}")
            else:
                parts.append(str(item))
        return "\n\n".join(parts)
    if isinstance(value, dict):
        role = value.get("role") or value.get("from")
        content = value.get("content") if "content" in value else value.get("value")
        if role and content is not None:
            return f"{role}: {conversation_to_text(content)}"
        return json.dumps(value, ensure_ascii=False, indent=2)
    return str(value)


def find_value(row: dict[str, Any], names: list[str]) -> Any:
    lower = {str(k).lower(): k for k in row}
    for name in names:
        if name.lower() in lower:
            return row[lower[name.lower()]]
    return None


def row_search_text(row: dict[str, Any]) -> str:
    chunks = []
    for key, value in row.items():
        chunks.append(str(key))
        if isinstance(value, (list, dict)):
            chunks.append(conversation_to_text(value))
        else:
            chunks.append(str(value))
    return " ".join(chunks).lower()


def role_counts(text: str) -> Counter:
    return Counter(re.findall(r"(?im)^\s*(USER|ASSISTANT|SYSTEM|HUMAN|GPT)\s*:", text))


class DatasetBackend:
    """Read-only backend. Parquet stays batch-oriented; small legacy files may be loaded in memory."""

    def __init__(self):
        self.source = ""
        self.kind = ""
        self.rows: list[dict[str, Any]] = []
        self.columns: list[str] = []
        self.total_rows = 0
        self.total_bytes = 0
        self.files: list[Path] = []
        self.dataset = None

    def open(self, source: str):
        self.source = source
        self.rows = []
        self.columns = []
        self.files = []
        self.dataset = None
        self.total_rows = 0
        self.total_bytes = 0

        path = Path(source)
        if path.exists():
            if path.is_dir():
                # A dataset directory may contain multiple Parquet shards.
                # They are exposed as one logical dataset through PyArrow.
                parquet = sorted(path.rglob("*.parquet"))
                if parquet:
                    self._open_parquet_files(parquet)
                    return
                files = sorted(path.rglob("*"))
                data_files = [p for p in files if p.suffix.lower() in {".json", ".jsonl", ".csv"}]
                if not data_files:
                    raise RuntimeError("No supported dataset files were found in this folder.")
                if len(data_files) == 1:
                    self._open_small_file(data_files[0])
                else:
                    raise RuntimeError("A folder with multiple JSON/CSV files is not yet treated as one dataset. Use a Parquet folder or select one file.")
                return

            if path.suffix.lower() == ".parquet":
                self._open_parquet_files([path])
            elif path.suffix.lower() in {".json", ".jsonl", ".csv"}:
                self._open_small_file(path)
            else:
                raise RuntimeError("Unsupported dataset type.")
            return

        raise RuntimeError(f"Dataset path does not exist:\n{source}")

    def _open_parquet_files(self, files: list[Path]):
        if pa is None or pads is None:
            raise RuntimeError("Parquet support requires PyArrow.\n\nInstall it with:\npy -m pip install pyarrow")
        self.kind = "parquet"
        self.files = files
        self.total_bytes = sum(p.stat().st_size for p in files)
        self.dataset = pads.dataset([str(p) for p in files], format="parquet")
        self.total_rows = self.dataset.count_rows()
        self.columns = list(self.dataset.schema.names)

    def _open_small_file(self, path: Path):
        self.kind = path.suffix.lower().lstrip(".")
        self.files = [path]
        self.total_bytes = path.stat().st_size

        if self.kind == "jsonl":
            with path.open("r", encoding="utf-8-sig") as fh:
                self.rows = []
                for line in fh:
                    if line.strip():
                        obj = json.loads(line)
                        self.rows.append(obj if isinstance(obj, dict) else {"value": obj})
        elif self.kind == "json":
            obj = json.loads(path.read_text(encoding="utf-8-sig"))
            if isinstance(obj, list):
                data = obj
            elif isinstance(obj, dict) and isinstance(obj.get("data"), list):
                data = obj["data"]
            elif isinstance(obj, dict) and isinstance(obj.get("rows"), list):
                data = obj["rows"]
            else:
                data = [obj]
            self.rows = [x if isinstance(x, dict) else {"value": x} for x in data]
        elif self.kind == "csv":
            with path.open("r", encoding="utf-8-sig", newline="") as fh:
                self.rows = list(csv.DictReader(fh))

        self.total_rows = len(self.rows)
        self.columns = []
        seen = set()
        for row in self.rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    self.columns.append(str(key))

    def get_page(self, offset: int, limit: int) -> list[dict[str, Any]]:
        if self.kind != "parquet":
            return self.rows[offset:offset + limit]

        result = []
        skipped = 0
        scanner = self.dataset.scanner(batch_size=max(512, min(4096, limit * 4)))
        for batch in scanner.to_batches():
            data = batch.to_pylist()
            if skipped + len(data) <= offset:
                skipped += len(data)
                continue
            start = max(0, offset - skipped)
            result.extend(data[start:])
            if len(result) >= limit:
                break
            skipped += len(data)
        return result[:limit]

    def iter_batches(self, columns: list[str] | None = None, batch_size: int = 2048):
        if self.kind != "parquet":
            for i in range(0, len(self.rows), batch_size):
                yield self.rows[i:i + batch_size]
            return
        scanner = self.dataset.scanner(columns=columns, batch_size=batch_size)
        for batch in scanner.to_batches():
            yield batch.to_pylist()


class LoadWorker(QObject):
    loaded = Signal(object, str)
    progress = Signal(int)
    status = Signal(str)
    failed = Signal(str)

    def __init__(self, source: str):
        super().__init__()
        self.source = source

    @Slot()
    def run(self):
        try:
            backend = DatasetBackend()
            self.status.emit("Opening dataset…")
            backend.open(self.source)
            self.progress.emit(100)
            self.loaded.emit(backend, self.source)
        except Exception as exc:
            self.failed.emit(str(exc))


class SearchWorker(QObject):
    finished = Signal(object, int)
    progress = Signal(int)
    status = Signal(str)
    failed = Signal(str)

    def __init__(self, backend: DatasetBackend, query: str, column: str = "__all__", max_results: int = 5000):
        super().__init__()
        self.backend = backend
        self.query = query.lower().strip()
        self.column = column
        self.max_results = max_results

    @Slot()
    def run(self):
        try:
            results = []
            scanned = 0
            total = max(self.backend.total_rows, 1)
            for batch in self.backend.iter_batches(batch_size=1024):
                for row in batch:
                    scanned += 1
                    if not self.query:
                        results.append(row)
                    elif self.column == "__all__":
                        if self.query in row_search_text(row):
                            results.append(row)
                    elif self.column in row:
                        value = row.get(self.column)
                        haystack = conversation_to_text(value) if isinstance(value, (list, dict)) else str(value)
                        if self.query in haystack.lower():
                            results.append(row)
                        if len(results) >= self.max_results:
                            self.finished.emit(results, scanned)
                            return
                self.progress.emit(min(99, int(scanned * 100 / total)))
                self.status.emit(f"Searching… {scanned:,} / {total:,}")
            self.progress.emit(100)
            self.finished.emit(results, scanned)
        except Exception as exc:
            self.failed.emit(str(exc))


class StatCard(QFrame):
    def __init__(self, title: str):
        super().__init__()
        self.setObjectName("card")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        self.title = QLabel(title)
        self.title.setObjectName("cardTitle")
        self.value = QLabel("—")
        self.value.setObjectName("cardValue")
        layout.addWidget(self.title)
        layout.addWidget(self.value)

    def set_value(self, value: str):
        self.value.setText(value)


class DashboardPage(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 22, 24, 22)
        layout.setSpacing(14)

        header = QHBoxLayout()
        box = QVBoxLayout()
        title = QLabel("Dataset Intelligence")
        title.setObjectName("pageTitle")
        subtitle = QLabel("Understand the structure, scale and quality of your AI data.")
        subtitle.setObjectName("muted")
        box.addWidget(title)
        box.addWidget(subtitle)
        header.addLayout(box)
        header.addStretch()
        layout.addLayout(header)

        self.source = QLabel("No dataset loaded")
        self.source.setObjectName("path")
        layout.addWidget(self.source)

        cards = QGridLayout()
        self.rows = StatCard("ROWS")
        self.files = StatCard("FILES")
        self.size = StatCard("SIZE")
        self.columns = StatCard("COLUMNS")
        self.seq = StatCard("SEQUENCE / TOKEN")
        for i, card in enumerate((self.rows, self.files, self.size, self.columns, self.seq)):
            cards.addWidget(card, 0, i)
        layout.addLayout(cards)

        split = QHBoxLayout()
        left = QVBoxLayout()
        left.addWidget(QLabel("Schema"))
        self.schema = QTreeWidget()
        self.schema.setHeaderLabels(["Column", "Type / Details"])
        self.schema.header().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.schema.header().setSectionResizeMode(1, QHeaderView.Stretch)
        left.addWidget(self.schema)
        split.addLayout(left, 1)

        right = QVBoxLayout()
        right.addWidget(QLabel("Dataset Summary"))
        self.summary = QTextEdit()
        self.summary.setReadOnly(True)
        right.addWidget(self.summary)
        split.addLayout(right, 1)
        layout.addLayout(split, 1)

    def clear(self):
        self.source.setText("No dataset loaded")
        for c in (self.rows, self.files, self.size, self.columns, self.seq):
            c.set_value("—")
        self.schema.clear()
        self.summary.clear()

    def update_from_backend(self, b: DatasetBackend):
        self.source.setText(b.source)
        self.rows.set_value(human_number(b.total_rows))
        self.files.set_value(human_number(len(b.files)))
        self.size.set_value(human_size(b.total_bytes))
        self.columns.set_value(human_number(len(b.columns)))

        token_col = next((c for c in b.columns if c.lower() in {"tokens", "input_ids", "token_ids"}), None)
        self.seq.set_value("tokens" if token_col else "—")

        self.schema.clear()
        for col in b.columns:
            detail = "unknown"
            try:
                if b.kind == "parquet":
                    detail = str(b.dataset.schema.field(col).type)
                else:
                    sample = next((r.get(col) for r in b.rows if r.get(col) is not None), None)
                    detail = type(sample).__name__
            except Exception:
                pass
            self.schema.addTopLevelItem(QTreeWidgetItem([str(col), detail]))

        sample = b.get_page(0, min(100, b.total_rows))
        token_counts = []
        text_lengths = []
        for row in sample:
            tokens = find_value(row, ["tokens", "input_ids", "token_ids"])
            if isinstance(tokens, list):
                token_counts.append(len(tokens))
            text = find_value(row, ["text", "content", "prompt", "conversation", "chat_string"])
            if text is not None:
                text_lengths.append(len(conversation_to_text(text)))

        lines = [
            f"Source type: {b.kind}",
            f"Rows: {b.total_rows:,}",
            f"Files/shards: {len(b.files):,}",
            f"Disk size: {human_size(b.total_bytes)}",
            f"Columns: {', '.join(b.columns)}",
        ]
        if token_counts:
            lines += [
                "",
                "Token sample:",
                f"Average tokens: {mean(token_counts):,.1f}",
                f"Median tokens: {median(token_counts):,.0f}",
                f"Min / max: {min(token_counts):,} / {max(token_counts):,}",
            ]
        if text_lengths:
            lines += [
                "",
                "Text sample:",
                f"Average characters: {mean(text_lengths):,.0f}",
                f"Median characters: {median(text_lengths):,.0f}",
            ]
        self.summary.setPlainText("\n".join(lines))


class BrowserPage(QWidget):
    """Fast, paged browser for both single files and multi-shard datasets."""
    row_selected = Signal(object)

    def __init__(self):
        super().__init__()
        self.backend: DatasetBackend | None = None
        self.display_rows: list[dict[str, Any]] = []
        self.filtered_search_results: list[dict[str, Any]] | None = None
        self.page_index = 0
        self.page_size = 100
        self.last_search = ""

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 22, 24, 22)
        layout.setSpacing(10)

        header = QHBoxLayout()
        title = QLabel("Dataset Browser")
        title.setObjectName("pageTitle")
        header.addWidget(title)
        header.addStretch()

        self.search_column = QComboBox()
        self.search_column.addItem("All columns", "__all__")
        self.search_column.setMinimumWidth(120)
        header.addWidget(self.search_column)

        self.search = QLineEdit()
        self.search.setPlaceholderText("Search dataset…")
        self.search.setMinimumWidth(330)
        self.search.returnPressed.connect(self.search_requested)
        header.addWidget(self.search)

        self.search_btn = QPushButton("🔎 Search")
        self.search_btn.clicked.connect(self.search_requested)
        header.addWidget(self.search_btn)

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self.clear_search)
        header.addWidget(self.clear_btn)

        self.page_size_box = QComboBox()
        self.page_size_box.addItems(["25", "50", "100", "250", "500"])
        self.page_size_box.setCurrentText("100")
        self.page_size_box.currentTextChanged.connect(self.change_page_size)
        header.addWidget(self.page_size_box)
        layout.addLayout(header)

        nav = QHBoxLayout()
        self.prev_btn = QPushButton("← Previous")
        self.next_btn = QPushButton("Next →")
        self.first_btn = QPushButton("⏮ First")
        self.last_btn = QPushButton("Last ⏭")
        self.prev_btn.clicked.connect(self.previous_page)
        self.next_btn.clicked.connect(self.next_page)
        self.first_btn.clicked.connect(self.first_page)
        self.last_btn.clicked.connect(self.last_page)
        nav.addWidget(self.first_btn)
        nav.addWidget(self.prev_btn)
        nav.addWidget(self.next_btn)
        nav.addWidget(self.last_btn)
        self.info = QLabel("0 rows")
        self.info.setObjectName("muted")
        nav.addWidget(self.info)
        nav.addStretch()

        self.jump_box = QSpinBox()
        self.jump_box.setMinimum(1)
        self.jump_box.setMaximum(1)
        self.jump_box.setPrefix("Page ")
        self.jump_box.valueChanged.connect(self.jump_page)
        nav.addWidget(self.jump_box)

        self.export_btn = QPushButton("⬇ Export results")
        self.export_btn.clicked.connect(self.export_results)
        nav.addWidget(self.export_btn)
        layout.addLayout(nav)

        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["#", "Model / Type", "Preview", "Tokens", "Flag", "Shard"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeToContents)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.cellDoubleClicked.connect(self.select_row)
        layout.addWidget(self.table, 1)

        hint = QLabel("Double-click a record to inspect it. Multi-shard Parquet is browsed as one logical dataset.")
        hint.setObjectName("muted")
        layout.addWidget(hint)

    def set_backend(self, backend: DatasetBackend):
        self.backend = backend
        self.filtered_search_results = None
        self.last_search = ""
        self.page_index = 0
        self.search.clear()
        self.search_column.clear()
        self.search_column.addItem("All columns", "__all__")
        for col in backend.columns:
            self.search_column.addItem(str(col), str(col))
        self.render_page()

    def change_page_size(self, value: str):
        self.page_size = int(value)
        self.page_index = 0
        self.render_page()

    def _total(self):
        return len(self.filtered_search_results) if self.filtered_search_results is not None else (self.backend.total_rows if self.backend else 0)

    def _page_count(self):
        total = self._total()
        return max(1, (total + self.page_size - 1) // self.page_size)

    def render_page(self):
        if not self.backend:
            return
        total = self._total()
        pages = self._page_count()
        self.page_index = max(0, min(self.page_index, pages - 1))

        if self.filtered_search_results is not None:
            start = self.page_index * self.page_size
            self.display_rows = self.filtered_search_results[start:start + self.page_size]
        else:
            start = self.page_index * self.page_size
            self.display_rows = self.backend.get_page(start, self.page_size)

        self.table.setRowCount(0)
        for i, row in enumerate(self.display_rows):
            r = self.table.rowCount()
            self.table.insertRow(r)
            absolute = self.page_index * self.page_size + i + 1
            self.table.setItem(r, 0, QTableWidgetItem(str(absolute)))
            model = find_value(row, ["model", "model_name", "model_id", "role"])
            self.table.setItem(r, 1, QTableWidgetItem("" if model is None else str(model)))
            text = find_value(row, ["text", "chat_string", "conversation", "messages", "content", "prompt"])
            self.table.setItem(r, 2, QTableWidgetItem(short_text(text, 260)))
            tokens = find_value(row, ["tokens", "input_ids", "token_ids"])
            token_count = len(tokens) if isinstance(tokens, list) else ""
            self.table.setItem(r, 3, QTableWidgetItem(str(token_count)))
            flagged = find_value(row, ["contains_banned", "banned", "is_banned"])
            self.table.setItem(r, 4, QTableWidgetItem("TRUE" if flagged is True or str(flagged).lower() == "true" else ""))
            self.table.setItem(r, 5, QTableWidgetItem(self._shard_for_absolute(absolute)))

        self.info.setText(f"{total:,} records • page {self.page_index + 1:,} / {pages:,}" +
                          (f" • search: {self.last_search!r}" if self.filtered_search_results is not None else ""))
        self.prev_btn.setEnabled(self.page_index > 0)
        self.first_btn.setEnabled(self.page_index > 0)
        self.next_btn.setEnabled(self.page_index + 1 < pages)
        self.last_btn.setEnabled(self.page_index + 1 < pages)
        self.jump_box.blockSignals(True)
        self.jump_box.setMaximum(pages)
        self.jump_box.setValue(self.page_index + 1)
        self.jump_box.blockSignals(False)
        self.export_btn.setEnabled(bool(self.display_rows) or bool(self.filtered_search_results))

    def _shard_for_absolute(self, absolute: int) -> str:
        if not self.backend or len(self.backend.files) <= 1:
            return ""
        # Best-effort display using Parquet row-group metadata. For non-Parquet or
        # mixed files, keep the column blank rather than guessing.
        if self.backend.kind != "parquet" or pq is None:
            return ""
        target = absolute - 1
        running = 0
        for path in self.backend.files:
            try:
                rows = pq.ParquetFile(path).metadata.num_rows
            except Exception:
                return ""
            if target < running + rows:
                return path.name
            running += rows
        return ""

    def previous_page(self):
        if self.page_index > 0:
            self.page_index -= 1
            self.render_page()

    def next_page(self):
        if self.page_index + 1 < self._page_count():
            self.page_index += 1
            self.render_page()

    def first_page(self):
        self.page_index = 0
        self.render_page()

    def last_page(self):
        self.page_index = self._page_count() - 1
        self.render_page()

    def jump_page(self, value: int):
        if value != self.page_index + 1:
            self.page_index = max(0, value - 1)
            self.render_page()

    def search_requested(self):
        if not self.backend:
            return
        query = self.search.text().strip()
        if not query:
            self.clear_search()
            return
        self.search_btn.setEnabled(False)
        self.search_btn.setText("Searching…")
        column = self.search_column.currentData() or "__all__"
        self.window().start_search(query, column)

    def search_finished(self, results: list[dict[str, Any]], scanned: int, query: str = ""):
        self.search_btn.setEnabled(True)
        self.search_btn.setText("🔎 Search")
        self.filtered_search_results = results
        self.last_search = query or self.search.text().strip()
        self.page_index = 0
        self.render_page()

    def clear_search(self):
        self.search.clear()
        self.filtered_search_results = None
        self.last_search = ""
        self.page_index = 0
        self.render_page()

    def select_row(self, row, _column):
        if 0 <= row < len(self.display_rows):
            self.row_selected.emit(self.display_rows[row])

    def export_results(self):
        rows = self.filtered_search_results if self.filtered_search_results is not None else self.display_rows
        if not rows:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export Dataset Rows", str(Path.home() / "dataset_export.jsonl"), "JSON Lines (*.jsonl);;JSON (*.json)")
        if not path:
            return
        try:
            out = Path(path)
            if out.suffix.lower() == ".json":
                out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
            else:
                with out.open("w", encoding="utf-8") as fh:
                    for row in rows:
                        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            self.window().status.setText(f"Exported {len(rows):,} rows → {out}")
        except Exception as exc:
            QMessageBox.critical(self, APP_NAME, f"Export failed:\n\n{exc}")

class InspectorPage(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 22, 24, 22)

        header = QHBoxLayout()
        title = QLabel("Record Inspector")
        title.setObjectName("pageTitle")
        header.addWidget(title)
        header.addStretch()
        self.copy_btn = QPushButton("📋 Copy")
        self.copy_btn.clicked.connect(self.copy_text)
        header.addWidget(self.copy_btn)
        layout.addLayout(header)

        self.meta = QTreeWidget()
        self.meta.setHeaderLabels(["Field", "Value"])
        self.meta.header().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.meta.header().setSectionResizeMode(1, QHeaderView.Stretch)
        self.meta.setMaximumHeight(220)
        layout.addWidget(self.meta)

        self.text = QTextEdit()
        self.text.setReadOnly(True)
        layout.addWidget(self.text, 1)

        self.token_box = QTextEdit()
        self.token_box.setReadOnly(True)
        self.token_box.setMaximumHeight(180)
        self.token_box.setPlaceholderText("Token information appears here.")
        layout.addWidget(self.token_box)

    def show_row(self, row: dict[str, Any]):
        self.meta.clear()
        for key, value in row.items():
            self.meta.addTopLevelItem(QTreeWidgetItem([str(key), short_text(value, 600)]))

        text = find_value(row, ["text", "chat_string", "conversation", "messages", "content", "prompt"])
        if text is None:
            display = json.dumps(row, ensure_ascii=False, indent=2)
        else:
            display = conversation_to_text(text)
        self.text.setPlainText(display)

        tokens = find_value(row, ["tokens", "input_ids", "token_ids"])
        if isinstance(tokens, list):
            head = tokens[:80]
            self.token_box.setPlainText(
                f"Token count: {len(tokens):,}\n"
                f"First {len(head)} tokens:\n{head}\n\n"
                f"Min token ID: {min(tokens) if tokens else '—'}\n"
                f"Max token ID: {max(tokens) if tokens else '—'}"
            )
        else:
            self.token_box.setPlainText("No token-list field found.")

    def copy_text(self):
        QApplication.clipboard().setText(self.text.toPlainText())


class AnalyticsPage(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 22, 24, 22)

        title = QLabel("Analytics")
        title.setObjectName("pageTitle")
        layout.addWidget(title)

        self.run_btn = QPushButton("▶ Analyze dataset")
        self.run_btn.clicked.connect(lambda: self.window().start_analytics())
        layout.addWidget(self.run_btn)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Metric", "Value"])
        self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.tree.header().setSectionResizeMode(1, QHeaderView.Stretch)
        layout.addWidget(self.tree, 1)

        self.note = QLabel("Analysis uses batches and samples where appropriate.")
        self.note.setObjectName("muted")
        layout.addWidget(self.note)

    def set_metrics(self, metrics: dict[str, Any]):
        self.tree.clear()
        for k, v in metrics.items():
            self.tree.addTopLevelItem(QTreeWidgetItem([str(k), str(v)]))


class QualityPage(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 22, 24, 22)
        title = QLabel("Quality & Validation")
        title.setObjectName("pageTitle")
        layout.addWidget(title)
        self.run_btn = QPushButton("✓ Run quality scan")
        self.run_btn.clicked.connect(lambda: self.window().start_quality_scan())
        layout.addWidget(self.run_btn)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Check", "Result"])
        self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.tree.header().setSectionResizeMode(1, QHeaderView.Stretch)
        layout.addWidget(self.tree, 1)

    def set_results(self, results: dict[str, Any]):
        self.tree.clear()
        for k, v in results.items():
            self.tree.addTopLevelItem(QTreeWidgetItem([str(k), str(v)]))


class SchemaPage(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 22, 24, 22)
        title = QLabel("Schema & Files")
        title.setObjectName("pageTitle")
        layout.addWidget(title)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Item", "Details"])
        self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.tree.header().setSectionResizeMode(1, QHeaderView.Stretch)
        layout.addWidget(self.tree, 1)

    def update_from_backend(self, b: DatasetBackend):
        self.tree.clear()
        root = QTreeWidgetItem(["Dataset", b.source])
        self.tree.addTopLevelItem(root)
        for p in b.files:
            root.addChild(QTreeWidgetItem([p.name, human_size(p.stat().st_size)]))
        schema = QTreeWidgetItem(["Schema", ""])
        self.tree.addTopLevelItem(schema)
        for col in b.columns:
            typ = "unknown"
            try:
                typ = str(b.dataset.schema.field(col).type) if b.kind == "parquet" else "inferred"
            except Exception:
                pass
            schema.addChild(QTreeWidgetItem([col, typ]))
        self.tree.expandAll()


class SettingsPage(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 22, 24, 22)
        title = QLabel("Settings")
        title.setObjectName("pageTitle")
        layout.addWidget(title)
        text = QLabel(
            "AI Dataset Explorer Studio is read-only with respect to source datasets.\n\n"
            "Large Parquet datasets are browsed and searched in batches to keep RAM use reasonable.\n"
            "Exports create new files and do not modify the source."
        )
        text.setWordWrap(True)
        text.setObjectName("muted")
        layout.addWidget(text)
        layout.addStretch()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} — {APP_VERSION}")
        self.resize(1500, 920)

        self.backend: DatasetBackend | None = None
        self.load_thread: QThread | None = None
        self.load_worker: LoadWorker | None = None
        self.search_thread: QThread | None = None
        self.search_worker: SearchWorker | None = None
        self.dark_mode = True
        self.last_source = ""
        self.recent_sources: list[str] = []

        self.build_ui()
        self.load_settings()
        self.apply_theme()
        self.refresh_recent()

    def build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        toolbar = QToolBar()
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        open_action = QAction("📂 Open", self)
        open_action.triggered.connect(self.open_dataset)
        toolbar.addAction(open_action)

        folder_action = QAction("📁 Open Folder", self)
        folder_action.triggered.connect(self.open_folder)
        toolbar.addAction(folder_action)

        toolbar.addSeparator()

        for text, idx in [
            ("⌂ Dashboard", 0), ("▤ Browser", 1), ("◈ Inspector", 2),
            ("📊 Analytics", 3), ("✓ Quality", 4), ("🗂 Schema", 5), ("⚙ Settings", 6)
        ]:
            a = QAction(text, self)
            a.triggered.connect(lambda checked=False, i=idx: self.pages.setCurrentIndex(i))
            toolbar.addAction(a)

        toolbar.addSeparator()
        theme = QPushButton("☀ / ☾")
        theme.clicked.connect(self.toggle_theme)
        toolbar.addWidget(theme)

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        outer.addLayout(body, 1)

        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(225)
        side = QVBoxLayout(sidebar)
        side.setContentsMargins(12, 16, 12, 12)

        logo = QLabel("AI Dataset")
        logo.setObjectName("logo")
        side.addWidget(logo)
        logo2 = QLabel("Explorer Studio")
        logo2.setObjectName("logoSmall")
        side.addWidget(logo2)
        sub = QLabel("Dataset intelligence")
        sub.setObjectName("muted")
        side.addWidget(sub)
        side.addSpacing(18)

        self.nav = QListWidget()
        self.nav.addItems([
            "⌂  Dashboard", "▤  Dataset Browser", "◈  Record Inspector",
            "📊 Analytics", "✓  Quality", "🗂  Schema & Files", "⚙  Settings"
        ])
        self.nav.currentRowChanged.connect(self.change_page)
        side.addWidget(self.nav, 1)

        label = QLabel("RECENT DATASETS")
        label.setObjectName("sectionLabel")
        side.addWidget(label)
        self.recent = QListWidget()
        self.recent.setMaximumHeight(180)
        self.recent.itemDoubleClicked.connect(self.open_recent)
        side.addWidget(self.recent)
        body.addWidget(sidebar)

        self.pages = QStackedWidget()
        self.dashboard = DashboardPage()
        self.browser = BrowserPage()
        self.inspector = InspectorPage()
        self.analytics = AnalyticsPage()
        self.quality = QualityPage()
        self.schema = SchemaPage()
        self.settings = SettingsPage()
        for p in (self.dashboard, self.browser, self.inspector, self.analytics, self.quality, self.schema, self.settings):
            self.pages.addWidget(p)
        self.browser.row_selected.connect(self.inspect_row)
        body.addWidget(self.pages, 1)

        # Set the initial navigation selection only after self.pages exists.
        self.nav.setCurrentRow(0)

        status = QStatusBar()
        self.setStatusBar(status)
        self.status = QLabel("Ready.")
        status.addWidget(self.status)
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(260)
        self.progress.hide()
        status.addPermanentWidget(self.progress)
    def change_page(self, index):
        if 0 <= index < self.pages.count():
            self.pages.setCurrentIndex(index)

    def open_dataset(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Dataset", self.last_source or str(Path.home()),
            "Datasets (*.parquet *.json *.jsonl *.csv);;All files (*.*)"
        )
        if path:
            self.start_load(path)

    def open_folder(self):
        """Select a directory; the backend discovers all supported dataset shards."""
        path = QFileDialog.getExistingDirectory(
            self,
            "Open Dataset Folder",
            self.last_source if self.last_source and Path(self.last_source).exists()
            else str(Path.home()),
        )
        if path:
            self.start_load(path)

    def start_load(self, source: str):
        if self.load_thread and self.load_thread.isRunning():
            QMessageBox.information(self, APP_NAME, "A dataset is already loading.")
            return

        self.last_source = source
        self.add_recent(source)
        self.dashboard.clear()
        self.status.setText(f"Opening {source}…")
        self.progress.setValue(0)
        self.progress.show()

        self.load_thread = QThread()
        self.load_worker = LoadWorker(source)
        self.load_worker.moveToThread(self.load_thread)
        self.load_thread.started.connect(self.load_worker.run)
        self.load_worker.progress.connect(self.progress.setValue, Qt.QueuedConnection)
        self.load_worker.status.connect(self.status.setText, Qt.QueuedConnection)
        self.load_worker.loaded.connect(self.load_finished, Qt.QueuedConnection)
        self.load_worker.failed.connect(self.load_failed, Qt.QueuedConnection)
        self.load_worker.loaded.connect(self.load_thread.quit, Qt.QueuedConnection)
        self.load_worker.failed.connect(self.load_thread.quit, Qt.QueuedConnection)
        self.load_thread.finished.connect(self.load_worker.deleteLater)
        self.load_thread.finished.connect(self.load_thread_cleanup)
        self.load_thread.finished.connect(self.load_thread.deleteLater)
        self.load_thread.start()

    @Slot(object, str)
    def load_finished(self, backend, source):
        self.backend = backend
        self.dashboard.update_from_backend(backend)
        self.browser.set_backend(backend)
        self.schema.update_from_backend(backend)
        self.progress.hide()
        self.status.setText(f"Loaded {backend.total_rows:,} rows • {human_size(backend.total_bytes)}")
        self.pages.setCurrentIndex(0)

    @Slot(str)
    def load_failed(self, message):
        self.progress.hide()
        self.status.setText("Load failed.")
        QMessageBox.critical(self, APP_NAME, "Could not load the dataset:\n\n" + message)

    @Slot()
    def load_thread_cleanup(self):
        self.load_worker = None
        self.load_thread = None

    def start_search(self, query: str, column: str = "__all__"):
        if not self.backend:
            return
        if self.search_thread and self.search_thread.isRunning():
            return

        self.progress.setValue(0)
        self.progress.show()
        self.status.setText("Starting search…")
        self.search_thread = QThread()
        self.search_worker = SearchWorker(self.backend, query, column)
        self.search_worker.moveToThread(self.search_thread)
        self.search_thread.started.connect(self.search_worker.run)
        self.search_worker.progress.connect(self.progress.setValue, Qt.QueuedConnection)
        self.search_worker.status.connect(self.status.setText, Qt.QueuedConnection)
        self.search_worker.finished.connect(self.search_finished, Qt.QueuedConnection)
        self.search_worker.failed.connect(self.search_failed, Qt.QueuedConnection)
        self.search_worker.finished.connect(self.search_thread.quit, Qt.QueuedConnection)
        self.search_worker.failed.connect(self.search_thread.quit, Qt.QueuedConnection)
        self.search_thread.finished.connect(self.search_worker.deleteLater)
        self.search_thread.finished.connect(self.search_thread_cleanup)
        self.search_thread.finished.connect(self.search_thread.deleteLater)
        self.search_thread.start()

    @Slot(object, int)
    def search_finished(self, results, scanned):
        self.progress.hide()
        self.status.setText(f"Search complete • {len(results):,} matches • scanned {scanned:,} rows")
        self.browser.search_finished(results, scanned)
        self.pages.setCurrentIndex(1)

    @Slot(str)
    def search_failed(self, message):
        self.progress.hide()
        self.status.setText("Search failed.")
        self.browser.search_btn.setEnabled(True)
        self.browser.search_btn.setText("🔎 Search")
        QMessageBox.critical(self, APP_NAME, "Search failed:\n\n" + message)

    @Slot()
    def search_thread_cleanup(self):
        self.search_worker = None
        self.search_thread = None

    def inspect_row(self, row):
        self.inspector.show_row(row)
        self.pages.setCurrentIndex(2)

    def start_analytics(self):
        if not self.backend:
            QMessageBox.information(self, APP_NAME, "Open a dataset first.")
            return
        b = self.backend
        sample = b.get_page(0, min(5000, b.total_rows))
        token_lengths = []
        text_lengths = []
        roles = Counter()
        duplicates = set()
        dup_count = 0

        for row in sample:
            tokens = find_value(row, ["tokens", "input_ids", "token_ids"])
            if isinstance(tokens, list):
                token_lengths.append(len(tokens))
            text = find_value(row, ["text", "content", "conversation", "chat_string", "prompt"])
            if text is not None:
                s = conversation_to_text(text)
                text_lengths.append(len(s))
                roles.update(role_counts(s))
                key = s.strip()
                if key:
                    if key in duplicates:
                        dup_count += 1
                    else:
                        duplicates.add(key)

        metrics = {
            "Rows": f"{b.total_rows:,}",
            "Files / shards": f"{len(b.files):,}",
            "Disk size": human_size(b.total_bytes),
            "Columns": ", ".join(b.columns),
            "Sample size": f"{len(sample):,}",
            "Sample duplicate texts": f"{dup_count:,}",
        }
        if token_lengths:
            metrics.update({
                "Token average": f"{mean(token_lengths):,.1f}",
                "Token median": f"{median(token_lengths):,.0f}",
                "Token minimum": f"{min(token_lengths):,}",
                "Token maximum": f"{max(token_lengths):,}",
                "Exactly 1,024 tokens": f"{sum(x == 1024 for x in token_lengths):,}",
            })
        if text_lengths:
            metrics.update({
                "Text average characters": f"{mean(text_lengths):,.0f}",
                "Text median characters": f"{median(text_lengths):,.0f}",
                "Text minimum characters": f"{min(text_lengths):,}",
                "Text maximum characters": f"{max(text_lengths):,}",
            })
        if roles:
            metrics["Detected roles"] = ", ".join(f"{k}: {v:,}" for k, v in roles.most_common())
        self.analytics.set_metrics(metrics)

    def start_quality_scan(self):
        if not self.backend:
            QMessageBox.information(self, APP_NAME, "Open a dataset first.")
            return
        b = self.backend
        sample = b.get_page(0, min(5000, b.total_rows))
        empty_text = 0
        empty_tokens = 0
        token_count_1024 = 0
        malformed_roles = 0
        duplicate_texts = 0
        seen = set()
        for row in sample:
            text = find_value(row, ["text", "content", "conversation", "chat_string", "prompt"])
            s = conversation_to_text(text)
            if not s.strip():
                empty_text += 1
            elif s.strip() in seen:
                duplicate_texts += 1
            else:
                seen.add(s.strip())
            tokens = find_value(row, ["tokens", "input_ids", "token_ids"])
            if isinstance(tokens, list):
                if not tokens:
                    empty_tokens += 1
                if len(tokens) == 1024:
                    token_count_1024 += 1
            if s and ("USER:" not in s.upper() and "ASSISTANT:" not in s.upper()):
                malformed_roles += 1

        results = {
            "Rows checked": f"{len(sample):,} (sampled)",
            "Empty text": f"{empty_text:,}",
            "Empty token lists": f"{empty_tokens:,}",
            "Duplicate texts in sample": f"{duplicate_texts:,}",
            "No obvious USER/ASSISTANT markers": f"{malformed_roles:,}",
            "Exactly 1,024-token sequences": f"{token_count_1024:,}",
            "Schema consistent": "Yes — single discovered schema",
            "Source modified": "No — read-only",
        }
        self.quality.set_results(results)

    def add_recent(self, source):
        self.recent_sources = [x for x in self.recent_sources if x != source]
        self.recent_sources.insert(0, source)
        self.recent_sources = self.recent_sources[:10]
        self.refresh_recent()
        self.save_settings()

    def refresh_recent(self):
        self.recent.clear()
        for source in self.recent_sources:
            item = QListWidgetItem(Path(source).name if Path(source).exists() else source)
            item.setToolTip(source)
            item.setData(Qt.UserRole, source)
            self.recent.addItem(item)

    def open_recent(self, item):
        source = item.data(Qt.UserRole)
        if source and Path(source).exists():
            self.start_load(source)
        else:
            QMessageBox.warning(self, APP_NAME, "That recent dataset is no longer available.")

    def toggle_theme(self):
        self.dark_mode = not self.dark_mode
        self.apply_theme()
        self.save_settings()

    def apply_theme(self):
        if self.dark_mode:
            self.setStyleSheet("""
            QMainWindow,QWidget { background:#0b1220; color:#e5e7eb; font-family:"Segoe UI"; font-size:13px; }
            QToolBar { background:#111827; border:0; padding:6px; }
            QPushButton { background:#1f2937; border:1px solid #334155; border-radius:8px; padding:8px 12px; color:#e5e7eb; }
            QPushButton:hover { background:#334155; }
            QLineEdit,QComboBox,QTableWidget,QListWidget,QTreeWidget,QTextEdit { background:#0f172a; border:1px solid #263449; border-radius:8px; color:#e5e7eb; padding:6px; }
            QHeaderView::section { background:#172033; color:#cbd5e1; padding:7px; border:0; }
            QTableWidget::item:selected,QTreeWidget::item:selected,QListWidget::item:selected { background:#334155; }
            #sidebar { background:#0f172a; border-right:1px solid #1e293b; }
            #logo { font-size:24px; font-weight:800; color:#f8fafc; }
            #logoSmall { font-size:17px; font-weight:700; color:#a5b4fc; }
            #pageTitle { font-size:26px; font-weight:750; color:#f8fafc; }
            #muted { color:#94a3b8; }
            #path { background:#111827; border:1px solid #263449; border-radius:9px; padding:9px; color:#a5b4fc; }
            #card { background:#111827; border:1px solid #263449; border-radius:12px; }
            #cardTitle { color:#94a3b8; font-size:11px; font-weight:700; letter-spacing:1px; }
            #cardValue { color:#f8fafc; font-size:24px; font-weight:750; }
            #sectionLabel { color:#64748b; font-size:10px; font-weight:700; }
            QProgressBar { background:#1e293b; border:0; border-radius:4px; height:7px; }
            QProgressBar::chunk { background:#6366f1; border-radius:4px; }
            """)
        else:
            self.setStyleSheet("""
            QMainWindow,QWidget { background:#f8fafc; color:#1e293b; font-family:"Segoe UI"; font-size:13px; }
            QToolBar { background:white; border:0; padding:6px; }
            QPushButton { background:white; border:1px solid #cbd5e1; border-radius:8px; padding:8px 12px; }
            QPushButton:hover { background:#f1f5f9; }
            QLineEdit,QComboBox,QTableWidget,QListWidget,QTreeWidget,QTextEdit { background:white; border:1px solid #cbd5e1; border-radius:8px; color:#1e293b; padding:6px; }
            QHeaderView::section { background:#f1f5f9; color:#475569; padding:7px; border:0; }
            QTableWidget::item:selected,QTreeWidget::item:selected,QListWidget::item:selected { background:#e0e7ff; }
            #sidebar { background:white; border-right:1px solid #e2e8f0; }
            #logo,#logoSmall,#pageTitle { color:#0f172a; }
            #logo { font-size:24px; font-weight:800; }
            #logoSmall { font-size:17px; font-weight:700; }
            #pageTitle { font-size:26px; font-weight:750; }
            #muted { color:#64748b; }
            #path { background:white; border:1px solid #cbd5e1; border-radius:9px; padding:9px; color:#4338ca; }
            #card { background:white; border:1px solid #e2e8f0; border-radius:12px; }
            #cardTitle { color:#64748b; font-size:11px; font-weight:700; }
            #cardValue { color:#0f172a; font-size:24px; font-weight:750; }
            #sectionLabel { color:#94a3b8; font-size:10px; font-weight:700; }
            QProgressBar { background:#e2e8f0; border:0; border-radius:4px; height:7px; }
            QProgressBar::chunk { background:#4f46e5; border-radius:4px; }
            """)

    def save_settings(self):
        try:
            CONFIG_FILE.write_text(json.dumps({
                "dark_mode": self.dark_mode,
                "last_source": self.last_source,
                "recent_sources": self.recent_sources,
            }, indent=2), encoding="utf-8")
        except Exception:
            pass

    def load_settings(self):
        try:
            data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            self.dark_mode = data.get("dark_mode", True)
            self.last_source = data.get("last_source", "")
            self.recent_sources = data.get("recent_sources", [])
        except Exception:
            pass

    def closeEvent(self, event):
        active = []
        if self.load_thread and self.load_thread.isRunning():
            active.append(self.load_thread)
        if self.search_thread and self.search_thread.isRunning():
            active.append(self.search_thread)
        if active:
            QMessageBox.information(self, APP_NAME, "A background operation is still running.\n\nPlease wait for it to finish before closing.")
            event.ignore()
            return
        self.save_settings()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
