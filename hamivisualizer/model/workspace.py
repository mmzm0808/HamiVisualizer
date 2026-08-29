"""Application workspace persistence, preferences, and bounded undo history.

The model file remains the portable unit.  Workspace and preferences are kept
separately under ``~/.hvisual`` so opening a shared model never mutates a
user's UI configuration.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
import tempfile
from typing import Any
from uuid import uuid4


APP_DIR_NAME = ".hvisual"
WORKSPACE_VERSION = 1
PREFERENCES_VERSION = 1
MAX_PREFERENCES_BYTES = 64 * 1024
MAX_WORKSPACE_BYTES = 1 * 1024 * 1024
MAX_SESSIONS = 32
MAX_SPLITTER_SIZE = 1_000_000
_RESULT_NAMES = {"矩阵+晶格", "矩阵", "晶格", "能带", "波函数"}


def app_data_dir(home: str | Path | None = None) -> Path:
    """Return the default per-user data directory without creating it."""
    root = Path(home).expanduser() if home is not None else Path.home()
    return root / APP_DIR_NAME


@dataclass
class Preferences:
    ui_scale: int = 100
    undo_limit: int = 10
    autosave: bool = True
    calculation_mode: str = "automatic"
    debounce_ms: int = 300
    snap_step: float = 0.25
    snap_enabled: bool = True
    check_updates: bool = True
    # 外观主题: "light" | "dark" | "system"（跟随系统）
    theme: str = "system"

    def normalized(self) -> "Preferences":
        self.ui_scale = min(180, max(80, round(int(self.ui_scale) / 10) * 10))
        self.undo_limit = min(100, max(1, int(self.undo_limit)))
        self.autosave = self.autosave if isinstance(self.autosave, bool) else True
        self.snap_enabled = (
            self.snap_enabled if isinstance(self.snap_enabled, bool) else True
        )
        self.check_updates = (
            self.check_updates if isinstance(self.check_updates, bool) else True
        )
        self.calculation_mode = (
            self.calculation_mode
            if self.calculation_mode in {"automatic", "manual"}
            else "automatic"
        )
        self.debounce_ms = min(3000, max(100, int(self.debounce_ms)))
        self.snap_step = min(10.0, max(0.001, float(self.snap_step)))
        self.theme = self.theme if self.theme in {"light", "dark", "system"} else "system"
        return self


@dataclass
class ModelSessionData:
    id: str = field(default_factory=lambda: uuid4().hex)
    name: str = "未命名模型"
    path: str = ""
    dirty: bool = False
    result_tab: int = 0
    edit_mode: bool = False


@dataclass
class WorkspaceData:
    sessions: list[ModelSessionData] = field(default_factory=list)
    # 最近打开/保存的可移植模型；工作区文件而非模型文件的一部分。
    recent_models: list[str] = field(default_factory=list)
    current_index: int = 0
    split_enabled: bool = False
    split_model_id: str = ""
    split_result: str = "晶格"
    splitter_sizes: list[int] = field(default_factory=lambda: [1, 1])


class DocumentHistory:
    """Small, deterministic snapshot history used per model tab."""

    def __init__(self, limit: int = 10):
        self.limit = max(1, int(limit))
        self._undo: list[dict] = []
        self._redo: list[dict] = []
        # Labels travel with snapshots rather than being inferred after the
        # fact.  This mirrors command-stack affordances without making saved
        # workspace data depend on implementation-only UI text.
        self._undo_labels: list[str] = []
        self._redo_labels: list[str] = []

    @staticmethod
    def _same(a: dict, b: dict) -> bool:
        return a == b

    def seed(self, document: dict) -> None:
        self._undo = [deepcopy(document)]
        self._redo.clear()
        self._undo_labels = ["初始状态"]
        self._redo_labels.clear()

    def push(self, document: dict, label: str = "编辑模型") -> bool:
        snap = deepcopy(document)
        if self._undo and self._same(self._undo[-1], snap):
            return False
        self._undo.append(snap)
        self._undo_labels.append(str(label or "编辑模型"))
        if len(self._undo) > self.limit + 1:
            remove = len(self._undo) - self.limit - 1
            del self._undo[:remove]
            del self._undo_labels[:remove]
        self._redo.clear()
        self._redo_labels.clear()
        return True

    @property
    def can_undo(self) -> bool:
        return len(self._undo) > 1

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    @property
    def undo_label(self) -> str:
        return self._undo_labels[-1] if self.can_undo else ""

    @property
    def redo_label(self) -> str:
        return self._redo_labels[-1] if self.can_redo else ""

    def undo(self) -> dict | None:
        if not self.can_undo:
            return None
        self._redo.append(self._undo.pop())
        self._redo_labels.append(self._undo_labels.pop())
        return deepcopy(self._undo[-1])

    def redo(self) -> dict | None:
        if not self._redo:
            return None
        snap = self._redo.pop()
        label = self._redo_labels.pop()
        self._undo.append(deepcopy(snap))
        self._undo_labels.append(label)
        return deepcopy(snap)

    def set_limit(self, value: int) -> None:
        self.limit = max(1, int(value))
        if len(self._undo) > self.limit + 1:
            remove = len(self._undo) - self.limit - 1
            del self._undo[:remove]
            del self._undo_labels[:remove]


def _atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temp_name = stream.name
            json.dump(obj, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
        temp_name = None
    finally:
        if temp_name is not None:
            Path(temp_name).unlink(missing_ok=True)


def _read_json(path: Path, max_bytes: int) -> Any:
    """Read a bounded JSON document used by the local workspace layer."""
    if path.stat().st_size > max_bytes:
        raise ValueError(f"配置文件过大（最多 {max_bytes // 1024} KiB）")
    return json.loads(path.read_text(encoding="utf-8"))


def load_preferences(root: Path | None = None) -> Preferences:
    path = (root or app_data_dir()) / "preferences.json"
    try:
        obj = _read_json(path, MAX_PREFERENCES_BYTES)
        if not isinstance(obj, dict) or obj.get("version") != PREFERENCES_VERSION:
            raise ValueError
        values = {k: obj[k] for k in asdict(Preferences()) if k in obj}
        return Preferences(**values).normalized()
    except (OSError, ValueError, TypeError, OverflowError, json.JSONDecodeError):
        return Preferences()


def save_preferences(prefs: Preferences, root: Path | None = None) -> None:
    obj = {"version": PREFERENCES_VERSION, **asdict(prefs.normalized())}
    _atomic_json((root or app_data_dir()) / "preferences.json", obj)


def load_workspace(root: Path | None = None) -> WorkspaceData:
    path = (root or app_data_dir()) / "workspace.json"
    try:
        obj = _read_json(path, MAX_WORKSPACE_BYTES)
        if not isinstance(obj, dict) or obj.get("version") != WORKSPACE_VERSION:
            raise ValueError
        sessions = []
        raw_sessions = obj.get("sessions", [])
        if not isinstance(raw_sessions, list):
            raise ValueError
        for raw in raw_sessions[:MAX_SESSIONS]:
            if not isinstance(raw, dict):
                continue
            allowed = {k: raw[k] for k in asdict(ModelSessionData()) if k in raw}
            try:
                session = ModelSessionData(**allowed)
            except (TypeError, ValueError):
                continue
            if (
                not isinstance(session.id, str) or not session.id or len(session.id) > 128
                or not isinstance(session.name, str) or len(session.name) > 256
                or not isinstance(session.path, str) or len(session.path) > 4096
                or not isinstance(session.dirty, bool)
                or not isinstance(session.result_tab, int) or isinstance(session.result_tab, bool)
                or not 0 <= session.result_tab <= 4
                or not isinstance(session.edit_mode, bool)
            ):
                continue
            sessions.append(session)
        recent_in = obj.get("recent_models", [])
        if not isinstance(recent_in, list):
            recent_in = []
        recent = [
            value[:4096] for value in recent_in
            if isinstance(value, str) and value.strip() and len(value) <= 4096
        ][:10]
        raw_index = obj.get("current_index", 0)
        if isinstance(raw_index, bool) or not isinstance(raw_index, int):
            raw_index = 0
        current_index = min(max(0, raw_index), max(0, len(sessions) - 1))
        split_result = obj.get("split_result", "晶格")
        if split_result not in _RESULT_NAMES:
            split_result = "晶格"
        raw_sizes = obj.get("splitter_sizes", [1, 1])
        if not isinstance(raw_sizes, list) or len(raw_sizes) < 2:
            splitter_sizes = [1, 1]
        else:
            try:
                splitter_sizes = [
                    min(MAX_SPLITTER_SIZE, max(1, int(raw_sizes[0]))),
                    min(MAX_SPLITTER_SIZE, max(1, int(raw_sizes[1]))),
                ]
            except (TypeError, ValueError, OverflowError):
                splitter_sizes = [1, 1]
        split_enabled = obj.get("split_enabled", False)
        if not isinstance(split_enabled, bool):
            split_enabled = False
        return WorkspaceData(
            sessions=sessions,
            recent_models=recent,
            current_index=current_index,
            split_enabled=split_enabled,
            split_model_id=(
                obj.get("split_model_id", "")
                if isinstance(obj.get("split_model_id", ""), str)
                else ""
            )[:128],
            split_result=split_result,
            splitter_sizes=splitter_sizes,
        )
    except (OSError, ValueError, TypeError, OverflowError, json.JSONDecodeError):
        return WorkspaceData()


def save_workspace(data: WorkspaceData, root: Path | None = None) -> None:
    obj = {
        "version": WORKSPACE_VERSION,
        "sessions": [asdict(s) for s in data.sessions],
        "recent_models": list(data.recent_models[:10]),
        "current_index": int(data.current_index),
        "split_enabled": bool(data.split_enabled),
        "split_model_id": data.split_model_id,
        "split_result": data.split_result,
        "splitter_sizes": list(data.splitter_sizes[:2]),
    }
    _atomic_json((root or app_data_dir()) / "workspace.json", obj)
