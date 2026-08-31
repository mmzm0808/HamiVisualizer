"""启动环境诊断回归。

这些测试只调用纯 Python 的版本检查，不导入 Qt 原生模块；这样即使用户
环境损坏，也能保证诊断入口本身不会再制造一条难以解释的崩溃堆栈。
"""

from __future__ import annotations

import pytest

from hamivisualizer import main as startup


def test_runtime_compatibility_accepts_tested_stack():
    versions = startup._check_runtime_compatibility()

    assert versions["numpy"].split(".", 1)[0] == "1"
    assert tuple(int(part) for part in versions["PySide6"].split(".")[:2]) == (6, 6)
    assert versions["sympy"].split(".", 1)[0] == "1"


def test_runtime_compatibility_reports_missing_dependency(monkeypatch):
    def missing(distribution: str):
        return None if distribution == "PySide6" else "1.26.4"

    monkeypatch.setattr(startup, "_installed_version", missing)

    with pytest.raises(RuntimeError, match="缺少运行依赖：PySide6") as error:
        startup._check_runtime_compatibility()

    assert "pip install -r requirements.txt" in str(error.value)


def test_runtime_compatibility_blocks_numpy_two_before_native_import(monkeypatch):
    def incompatible(distribution: str):
        return {
            "numpy": "2.4.6",
            "PySide6": "6.6.3",
            "sympy": "1.14.0",
        }[distribution]

    monkeypatch.setattr(startup, "_installed_version", incompatible)

    with pytest.raises(RuntimeError, match="ABI 不兼容") as error:
        startup._check_runtime_compatibility()

    assert "numpy>=1.26,<2" in str(error.value)


def test_runtime_compatibility_blocks_unverified_qt_stack(monkeypatch):
    def unsupported(distribution: str):
        return {
            "numpy": "1.26.4",
            "PySide6": "6.7.0",
            "sympy": "1.14.0",
        }[distribution]

    monkeypatch.setattr(startup, "_installed_version", unsupported)

    with pytest.raises(RuntimeError, match="PySide6 版本为 6.7.0") as error:
        startup._check_runtime_compatibility()

    assert "PySide6>=6.6,<6.7" in str(error.value)
