"""Shared Qt test isolation.

PySide keeps top-level widgets alive until deferred-delete events are processed.
Without cleanup, application-wide theme changes re-polish every window created by
all preceding tests, making the full suite progressively slower and eventually
look hung even though the same theme tests pass in isolation.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def cleanup_qt_top_levels():
    yield

    # Import lazily so pure-model tests do not create a QApplication.
    from PySide6.QtCore import QCoreApplication, QEvent
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        return
    for widget in list(app.topLevelWidgets()):
        widget.hide()
        widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    app.processEvents()

