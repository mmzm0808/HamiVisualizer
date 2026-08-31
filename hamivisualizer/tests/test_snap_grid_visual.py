"""Regression checks for the non-interactive snap-grid decoration."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt

from hamivisualizer.view.lattice_view import (
    LatticeView,
    SNAP_HALO_SCALE,
    SNAP_HALO_GAP,
)
from hamivisualizer.view.rendermodel import LatticeSceneData


def test_snap_anchor_ring_is_compact_and_never_interactive():
    """光环只作定位提示，尺寸贴近格点且不扩大格点命中区。"""
    QApplication.instance() or QApplication([])
    scene = LatticeView()
    scene.set_edit_context(
        [(0.0, 0.0, "A"), (0.5, 0.5, "B")],
        cell_vectors=((1.0, 0.5), (0.0, 1.0)),
    )
    scene.set_edit_mode(True)
    scene.set_data(LatticeSceneData(
        sites=((0.0, 0.0, "1", "A"), (0.5, 0.5, "2", "B")),
    ))

    rings = [item for item in scene.items()
             if item.data(0) == "snap-grid-node"
             and item.data(1) == "site-anchor"]
    assert len(rings) == len(scene._edit_items)
    expected = max(scene._site_radius * SNAP_HALO_SCALE,
                   scene._site_radius + SNAP_HALO_GAP)
    assert all(item.rect().width() / 2 == pytest.approx(expected)
               for item in rings)
    assert all(item.acceptedMouseButtons() == Qt.NoButton for item in rings)

    # The editable disc's shape is unchanged and remains the only hit target.
    for site in scene._edit_items.values():
        assert site.shape().boundingRect().width() == site.rect().width()
        assert site.acceptedMouseButtons() != 0
        assert site.rect().width() / 2 == pytest.approx(scene._site_radius * 1.1)


def test_snap_anchor_ring_scales_with_large_spacing_without_shrinking_inside_site():
    """大尺度自定义晶格仍保持光环贴近且不小于格点本身。"""
    QApplication.instance() or QApplication([])
    scene = LatticeView()
    scene.set_edit_context(
        [(0.0, 0.0, "A"), (10.0, 0.0, "B")],
        cell_vectors=((10.0, 0.0), (0.0, 10.0)),
    )
    scene.set_edit_mode(True)
    scene.set_data(LatticeSceneData(
        sites=((0.0, 0.0, "1", "A"), (10.0, 0.0, "2", "B")),
        edges=((0, 1, "NN"),),
    ))
    rings = [item for item in scene.items()
             if item.data(1) == "site-anchor"]
    assert rings
    ring_radius = rings[0].rect().width() / 2
    assert ring_radius >= scene._site_radius
    assert ring_radius == pytest.approx(
        max(scene._site_radius * SNAP_HALO_SCALE,
            scene._site_radius + SNAP_HALO_GAP)
    )
