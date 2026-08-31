"""Shared hopping identity corpus for model and view wrappers."""

from __future__ import annotations

import pytest

from hamivisualizer.model.hopping_identity import (
    canonical_geometry,
    classify_amplitude,
    editor_parameter_key,
    editor_relation_key,
)
from hamivisualizer.view.control_panel import ControlPanel
from hamivisualizer.view.lattice_view import LatticeView


def _hop(**values):
    base = {
        "name": "t", "from_site": 0, "to_site": 1,
        "off_x": 1, "off_y": 0, "amplitude": "-t",
        "phase_mode": "none", "phase": "0", "phase_sign": 1,
    }
    base.update(values)
    return base


@pytest.mark.parametrize("forward, reverse", (
    (_hop(), _hop(from_site=1, to_site=0, off_x=-1)),
    (_hop(off_x=2, off_y=-1), _hop(from_site=1, to_site=0, off_x=-2, off_y=1)),
))
def test_forward_reverse_and_offset_forms_share_geometry(forward, reverse):
    assert canonical_geometry(forward) == canonical_geometry(reverse)
    assert editor_relation_key(forward) == editor_relation_key(reverse)


def test_cell_offset_schema_matches_off_columns():
    assert canonical_geometry(_hop()) == canonical_geometry(
        _hop(off_x=None, off_y=None, cell_offset=[1, 0])
    )


def test_phase_sign_is_not_identity():
    assert editor_relation_key(_hop(phase_sign=1)) == editor_relation_key(_hop(phase_sign=-1))
    assert editor_relation_key(_hop(phase="phi")) != editor_relation_key(_hop(phase="0"))


def test_parameter_families_and_unsupported_sources_are_isolated():
    assert editor_parameter_key(_hop(amplitude="-t")) != editor_parameter_key(_hop(amplitude="-t2"))
    assert editor_parameter_key(_hop(amplitude="t*u")) != editor_parameter_key(_hop(amplitude="t+1"))
    assert classify_amplitude("t*u").kind == "unsupported"


@pytest.mark.parametrize("payload", [None, {}, {"from_site": "bad", "to_site": 1,
                                                  "cell_offset": "bad"},
                                     {"from_site": True, "to_site": False,
                                      "cell_offset": ["x"]}])
def test_malformed_editor_payloads_have_stable_non_crashing_identity(payload):
    key = editor_relation_key(payload)
    assert key == editor_relation_key(payload)
    assert len(key) == 10


def test_amplitude_family_exposes_editability_and_stable_identity():
    linear = classify_amplitude("-4 * t")
    literal = classify_amplitude("1.25")
    unsupported = classify_amplitude("t*u")
    assert (linear.identity, linear.editable) == ("linear:t", True)
    assert (literal.identity, literal.editable) == ("literal", True)
    assert unsupported.identity.startswith("unsupported:")
    assert unsupported.editable is False
    assert unsupported.reason


def test_view_wrappers_delegate_exactly_to_shared_key():
    corpus = [
        _hop(), _hop(amplitude="4*t"), _hop(amplitude="t2", name="t2"),
        _hop(from_site=1, to_site=0, off_x=-1),
        _hop(phase_mode="phase", phase="phi"),
        _hop(amplitude="t*u"), _hop(off_x=2, off_y=1),
    ]
    for hop in corpus:
        assert LatticeView._editor_relation_key(hop) == editor_relation_key(hop)
        assert ControlPanel._editor_relation_key(hop) == editor_relation_key(hop)
