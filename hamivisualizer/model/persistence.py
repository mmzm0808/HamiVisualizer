"""模型 JSON 持久化: 完整自定义模型 (格点/跃迁/参数/边界/显示) 的存取.

格式 (version 1):
{
  "version": 1,
  "sites":  [{"x": 0.0, "y": 0.0, "sublattice": "A"}, ...],
  "hops":   [{"name": "t", "from_site": 0, "to_site": 1,
              "cell_offset": [0, 0], "amplitude": "-t",
              "phase_mode": "phase", "phase": "phi", "phase_sign": 1}, ...],
  "boundary": {"kind": "semi", "NX": 2, "NY": 2, "shape": "rectangle"},
  "order":   "cell",
  "params":  {"t": 1.0, "phi": 0.785, "omg": 1.0},
  "kx":      0.0,
  "symbolic": false,
  "smart":   true,
  "labels_bottom_up": true,
  "lattice_display": {"nn": true, "nnn": true, "ghosts": true, "cells": true}
}
boundary.shape 仅在 OBC 非矩形盘时写入，可取 rectangle/triangle/disk/hexagon；
缺省即为 rectangle，因此旧版 version 1 文件仍可无损读取。
跃迁表的 amplitude / phase 保存为**字符串** (如 "-t" / "phi" / "omg"),
与界面表格一致 —— 这是符号模式与数值模式共用的单一事实来源。
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

from .expression import parse_expression
from .boundary import BOUNDARY_SHAPES, SHAPE_RECTANGLE
from .hopping import KNOWN_PHASE_MODES

FORMAT_VERSION = 1
MAX_SITES = 256
MAX_HOPS = 4096
MAX_PARAMS = 64
# These are validation ceilings, not the old small-screen rendering limits.
# The UI accepts values throughout this range and its slider expands on
# demand. Actual feasible size remains hardware-dependent because a finite
# OBC Hamiltonian scales quadratically with the number of sites.
MAX_NX = 100_000
MAX_NY = 100_000
# A portable model is intentionally small.  This protects the GUI from
# allocating/parsing an unbounded user-supplied JSON document before the
# structural limits above can be checked.
MAX_MODEL_BYTES = 4 * 1024 * 1024


def _finite_number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} 必须是数值")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{path} 必须是有限数值")
    return value


def _integer(value: Any, path: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path} 必须是整数")
    if not minimum <= value <= maximum:
        raise ValueError(f"{path} 必须位于 {minimum}..{maximum}")
    return value


def validate_model_dict(obj: Any) -> dict:
    """Validate and normalize an untrusted JSON model document."""

    if not isinstance(obj, dict) or obj.get("version") != FORMAT_VERSION:
        raise ValueError(f"模型文件格式不支持: 需要 version={FORMAT_VERSION}")

    sites_in = obj.get("sites")
    if not isinstance(sites_in, list) or not 1 <= len(sites_in) <= MAX_SITES:
        raise ValueError(f"sites 必须包含 1..{MAX_SITES} 个格点")
    sites = []
    for i, site in enumerate(sites_in):
        if not isinstance(site, dict):
            raise ValueError(f"sites[{i}] 必须是对象")
        sub = site.get("sublattice", "")
        if sub is None:
            sub = ""
        if not isinstance(sub, str) or len(sub) > 32:
            raise ValueError(f"sites[{i}].sublattice 必须是短字符串")
        sites.append({
            "x": _finite_number(site.get("x"), f"sites[{i}].x"),
            "y": _finite_number(site.get("y"), f"sites[{i}].y"),
            "sublattice": sub,
        })

    cell_in = obj.get("cell")
    cell = None
    if cell_in is not None:
        if not isinstance(cell_in, dict):
            raise ValueError("cell 必须是对象或 null")
        if "a1" in cell_in or "a2" in cell_in:
            a1, a2 = cell_in.get("a1"), cell_in.get("a2")
            if not (isinstance(a1, list) and isinstance(a2, list) and len(a1) == len(a2) == 2):
                raise ValueError("cell.a1/a2 必须是两个二维数组")
            a1 = [_finite_number(v, f"cell.a1[{i}]") for i, v in enumerate(a1)]
            a2 = [_finite_number(v, f"cell.a2[{i}]") for i, v in enumerate(a2)]
            det = a1[0] * a2[1] - a1[1] * a2[0]
            if abs(det) < 1e-12:
                raise ValueError("cell.a1/a2 不可共线")
            for i, site in enumerate(sites):
                u = (site["x"] * a2[1] - site["y"] * a2[0]) / det
                v = (a1[0] * site["y"] - a1[1] * site["x"]) / det
                if not (-1e-10 <= u < 1 - 1e-10 and -1e-10 <= v < 1 - 1e-10):
                    raise ValueError(f"sites[{i}] 必须位于 cell.a1/a2 原胞内")
            cell = {"a1": a1, "a2": a2}
        else:
            lx = _finite_number(cell_in.get("Lx"), "cell.Lx")
            ly = _finite_number(cell_in.get("Ly"), "cell.Ly")
            if lx <= 0 or ly <= 0:
                raise ValueError("cell.Lx/cell.Ly 必须为正数")
            for i, site in enumerate(sites):
                if not (0 <= site["x"] < lx and 0 <= site["y"] < ly):
                    raise ValueError(
                        f"sites[{i}] 必须位于 [0,Lx)×[0,Ly)，当前 cell={lx}×{ly}"
                    )
            cell = {"Lx": lx, "Ly": ly}

    hops_in = obj.get("hops")
    if not isinstance(hops_in, list) or len(hops_in) > MAX_HOPS:
        raise ValueError(f"hops 必须是至多 {MAX_HOPS} 项的数组")
    hops = []
    for i, hop in enumerate(hops_in):
        if not isinstance(hop, dict):
            raise ValueError(f"hops[{i}] 必须是对象")
        name = hop.get("name", "t")
        if not isinstance(name, str) or not name.strip() or len(name) > 48:
            raise ValueError(f"hops[{i}].name 必须是非空短字符串")
        offset = hop.get("cell_offset", [0, 0])
        if not isinstance(offset, list) or len(offset) != 2:
            raise ValueError(f"hops[{i}].cell_offset 必须是两个整数")
        off_x = _integer(offset[0], f"hops[{i}].cell_offset[0]", -1000, 1000)
        off_y = _integer(offset[1], f"hops[{i}].cell_offset[1]", -1000, 1000)
        from_site = _integer(hop.get("from_site"), f"hops[{i}].from_site", 0, len(sites) - 1)
        to_site = _integer(hop.get("to_site"), f"hops[{i}].to_site", 0, len(sites) - 1)
        phase_mode = hop.get("phase_mode", "none")
        if phase_mode not in KNOWN_PHASE_MODES:
            raise ValueError(f"hops[{i}].phase_mode 无效: {phase_mode!r}")
        # Keep old documents loadable so the user can inspect and repair them,
        # but make the unsupported mode explicit instead of silently treating
        # it as a fixed phase.  The panel rejects it before a rebuild.
        sign = hop.get("phase_sign", 1)
        if sign not in {-1, 1} or isinstance(sign, bool):
            raise ValueError(f"hops[{i}].phase_sign 必须是 +1 或 -1")
        amplitude = hop.get("amplitude", "1.0")
        phase = hop.get("phase", "0")
        if not isinstance(amplitude, str) or not isinstance(phase, str):
            raise ValueError(f"hops[{i}] 的 amplitude/phase 必须是字符串")
        try:
            parse_expression(amplitude)
            parse_expression(phase, empty_value=0)
        except ValueError as exc:
            raise ValueError(f"hops[{i}] 表达式无效: {exc}") from None
        hops.append({
            "name": name.strip(),
            "from_site": from_site,
            "to_site": to_site,
            "cell_offset": [off_x, off_y],
            "amplitude": amplitude,
            "phase_mode": phase_mode,
            "phase": phase,
            "phase_sign": int(sign),
        })

    boundary_in = obj.get("boundary")
    if not isinstance(boundary_in, dict):
        raise ValueError("boundary 必须是对象")
    kind = boundary_in.get("kind")
    if kind not in {"semi", "obc"}:
        raise ValueError("boundary.kind 必须是 'semi' 或 'obc'")
    shape = boundary_in.get("shape", SHAPE_RECTANGLE)
    if shape not in BOUNDARY_SHAPES:
        raise ValueError(
            f"boundary.shape 必须是 {BOUNDARY_SHAPES!r} 之一"
        )
    boundary = {
        "kind": kind,
        "NX": _integer(boundary_in.get("NX", 2), "boundary.NX", 1, MAX_NX),
        "NY": _integer(boundary_in.get("NY", 2), "boundary.NY", 1, MAX_NY),
    }
    # Keep the version-1 rectangle document byte/schema compatible.  A
    # non-rectangular OBC shape is the only case that needs an extra field;
    # readers default a missing field to the historical rectangle.
    if shape != SHAPE_RECTANGLE:
        boundary["shape"] = shape
    shape_aspect = boundary_in.get("shape_aspect", 1.0)
    if (isinstance(shape_aspect, bool)
            or not isinstance(shape_aspect, (int, float))
            or not math.isfinite(float(shape_aspect))
            or float(shape_aspect) <= 0.0):
        raise ValueError("boundary.shape_aspect 必须是正的有限数值")
    if not math.isclose(float(shape_aspect), 1.0, rel_tol=0.0, abs_tol=1e-12):
        boundary["shape_aspect"] = float(shape_aspect)
    order = obj.get("order", "cell")
    if order not in {"cell", "site"}:
        raise ValueError("order 必须是 'cell' 或 'site'")
    params_in = obj.get("params", {})
    if not isinstance(params_in, dict) or len(params_in) > MAX_PARAMS:
        raise ValueError(f"params 必须是至多 {MAX_PARAMS} 项的对象")
    params = {}
    for name, value in params_in.items():
        if not isinstance(name, str) or not name.isidentifier() or name.startswith("_"):
            raise ValueError(f"非法参数名: {name!r}")
        params[name] = _finite_number(value, f"params.{name}")
    symbolic = obj.get("symbolic", False)
    smart = obj.get("smart", True)
    labels_bottom_up = obj.get("labels_bottom_up", True)
    if not all(isinstance(value, bool) for value in (
        symbolic, smart, labels_bottom_up,
    )):
        raise ValueError("symbolic/smart/labels_bottom_up 必须是布尔值")
    display_defaults = {"nn": True, "nnn": True, "ghosts": True, "cells": True}
    display_in = obj.get("lattice_display", {})
    if not isinstance(display_in, dict):
        raise ValueError("lattice_display 必须是对象")
    unknown_display = set(display_in) - set(display_defaults)
    if unknown_display:
        raise ValueError(
            f"lattice_display 包含未知选项: {sorted(unknown_display)!r}"
        )
    lattice_display = dict(display_defaults)
    for name, value in display_in.items():
        if not isinstance(value, bool):
            raise ValueError(f"lattice_display.{name} 必须是布尔值")
        lattice_display[name] = value
    return {
        "version": FORMAT_VERSION,
        "sites": sites,
        "cell": cell,
        "hops": hops,
        "boundary": boundary,
        "order": order,
        "params": params,
        "kx": _finite_number(obj.get("kx", 0.0), "kx"),
        "symbolic": symbolic,
        "smart": smart,
        "labels_bottom_up": labels_bottom_up,
        "lattice_display": lattice_display,
    }


def model_to_dict(
    site_rows: list[tuple],
    hop_rows: list[dict],
    boundary,
    order: str,
    params: dict,
    kx: float,
    symbolic: bool,
    smart: bool,
    cell: tuple[float, float] | None = None,
    cell_vectors: tuple[tuple[float, float], tuple[float, float]] | None = None,
    labels_bottom_up: bool = True,
    lattice_display: dict[str, bool] | None = None,
) -> dict:
    """把界面状态序列化为可 JSON 化的 dict."""
    obj = {
        "version": FORMAT_VERSION,
        "sites": [
            {"x": float(x), "y": float(y), "sublattice": sub or ""}
            for (x, y, sub) in site_rows
        ],
        "cell": (
            {"a1": [float(v) for v in cell_vectors[0]], "a2": [float(v) for v in cell_vectors[1]]}
            if cell_vectors is not None else
            {"Lx": float(cell[0]), "Ly": float(cell[1])}
            if cell is not None else None
        ),
        "hops": [
            {
                "name": h.get("name", "t"),
                "from_site": int(h.get("from_site", 0)),
                "to_site": int(h.get("to_site", 0)),
                "cell_offset": [int(h.get("off_x", 0)), int(h.get("off_y", 0))],
                "amplitude": str(h.get("amplitude", "1.0")),
                "phase_mode": str(h.get("phase_mode", "none")),
                "phase": str(h.get("phase", "0")),
                "phase_sign": int(h.get("phase_sign", 1)),
            }
            for h in hop_rows
        ],
        "boundary": {
            "kind": boundary.kind.value,
            "NX": int(boundary.NX),
            "NY": int(boundary.NY),
        },
        "order": order,
        "params": {k: float(v) for k, v in params.items()},
        "kx": float(kx),
        "symbolic": bool(symbolic),
        "smart": bool(smart),
        "labels_bottom_up": bool(labels_bottom_up),
        "lattice_display": dict(lattice_display or {
            "nn": True, "nnn": True, "ghosts": True, "cells": True,
        }),
    }
    shape = getattr(boundary, "shape", SHAPE_RECTANGLE)
    if shape != SHAPE_RECTANGLE:
        obj["boundary"]["shape"] = shape
    shape_aspect = float(getattr(boundary, "shape_aspect", 1.0))
    if not math.isclose(shape_aspect, 1.0, rel_tol=0.0, abs_tol=1e-12):
        obj["boundary"]["shape_aspect"] = shape_aspect
    return validate_model_dict(obj)


def save_model(path: str | Path, obj: dict) -> None:
    """Atomically write a validated UTF-8 JSON model."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    validated = validate_model_dict(obj)
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=target.parent,
            prefix=f".{target.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temp_name = stream.name
            json.dump(validated, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, target)
        temp_name = None
    finally:
        if temp_name is not None:
            Path(temp_name).unlink(missing_ok=True)


def load_model(path: str | Path) -> dict:
    """Read and fully validate an untrusted JSON model."""
    source = Path(path)
    try:
        size = source.stat().st_size
    except OSError:
        raise
    if size > MAX_MODEL_BYTES:
        raise ValueError(
            f"模型文件过大（最多 {MAX_MODEL_BYTES // (1024 * 1024)} MiB）"
        )
    with source.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    return validate_model_dict(obj)


def hop_dict_to_row(h: dict) -> list:
    """hop dict → 面板表格行 (与 HOP_COLS 顺序一致)."""
    off = h.get("cell_offset", [0, 0])
    return [
        h.get("name", "t"),
        h.get("from_site", 0),
        h.get("to_site", 0),
        int(off[0]),
        int(off[1]),
        h.get("amplitude", "1.0"),
        h.get("phase_mode", "none"),
        h.get("phase", "0"),
        h.get("phase_sign", 1),
    ]
