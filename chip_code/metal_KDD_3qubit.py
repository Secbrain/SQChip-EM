# -*- coding: utf-8 -*-
"""3-qubit dataset generator (qiskit-metal + pyEPR).

Layout: Three qubits in a row (Q1 - Q2 - Q3)
    Q1 <-- Bus_12 --> Q2 <-- Bus_23 --> Q3

Each qubit has its own readout resonator.
Bus couplers connect adjacent qubits: Q1-Q2 and Q2-Q3.
"""

from __future__ import annotations

import os
import sys
import csv
import copy
import importlib
import json
import math
import time
import gc
import subprocess
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any, Dict as TDict, List, Optional, Tuple


def _bootstrap_expected_python() -> None:
    if __name__ != "__main__":
        return
    expected = Path(__file__).resolve().parent.parent / "Scripts" / "python.exe"
    if not expected.is_file() or os.environ.get("METAL_3Q_BOOTSTRAPPED_PYTHON") == "1":
        return
    current = Path(sys.executable).resolve()
    if os.path.normcase(str(current)) == os.path.normcase(str(expected.resolve())):
        return
    os.environ["METAL_3Q_BOOTSTRAPPED_PYTHON"] = "1"
    rc = subprocess.call([str(expected), str(Path(__file__).resolve()), *sys.argv[1:]], env=os.environ.copy())
    raise SystemExit(int(rc))


_bootstrap_expected_python()


def _prefer_parent_env_site_packages() -> None:
    env_site = Path(__file__).resolve().parent.parent / "Lib" / "site-packages"
    if env_site.is_dir():
        env_site_s = str(env_site)
        if env_site_s not in sys.path:
            sys.path.insert(0, env_site_s)


_prefer_parent_env_site_packages()

import numpy as np
import pandas as pd
from shapely.geometry import LineString
from shapely.ops import unary_union

import metal_KDD as base
from qiskit_metal.qlibrary.core import QComponent
from qiskit_metal.qlibrary.qubits.transmon_pocket_6 import TransmonPocket6


# -------------------------
# layout
# -------------------------
@dataclass
class Layout3QParams:
    """3Q layout settings."""

    shared: base.LayoutParams = field(default_factory=base.LayoutParams)

    # Bus coupling parameters
    bus_width: str = "10um"
    bus_gap: str = "6um"
    bus_lead_start: str = "180um"
    bus_lead_end: str = "180um"
    bus_fillet: str = "35um"
    bus_spacing: str = "180um"
    bus_total_length: str = "5.2mm"
    bus_pad_width: str = "270um"
    bus_pad_height: str = "270um"
    bus_pad_gap: str = "3um"
    tee_offset_mm: float = 0.61
    readout_lane_dx_mm: float = 0.030
    q2_readout_lane_dx_mm: float = -0.15
    q13_lp_outward_offset_mm: float = 0.0
    swap_tee_ports: bool = True

    q1_ro_pad_w_um: Optional[float] = 420.0
    q2_ro_pad_w_um: Optional[float] = 400.0
    q3_ro_pad_w_um: Optional[float] = 420.0
    q1_ro_pad_h_um: Optional[float] = 420.0
    q2_ro_pad_h_um: Optional[float] = 400.0
    q3_ro_pad_h_um: Optional[float] = 420.0
    q1_ro_pad_gap_um: Optional[float] = 4.0
    q2_ro_pad_gap_um: Optional[float] = 4.0
    q3_ro_pad_gap_um: Optional[float] = 4.0
    q1_pocket_height_um: Optional[float] = 650.0
    q2_pocket_height_um: Optional[float] = 650.0
    q3_pocket_height_um: Optional[float] = 650.0

    ro1_serpentine_side: Optional[float] = -1.0
    ro2_serpentine_side: Optional[float] = -1.0
    ro3_serpentine_side: Optional[float] = 1.0
    ro2_serpentine_width_mm: Optional[float] = 1.20

    lp1_dx_mm: Optional[float] = 0.56
    lp2_dx_mm: Optional[float] = 0.56
    lp3_dx_mm: Optional[float] = 0.56

    ro1_total_length_mm: Optional[float] = 16.90
    ro2_total_length_mm: Optional[float] = 15.00
    ro3_total_length_mm: Optional[float] = 16.90
    ro1_tee_cap_gap_um: Optional[float] = 80.0
    ro2_tee_cap_gap_um: Optional[float] = 100.0
    ro3_tee_cap_gap_um: Optional[float] = 80.0
    ro1_tee_cap_width_um: Optional[float] = 0.28
    ro2_tee_cap_width_um: Optional[float] = 0.22
    ro3_tee_cap_width_um: Optional[float] = 0.28
    ro1_tee_cap_distance_um: Optional[float] = 170.0
    ro2_tee_cap_distance_um: Optional[float] = 170.0
    ro3_tee_cap_distance_um: Optional[float] = 170.0
    ro1_tee_finger_length_um: Optional[int] = 10
    ro2_tee_finger_length_um: Optional[int] = 10
    ro3_tee_finger_length_um: Optional[int] = 10
    ro1_tee_finger_count: Optional[int] = 1
    ro2_tee_finger_count: Optional[int] = 1
    ro3_tee_finger_count: Optional[int] = 1

    ro1_purcell_stub_length_mm: Optional[float] = 14.33
    ro2_purcell_stub_length_mm: Optional[float] = 14.95
    ro3_purcell_stub_length_mm: Optional[float] = 14.33
    ro1_purcell_stub_offset_mm: Optional[float] = 2.8
    ro2_purcell_stub_offset_mm: Optional[float] = 2.8
    ro3_purcell_stub_offset_mm: Optional[float] = 2.8
    ro1_purcell_stub_width_um: Optional[float] = 10.0
    ro2_purcell_stub_width_um: Optional[float] = 10.0
    ro3_purcell_stub_width_um: Optional[float] = 10.0
    ro1_purcell_stub_gap_um: Optional[float] = 6.0
    ro2_purcell_stub_gap_um: Optional[float] = 6.0
    ro3_purcell_stub_gap_um: Optional[float] = 6.0


class ManualCPWPath(QComponent):
    """Simple CPW path component with explicitly supplied centerline points."""

    component_metadata = base.Dict(short_name="cpw")
    default_options = base.Dict(
        points=[],
        trace_width="10um",
        trace_gap="6um",
        fillet="0um",
        layer="1",
    )

    def make(self):
        p = self.parse_options()
        pts = [(float(x), float(y)) for x, y in p.points]
        if len(pts) < 2:
            raise ValueError(f"{self.name} requires at least two path points")
        line = LineString(pts)
        self.add_qgeometry("path", {"trace": line}, width=p.trace_width, fillet=p.fillet, layer=p.layer)
        self.add_qgeometry(
            "path",
            {"cut": line},
            width=p.trace_width + 2 * p.trace_gap,
            fillet=p.fillet,
            layer=p.layer,
            subtract=True,
        )


def _env_int(name: str, default: int, *, lo: int, hi: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return int(default)
    try:
        v = int(float(raw))
    except Exception:
        return int(default)
    return int(max(int(lo), min(int(hi), v)))


def _env_float(name: str, default: float, *, lo: float, hi: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return float(default)
    try:
        v = float(raw)
    except Exception:
        return float(default)
    if not math.isfinite(v):
        return float(default)
    return float(max(float(lo), min(float(hi), v)))


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return bool(default)
    if raw in {"1", "true", "yes", "y", "on"}:
        return True
    if raw in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def _load_open_to_ground_cls() -> Any:
    module = importlib.import_module("qiskit_metal.qlibrary.terminations.open_to_ground")
    cls = getattr(module, "OpenToGround", None)
    if cls is None:
        raise ImportError("OpenToGround not found in qiskit_metal.qlibrary.terminations.open_to_ground")
    return cls


def _load_route_pathfinder_cls() -> Any:
    module = importlib.import_module("qiskit_metal.qlibrary.tlines.pathfinder")
    cls = getattr(module, "RoutePathfinder", None)
    if cls is None:
        raise ImportError("RoutePathfinder not found in qiskit_metal.qlibrary.tlines.pathfinder")
    return cls


def _load_route_straight_cls() -> Any:
    module = importlib.import_module("qiskit_metal.qlibrary.tlines.straight_path")
    cls = getattr(module, "RouteStraight", None)
    if cls is None:
        raise ImportError("RouteStraight not found in qiskit_metal.qlibrary.tlines.straight_path")
    return cls


def _layout_copy_with_overrides(src: base.LayoutParams, **overrides: Any) -> base.LayoutParams:
    out = copy.deepcopy(src)
    for key, value in overrides.items():
        if value is not None:
            setattr(out, key, value)
    return out


def _make_transmon(
    design,
    name: str,
    *,
    x_mm: float,
    y_mm: float,
    p: base.LayoutParams,
    lj_var: str,
    cj_var: str,
    orientation: float = 0,
    bus_pads: List[Tuple[str, int, int]] = None,
    bus_pad_width: Optional[str] = None,
    bus_pad_height: Optional[str] = None,
    bus_pad_gap: Optional[str] = None,
    readout_loc_W: int = +1,
    readout_loc_H: int = +1,
):
    """
    Create a TransmonPocket qubit with multiple bus pads for coupling.

    Parameters:
        bus_pads: List of (pad_name, loc_W, loc_H) tuples for bus connections.
                  e.g., [("bus_left", -1, -1), ("bus_right", +1, -1)]
        readout_loc_W, readout_loc_H: Location for readout pad
    """
    connection_pads = base.Dict(
        readout=base.Dict(
            loc_W=readout_loc_W,
            loc_H=readout_loc_H,
            pad_width=p.ro_pad_w,
            pad_height=p.ro_pad_h,
            pad_gap=p.ro_pad_gap,
        )
    )

    if bus_pads:
        for pad_name, loc_W, loc_H in bus_pads:
            connection_pads[pad_name] = base.Dict(
                loc_W=loc_W,
                loc_H=loc_H,
                pad_width=bus_pad_width if bus_pad_width is not None else p.ro_pad_w,
                pad_height=bus_pad_height if bus_pad_height is not None else p.ro_pad_h,
                pad_gap=bus_pad_gap if bus_pad_gap is not None else p.ro_pad_gap,
            )

    transmon_cls = TransmonPocket6 if readout_loc_W == 0 else base.TransmonPocket
    q = transmon_cls(
        design,
        name,
        options=base.Dict(
            pos_x=f"{x_mm}mm",
            pos_y=f"{y_mm}mm",
            orientation=str(orientation),
            pad_width=p.pad_width,
            pocket_height=p.pocket_height,
            connection_pads=connection_pads,
        ),
    )
    design.components[name].options["hfss_inductance"] = lj_var
    design.components[name].options["hfss_capacitance"] = cj_var
    return q


def _try_get_pin_xy(design, qname: str, pin: str) -> Tuple[float, float]:
    try:
        base.sanitize_center_readout_pin(design, qname, pin)
        x = float(design.components[qname].pins[pin]["middle"][0])
        y = float(design.components[qname].pins[pin]["middle"][1])
        return x, y
    except Exception:
        return 0.0, 0.0


def _add_connector_pad_subtract_cutouts(design, qnames: Tuple[str, ...] = ("Q1", "Q2", "Q3")) -> None:
    """Add local ground cutouts for TransmonPocket connector-pad metal outside rect_pk.

    Q3D AutoIdentifyNets can treat connector-pad metal outside the qubit pocket
    subtract as overlapping nearby nets. This mirrors the 2Q helper while keeping
    the qubit pocket size unchanged for HFSS.
    """
    try:
        poly = design.qgeometry.tables["poly"].copy()
    except Exception:
        return
    try:
        id_to_name = {comp.id: name for name, comp in design.components.items()}
        poly["comp_name"] = poly["component"].map(id_to_name)
        subtract_geoms = list(poly[poly["subtract"].astype(bool)]["geometry"])
        if not subtract_geoms:
            return
        subtract_union = unary_union(subtract_geoms)
    except Exception:
        return

    for qname in qnames:
        try:
            qcomp = design.components[qname]
        except Exception:
            continue
        additions: TDict[str, Any] = {}
        qrows = poly[(poly["comp_name"].eq(qname)) & (~poly["subtract"].astype(bool))]
        for _, row in qrows.iterrows():
            name = str(row.get("name", ""))
            if not name.endswith("_connector_pad"):
                continue
            try:
                outside = row["geometry"].difference(subtract_union)
            except Exception:
                continue
            if outside.is_empty or float(outside.area) <= 1e-12:
                continue
            additions[f"{name}_q3d_cutout"] = outside
        if additions:
            qcomp.add_qgeometry("poly", additions, subtract=True, chip="main")


def _add_q3d_readout_feed_markers(design, tee_names: Tuple[str, ...] = ("RO_TEE1", "RO_TEE2", "RO_TEE3")) -> None:
    """Add tiny metal marker polygons on tee feed-side plates for Q3D terminal naming.

    Q3D often merges the tee's second capacitor plate into the connected path,
    so AutoIdentifyNets does not expose a stable cap_body_1_* node.  A local
    marker placed on that feed-side plate gives the merged conductor a stable
    poly name without changing the coupling geometry in a meaningful way.
    """
    try:
        from shapely.geometry import box
    except Exception:
        return
    for tee_name in tee_names:
        try:
            tee = design.components[tee_name]
        except Exception:
            continue
        try:
            poly = design.qgeometry.tables["poly"]
            id_to_name = {comp.id: name for name, comp in design.components.items()}
            rows = poly[poly["component"].map(id_to_name).eq(tee_name)]
        except Exception:
            rows = []
        marker_geom = None
        for _, row in rows.iterrows():
            name = str(row.get("name", ""))
            if name not in {"cap_body_1", "cap_body_1_0", "cap_body_1_RO_TEE"}:
                continue
            geom = row.get("geometry")
            try:
                minx, miny, maxx, maxy = geom.bounds
                cx = 0.5 * (float(minx) + float(maxx))
                cy = 0.5 * (float(miny) + float(maxy))
                width = max(0.001, min(0.004, 0.5 * (float(maxx) - float(minx))))
                height = max(0.001, min(0.004, 0.5 * (float(maxy) - float(miny))))
                gap = _env_float("METAL_3Q_Q3D_FEED_MARKER_GAP_MM", 0.002, lo=0.0002, hi=0.05)
                if abs(float(maxx) - float(minx)) <= abs(float(maxy) - float(miny)):
                    mx = float(maxx) + gap + width / 2.0
                    my = cy
                else:
                    mx = cx
                    my = float(maxy) + gap + height / 2.0
                marker_geom = box(mx - width / 2.0, my - height / 2.0, mx + width / 2.0, my + height / 2.0)
                break
            except Exception:
                continue
        if marker_geom is not None:
            try:
                tee.add_qgeometry("poly", {f"feed_marker_{tee_name}": marker_geom}, subtract=False, chip="main")
            except Exception:
                pass


def _get_pin_frame(design, component: str, pin: str) -> Tuple[float, float, float, float, float, float]:
    """Return (x, y, nx, ny, tx, ty) for a pin in mm."""
    try:
        base.sanitize_center_readout_pin(design, component, pin)
        p = design.components[component].pins[pin]
        mid = p.get("middle", (0.0, 0.0))
        normal = p.get("normal", (0.0, 0.0))
        tangent = p.get("tangent", (0.0, 0.0))
        return (
            float(mid[0]),
            float(mid[1]),
            float(normal[0]),
            float(normal[1]),
            float(tangent[0]),
            float(tangent[1]),
        )
    except Exception:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0


def _tee_orientation_from_readout_normal(nx: float, ny: float) -> str:
    """Pick CapNInterdigitalTee orientation so second_end points back to qubit."""
    if abs(ny) >= abs(nx):
        return "0" if ny > 0 else "180"
    return "270" if nx > 0 else "90"


def _clamp_lp_dx_along_normal(
    *,
    coupler_x_mm: float,
    coupler_y_mm: float,
    nx: float,
    ny: float,
    p: base.LayoutParams,
    edge_margin_mm: float = 0.5,
) -> float:
    """Clamp lp_dx so LP stays within chip bounds when moving along (nx, ny)."""
    lp_dx = float(getattr(p, "lp_dx", 0.0) or 0.0)

    chip_x = base._value_to_mm(getattr(p, "chip_size_x", 0.0))
    chip_y = base._value_to_mm(getattr(p, "chip_size_y", 0.0))
    half_x = (chip_x / 2.0) if chip_x else 0.0
    half_y = (chip_y / 2.0) if chip_y else 0.0

    if abs(nx) >= abs(ny):
        if half_x > 0:
            max_dx = (
                (half_x - edge_margin_mm) - coupler_x_mm
                if nx > 0
                else coupler_x_mm - (-half_x + edge_margin_mm)
            )
            if max_dx > 0:
                lp_dx = min(lp_dx, float(max_dx))
    else:
        if half_y > 0:
            max_dy = (
                (half_y - edge_margin_mm) - coupler_y_mm
                if ny > 0
                else coupler_y_mm - (-half_y + edge_margin_mm)
            )
            if max_dy > 0:
                lp_dx = min(lp_dx, float(max_dy))

    if lp_dx <= 0:
        lp_dx = float(getattr(p, "lp_dx", 0.0) or 0.0)
    return float(lp_dx)


def _bump_trace_width(width: str, *, delta_um: float) -> str:
    mm = float(base._value_to_mm(width) or 0.0)
    if mm <= 0:
        return width
    mm2 = mm + (float(delta_um) / 1000.0)
    um2 = mm2 * 1000.0
    if abs(um2 - round(um2)) < 1e-9:
        return f"{int(round(um2))}um"
    return f"{um2:.3f}um"


def _pick_launchpad_params(
    *,
    coupler_x_mm: float,
    coupler_y_mm: float,
    tee_orientation: str,
    p: base.LayoutParams,
    edge_margin_mm: float = 0.5,
) -> Tuple[int, float]:
    """Pick an outward launchpad direction and clamp lp_dx to chip bounds."""
    ori = int(float(tee_orientation)) % 360
    lp_dx = float(getattr(p, "lp_dx", 0.0) or 0.0)

    chip_x = base._value_to_mm(getattr(p, "chip_size_x", 0.0))
    chip_y = base._value_to_mm(getattr(p, "chip_size_y", 0.0))
    half_x = (chip_x / 2.0) if chip_x else 0.0
    half_y = (chip_y / 2.0) if chip_y else 0.0

    if ori in (0, 180):
        s = -1 if float(coupler_x_mm) < 0 else 1
        lp_direction = s if ori == 0 else -s
        if half_x > 0:
            if ori == 0:
                max_dx = (
                    (half_x - edge_margin_mm) - coupler_x_mm
                    if lp_direction > 0
                    else coupler_x_mm - (-half_x + edge_margin_mm)
                )
            else:  # 180
                max_dx = (
                    coupler_x_mm - (-half_x + edge_margin_mm)
                    if lp_direction > 0
                    else (half_x - edge_margin_mm) - coupler_x_mm
                )
            if max_dx > 0:
                lp_dx = min(lp_dx, float(max_dx))
    else:
        s = -1 if float(coupler_y_mm) < 0 else 1
        lp_direction = s if ori == 90 else -s
        if half_y > 0:
            if ori == 90:
                max_dx = (
                    (half_y - edge_margin_mm) - coupler_y_mm
                    if lp_direction > 0
                    else coupler_y_mm - (-half_y + edge_margin_mm)
                )
            else:  # 270
                max_dx = (
                    coupler_y_mm - (-half_y + edge_margin_mm)
                    if lp_direction > 0
                    else (half_y - edge_margin_mm) - coupler_y_mm
                )
            if max_dx > 0:
                lp_dx = min(lp_dx, float(max_dx))

    if lp_dx <= 0:
        lp_dx = float(getattr(p, "lp_dx", 0.0) or 0.0)
    return int(lp_direction), float(lp_dx)


def _dedupe_points(points: List[Tuple[float, float]], *, eps: float = 1e-9) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    for x, y in points:
        pt = (float(x), float(y))
        if out and abs(out[-1][0] - pt[0]) <= eps and abs(out[-1][1] - pt[1]) <= eps:
            continue
        out.append(pt)
    return out


def _manual_readout_serpentine_points(
    *,
    start_xy: Tuple[float, float],
    end_xy: Tuple[float, float],
    target_len_mm: float,
    p: base.LayoutParams,
    suffix: str,
) -> List[Tuple[float, float]]:
    """Build a compact non-self-intersecting readout resonator centerline."""
    sx, sy = float(start_xy[0]), float(start_xy[1])
    ex, ey = float(end_xy[0]), float(end_xy[1])

    side_opt = getattr(p, "ro_serpentine_side", None)
    side = float(side_opt) if side_opt is not None else (-1.0 if ex < -0.25 else 1.0)
    inner_offset = _env_float("METAL_3Q_MANUAL_RO_INNER_OFFSET_MM", 0.070, lo=0.05, hi=0.7)
    if abs(ex) < 0.25:
        inner_offset = _env_float("METAL_3Q_MANUAL_RO_Q2_INNER_OFFSET_MM", inner_offset, lo=0.05, hi=0.7)
    if abs(side) < 1e-9:
        inner_x = ex
    else:
        inner_x = ex + side * inner_offset

    bottom_clearance = _env_float("METAL_3Q_MANUAL_RO_BOTTOM_CLEARANCE_MM", 0.24, lo=0.04, hi=0.5)
    if abs(side) < 1e-9:
        candidate_bottom_y = min(sy - bottom_clearance, ey - bottom_clearance)
    else:
        candidate_bottom_y = min(sy - 0.33, ey - bottom_clearance)
    max_y_span = _env_float("METAL_3Q_MANUAL_RO_MAX_Y_SPAN_MM", 0.54, lo=0.25, hi=1.5)
    bottom_y = max(candidate_bottom_y, sy - max_y_span)
    y_span = max(0.25, sy - bottom_y)

    raw_spacing = float(base._value_to_mm(getattr(p, "ro_spacing", "75um")) or 0.075)
    spacing = min(raw_spacing, _env_float("METAL_3Q_MANUAL_RO_SPACING_MM", 0.030, lo=0.024, hi=0.18))
    n_max = max(6, int(math.floor(y_span / max(spacing, 1e-6))) + 1)
    if n_max % 2:
        n_max -= 1
    n_max = max(6, n_max)

    half_x = float(base._value_to_mm(getattr(p, "chip_size_x", "0mm")) or 0.0) / 2.0
    edge_margin = _env_float("METAL_3Q_MANUAL_RO_EDGE_MARGIN_MM", 0.05, lo=0.04, hi=0.6)
    if abs(side) < 1e-9:
        center_width = float(getattr(p, "ro_center_serpentine_width_mm", 0.90) or 0.90)
        max_width = max(0.35, center_width)
    elif half_x > 0.0:
        outer_limit = side * (half_x - edge_margin)
        max_width = max(0.35, abs(outer_limit - inner_x))
    else:
        max_width = 1.55

    fixed_len = abs(sx - inner_x) + y_span + abs(inner_x - ex) + abs(ey - bottom_y)
    chosen_n = n_max
    chosen_width = (float(target_len_mm) - fixed_len) / float(chosen_n)
    for n_runs in range(n_max, 5, -2):
        width = (float(target_len_mm) - fixed_len) / float(n_runs)
        if 0.35 <= width <= max_width:
            chosen_n = n_runs
            chosen_width = width
            break

    chosen_width = max(0.35, min(max_width, chosen_width))
    if abs(side) < 1e-9:
        center_x = inner_x
        left_x = center_x - 0.5 * chosen_width
        right_x = center_x + 0.5 * chosen_width
        start_side_x = right_x if sx >= center_x else left_x
        other_side_x = left_x if start_side_x == right_x else right_x
        points: List[Tuple[float, float]] = [(sx, sy)]
        if abs(sx - start_side_x) > 1e-9:
            points.append((start_side_x, sy))
        current_x = start_side_x
        current_y = sy
        for k in range(chosen_n):
            next_x = other_side_x if abs(current_x - start_side_x) < 1e-9 else start_side_x
            points.append((next_x, current_y))
            if k < chosen_n - 1:
                current_y = sy - (k + 1) * spacing
                points.append((next_x, current_y))
            current_x = next_x
        if abs(current_y - bottom_y) > 1e-9:
            points.append((current_x, bottom_y))
        if abs(current_x - ex) > 1e-9:
            points.append((ex, bottom_y))
        points.append((ex, ey))

        line = LineString(_dedupe_points(points))
        if not line.is_simple:
            raise ValueError(f"manual readout path RO_RES{suffix} is self-intersecting")
        return _dedupe_points(points)

    outer_x = inner_x + (1.0 if abs(side) < 1e-9 else side) * chosen_width

    points: List[Tuple[float, float]] = [(sx, sy), (inner_x, sy)]
    current_x = inner_x
    current_y = sy
    for k in range(chosen_n):
        next_x = outer_x if abs(current_x - inner_x) < 1e-9 else inner_x
        points.append((next_x, current_y))
        if k < chosen_n - 1:
            current_y = sy - (k + 1) * spacing
            points.append((next_x, current_y))
        current_x = next_x

    if abs(current_x - inner_x) > 1e-9:
        points.append((inner_x, current_y))
        current_x = inner_x
    if abs(current_y - bottom_y) > 1e-9:
        points.append((inner_x, bottom_y))
    points.append((ex, bottom_y))
    points.append((ex, ey))

    line = LineString(_dedupe_points(points))
    if not line.is_simple:
        raise ValueError(f"manual readout path RO_RES{suffix} is self-intersecting")
    return _dedupe_points(points)


def _add_readout_chain(
    design,
    *,
    suffix: str,
    qname: str,
    q_pin: str,
    coupler_x_mm: float,
    coupler_y_mm: float,
    p: base.LayoutParams,
    tee_orientation: str = "0",
    lp_pos_x_mm: Optional[float] = None,
    lp_pos_y_mm: Optional[float] = None,
    lp_orientation: Optional[str] = None,
    prime_pin_override: Optional[str] = None,
    lp_direction: int = -1,
    lp_dx_mm: Optional[float] = None,
    ro_asymmetry: Optional[str] = None,
    ro_trace_width: Optional[str] = None,
    ro_trace_gap: Optional[str] = None,
    ro_start_pin: str = "second_end",
    swap_tee_ports: bool = True,
):
    """Add one launchpad + tee + long feed + short readout connection."""
    ori = int(float(tee_orientation)) % 360

    if lp_pos_x_mm is not None and lp_pos_y_mm is not None:
        lp_x = float(lp_pos_x_mm)
        lp_y = float(lp_pos_y_mm)
        if lp_orientation is not None:
            lp_ori = str(lp_orientation)
        else:
            dx = float(coupler_x_mm) - lp_x
            dy = float(coupler_y_mm) - lp_y
            if abs(dx) >= abs(dy):
                lp_ori = "0" if dx > 0 else "180"
            else:
                lp_ori = "90" if dy > 0 else "270"
    else:
        lp_dx = float(lp_dx_mm) if lp_dx_mm is not None else float(p.lp_dx)
        if ori == 0:
            lp_x = coupler_x_mm + lp_direction * lp_dx
            lp_y = coupler_y_mm
            lp_ori = "0" if lp_direction < 0 else "180"
        elif ori == 90:
            lp_x = coupler_x_mm
            lp_y = coupler_y_mm + lp_direction * lp_dx
            lp_ori = "90" if lp_direction < 0 else "270"
        elif ori == 180:
            lp_x = coupler_x_mm - lp_direction * lp_dx
            lp_y = coupler_y_mm
            lp_ori = "180" if lp_direction < 0 else "0"
        else:
            lp_x = coupler_x_mm
            lp_y = coupler_y_mm - lp_direction * lp_dx
            lp_ori = "270" if lp_direction < 0 else "90"

    if prime_pin_override in ("prime_start", "prime_end"):
        prime_pin = str(prime_pin_override)
    else:
        if ori == 0:
            prime_pin = "prime_start" if lp_x < coupler_x_mm else "prime_end"
        elif ori == 180:
            prime_pin = "prime_start" if lp_x > coupler_x_mm else "prime_end"
        elif ori == 90:
            prime_pin = "prime_start" if lp_y < coupler_y_mm else "prime_end"
        else:
            prime_pin = "prime_start" if lp_y > coupler_y_mm else "prime_end"

    lp = base._LAUNCHPAD_CLS(
        design,
        f"LP_RO{suffix}",
        options=base.Dict(
            pos_x=f"{lp_x}mm",
            pos_y=f"{lp_y}mm",
            orientation=lp_ori,
            trace_width=p.cpw_width,
            trace_gap=p.cpw_gap,
        ),
    )

    tee = base.CapNInterdigitalTee(
        design,
        f"RO_TEE{suffix}",
        options=base.Dict(
            pos_x=f"{coupler_x_mm}mm",
            pos_y=f"{coupler_y_mm}mm",
            orientation=tee_orientation,
            prime_width=p.prime_width,
            prime_gap=p.prime_gap,
            second_width=p.second_width,
            second_gap=p.second_gap,
            cap_gap=p.cap_gap,
            cap_width=p.cap_width,
            finger_length=p.finger_length,
            finger_count=p.finger_count,
            cap_distance=p.cap_distance,
        ),
    )

    def _best_prime_pin_toward_target(target_xy: Tuple[float, float]) -> str:
        tx, ty = float(target_xy[0]), float(target_xy[1])
        best_pin = "prime_start"
        best_score = -1e99
        for pin_name in ("prime_start", "prime_end"):
            try:
                pin = tee.pins[pin_name]
                mx, my = float(pin["middle"][0]), float(pin["middle"][1])
                nx, ny = float(pin["normal"][0]), float(pin["normal"][1])
            except Exception:
                continue
            vx, vy = tx - mx, ty - my
            score = nx * vx + ny * vy
            if score > best_score:
                best_score = score
                best_pin = pin_name
        return best_pin

    tee_prime_pin = prime_pin
    feed_end_pin = "second_end" if swap_tee_ports else tee_prime_pin
    if swap_tee_ports:
        qx_for_pin, qy_for_pin = _try_get_pin_xy(design, qname, q_pin)
        res_start_pin = _best_prime_pin_toward_target((qx_for_pin, qy_for_pin))
    else:
        res_start_pin = ro_start_pin

    purcell_stub_length_mm = float(getattr(p, "purcell_stub_length_mm", 0.0) or 0.0)
    purcell_stub_offset_mm = float(getattr(p, "purcell_stub_offset_mm", max(0.8, 0.6 * abs(float(lp_dx_mm or p.lp_dx)))) or max(0.8, 0.6 * abs(float(lp_dx_mm or p.lp_dx))))
    purcell_stub_width = getattr(p, "purcell_stub_width", p.cpw_width)
    purcell_stub_gap = getattr(p, "purcell_stub_gap", p.cpw_gap)
    filter_link_fillet = os.environ.get("METAL_3Q_FILTER_LINK_FILLET", "0um")
    filter_components: List[str] = []
    feed_target_component = tee.name
    feed_target_pin = feed_end_pin

    if purcell_stub_length_mm > 0.0:
        lp_delta = max(abs(float(lp_dx_mm if lp_dx_mm is not None else p.lp_dx)), 1e-9)
        filter_step_mm = min(max(0.6, purcell_stub_offset_mm), max(0.6, lp_delta - 0.25))
        filter_x = coupler_x_mm + ((lp_x - coupler_x_mm) / lp_delta) * filter_step_mm
        filter_y = coupler_y_mm + ((lp_y - coupler_y_mm) / lp_delta) * filter_step_mm

        filter_tee = base.CapNInterdigitalTee(
            design,
            f"RO_FILTER_TEE{suffix}",
            options=base.Dict(
                pos_x=f"{filter_x}mm",
                pos_y=f"{filter_y}mm",
                orientation=tee_orientation,
                prime_width=p.prime_width,
                prime_gap=p.prime_gap,
                second_width=p.second_width,
                second_gap=p.second_gap,
                cap_gap=p.cap_gap,
                cap_width=p.cap_width,
                finger_length=p.finger_length,
                finger_count=p.finger_count,
                cap_distance=p.cap_distance,
            ),
        )

        if ori == 0:
            filter_feed_pin = "prime_start" if lp_x < filter_x else "prime_end"
        elif ori == 180:
            filter_feed_pin = "prime_start" if lp_x > filter_x else "prime_end"
        elif ori == 90:
            filter_feed_pin = "prime_start" if lp_y < filter_y else "prime_end"
        else:
            filter_feed_pin = "prime_start" if lp_y > filter_y else "prime_end"
        filter_link_pin = "prime_end" if filter_feed_pin == "prime_start" else "prime_start"
        stub_start_pin = "second_end"

        try:
            stub_pin = filter_tee.pins[stub_start_pin]
            stub_x = float(stub_pin["middle"][0])
            stub_y = float(stub_pin["middle"][1])
            stub_nx = float(stub_pin["normal"][0])
            stub_ny = float(stub_pin["normal"][1])
        except Exception:
            stub_x = float(filter_x)
            stub_y = float(filter_y)
            if ori == 0:
                stub_nx, stub_ny = 1.0, 0.0
            elif ori == 180:
                stub_nx, stub_ny = -1.0, 0.0
            elif ori == 90:
                stub_nx, stub_ny = 0.0, 1.0
            else:
                stub_nx, stub_ny = 0.0, -1.0

        stub_gnd_offset_mm = max(0.45, min(1.0, 0.35 * purcell_stub_offset_mm))
        stub_gnd_x = stub_x + stub_nx * stub_gnd_offset_mm
        stub_gnd_y = stub_y + stub_ny * stub_gnd_offset_mm
        stub_ori = int(round((math.degrees(math.atan2(stub_ny, stub_nx)) + 360.0) % 360.0))

        open_to_ground_cls = _load_open_to_ground_cls()
        stub_gnd = open_to_ground_cls(
            design,
            f"RO_PSTUB_GND{suffix}",
            options=base.Dict(
                pos_x=f"{stub_gnd_x}mm",
                pos_y=f"{stub_gnd_y}mm",
                orientation=str(stub_ori),
            ),
        )

        base.RouteMeander(
            design,
            f"RO_PSTUB{suffix}",
            options=base.Dict(
                total_length=f"{purcell_stub_length_mm:.3f}mm",
                fillet=p.feed_fillet,
                trace_width=purcell_stub_width,
                trace_gap=purcell_stub_gap,
                meander=base.Dict(spacing=p.feed_spacing),
                lead=base.Dict(start_straight=p.feed_lead_start, end_straight=p.feed_lead_end),
                pin_inputs=base.Dict(
                    start_pin=base.Dict(component=filter_tee.name, pin=stub_start_pin),
                    end_pin=base.Dict(component=stub_gnd.name, pin="open"),
                ),
            ),
            type="CPW",
        )

        RoutePathfinder = _load_route_pathfinder_cls()
        RoutePathfinder(
            design,
            f"RO_FILTER_LINK{suffix}",
            options=base.Dict(
                fillet=filter_link_fillet,
                trace_width=p.cpw_width,
                trace_gap=p.cpw_gap,
                lead=base.Dict(start_straight=p.feed_lead_start, end_straight=p.feed_lead_end),
                pin_inputs=base.Dict(
                    start_pin=base.Dict(component=filter_tee.name, pin=filter_link_pin),
                    end_pin=base.Dict(component=tee.name, pin=feed_end_pin),
                ),
            ),
            type="CPW",
        )

        filter_components.extend([filter_tee.name, f"RO_FILTER_LINK{suffix}", f"RO_PSTUB{suffix}", stub_gnd.name])
        feed_target_component = filter_tee.name
        feed_target_pin = filter_feed_pin

    try:
        feed_start = lp.pins["tie"]
        feed_end = design.components[feed_target_component].pins[feed_target_pin]
        feed_points = [
            (float(feed_start["middle"][0]), float(feed_start["middle"][1])),
            (float(feed_end["middle"][0]), float(feed_end["middle"][1])),
        ]
        ManualCPWPath(
            design,
            f"RO_FEED{suffix}",
            options=base.Dict(
                points=feed_points,
                trace_width=p.cpw_width,
                trace_gap=p.cpw_gap,
                fillet="0um",
            ),
        )
    except Exception:
        RouteStraight = _load_route_straight_cls()
        RouteStraight(
            design,
            f"RO_FEED{suffix}",
            options=base.Dict(
                fillet="0um",
                trace_width=p.cpw_width,
                trace_gap=p.cpw_gap,
                lead=base.Dict(start_straight="0um", end_straight="0um"),
                pin_inputs=base.Dict(
                    start_pin=base.Dict(component=lp.name, pin="tie"),
                    end_pin=base.Dict(component=feed_target_component, pin=feed_target_pin),
                ),
            ),
            type="CPW",
        )

    base.sanitize_center_readout_pin(design, qname, q_pin)
    ro_w = ro_trace_width if ro_trace_width is not None else p.cpw_width
    ro_g = ro_trace_gap if ro_trace_gap is not None else p.cpw_gap
    ro_mm = base._mm_str_to_mm(p.ro_total_length)
    ro_points = []
    candidate_pins: List[str] = []
    for pin_name in (res_start_pin, "prime_end", "prime_start"):
        if pin_name in tee.pins and pin_name not in candidate_pins:
            candidate_pins.append(pin_name)
    for pin_name in candidate_pins:
        try:
            start_pin = tee.pins[pin_name]
            end_pin = design.components[qname].pins[q_pin]
            ro_points = _manual_readout_serpentine_points(
                start_xy=(float(start_pin["middle"][0]), float(start_pin["middle"][1])),
                end_xy=(float(end_pin["middle"][0]), float(end_pin["middle"][1])),
                target_len_mm=float(ro_mm),
                p=p,
                suffix=suffix,
            )
            res_start_pin = pin_name
            break
        except Exception:
            ro_points = []

    if ro_points:
        ManualCPWPath(
            design,
            f"RO_RES{suffix}",
            options=base.Dict(
                points=ro_points,
                trace_width=ro_w,
                trace_gap=ro_g,
                fillet=p.ro_fillet,
            ),
        )
    else:
        ro_mm = base._force_meander_total_len(ro_mm, p.ro_lead_start, p.ro_lead_end, p.ro_spacing)
        meander_ro = base.Dict(spacing=p.ro_spacing)
        if ro_asymmetry:
            meander_ro["asymmetry"] = ro_asymmetry
        base.RouteMeander(
            design,
            f"RO_RES{suffix}",
            options=base.Dict(
                total_length=f"{ro_mm:.3f}mm",
                fillet=p.ro_fillet,
                trace_width=ro_w,
                trace_gap=ro_g,
                meander=meander_ro,
                lead=base.Dict(start_straight=p.ro_lead_start, end_straight=p.ro_lead_end),
                pin_inputs=base.Dict(
                    start_pin=base.Dict(component=tee.name, pin=res_start_pin),
                    end_pin=base.Dict(component=qname, pin=q_pin),
                ),
            ),
            type="CPW",
    )

    return lp, tee, filter_components


def populate_design_3qubit(
    design,
    *,
    q1_x_mm: float,
    q1_y_mm: float,
    q2_x_mm: float,
    q2_y_mm: float,
    q3_x_mm: float,
    q3_y_mm: float,
    p3: Layout3QParams,
    lj_vars: Tuple[str, str, str] = ("Lj1", "Lj2", "Lj3"),
    cj_vars: Tuple[str, str, str] = ("Cj1", "Cj2", "Cj3"),
) -> None:
    """Populate planar design with Q1/Q2/Q3 in a row.

    Layout (horizontal arrangement):
        Q1 <-- Bus_12 --> Q2 <-- Bus_23 --> Q3

    Qubit pad configuration:
        - Q1: bus_right facing Q2, readout on top
        - Q2: bus_left facing Q1, bus_right facing Q3, readout on top
        - Q3: bus_left facing Q2, readout on top
    """

    design.overwrite_enabled = True
    p = p3.shared
    p_q1 = _layout_copy_with_overrides(
        p,
        pocket_height=(f"{float(p3.q1_pocket_height_um)}um" if p3.q1_pocket_height_um is not None else None),
        ro_pad_w=(f"{float(p3.q1_ro_pad_w_um)}um" if p3.q1_ro_pad_w_um is not None else None),
        ro_pad_h=(f"{float(p3.q1_ro_pad_h_um)}um" if p3.q1_ro_pad_h_um is not None else None),
        ro_pad_gap=(f"{float(p3.q1_ro_pad_gap_um)}um" if p3.q1_ro_pad_gap_um is not None else None),
    )
    p_q2 = _layout_copy_with_overrides(
        p,
        pocket_height=(f"{float(p3.q2_pocket_height_um)}um" if p3.q2_pocket_height_um is not None else None),
        ro_pad_w=(f"{float(p3.q2_ro_pad_w_um)}um" if p3.q2_ro_pad_w_um is not None else None),
        ro_pad_h=(f"{float(p3.q2_ro_pad_h_um)}um" if p3.q2_ro_pad_h_um is not None else None),
        ro_pad_gap=(f"{float(p3.q2_ro_pad_gap_um)}um" if p3.q2_ro_pad_gap_um is not None else None),
    )
    p_q3 = _layout_copy_with_overrides(
        p,
        pocket_height=(f"{float(p3.q3_pocket_height_um)}um" if p3.q3_pocket_height_um is not None else None),
        ro_pad_w=(f"{float(p3.q3_ro_pad_w_um)}um" if p3.q3_ro_pad_w_um is not None else None),
        ro_pad_h=(f"{float(p3.q3_ro_pad_h_um)}um" if p3.q3_ro_pad_h_um is not None else None),
        ro_pad_gap=(f"{float(p3.q3_ro_pad_gap_um)}um" if p3.q3_ro_pad_gap_um is not None else None),
    )
    p_ro = {
        1: _layout_copy_with_overrides(
            p_q1,
            lp_dx=p3.lp1_dx_mm,
            ro_total_length=(f"{float(p3.ro1_total_length_mm)}mm" if p3.ro1_total_length_mm is not None else None),
            cap_gap=(f"{float(p3.ro1_tee_cap_gap_um)}um" if p3.ro1_tee_cap_gap_um is not None else None),
            cap_width=(f"{float(p3.ro1_tee_cap_width_um)}um" if p3.ro1_tee_cap_width_um is not None else None),
            cap_distance=(f"{float(p3.ro1_tee_cap_distance_um)}um" if p3.ro1_tee_cap_distance_um is not None else None),
            finger_length=(f"{int(p3.ro1_tee_finger_length_um)}um" if p3.ro1_tee_finger_length_um is not None else None),
            finger_count=(str(int(p3.ro1_tee_finger_count)) if p3.ro1_tee_finger_count is not None else None),
            ro_serpentine_side=p3.ro1_serpentine_side,
            ro_center_serpentine_width_mm=p3.ro2_serpentine_width_mm,
            purcell_stub_length_mm=p3.ro1_purcell_stub_length_mm,
            purcell_stub_offset_mm=p3.ro1_purcell_stub_offset_mm,
            purcell_stub_width=(f"{float(p3.ro1_purcell_stub_width_um)}um" if p3.ro1_purcell_stub_width_um is not None else None),
            purcell_stub_gap=(f"{float(p3.ro1_purcell_stub_gap_um)}um" if p3.ro1_purcell_stub_gap_um is not None else None),
        ),
        2: _layout_copy_with_overrides(
            p_q2,
            lp_dx=p3.lp2_dx_mm,
            ro_total_length=(f"{float(p3.ro2_total_length_mm)}mm" if p3.ro2_total_length_mm is not None else None),
            cap_gap=(f"{float(p3.ro2_tee_cap_gap_um)}um" if p3.ro2_tee_cap_gap_um is not None else None),
            cap_width=(f"{float(p3.ro2_tee_cap_width_um)}um" if p3.ro2_tee_cap_width_um is not None else None),
            cap_distance=(f"{float(p3.ro2_tee_cap_distance_um)}um" if p3.ro2_tee_cap_distance_um is not None else None),
            finger_length=(f"{int(p3.ro2_tee_finger_length_um)}um" if p3.ro2_tee_finger_length_um is not None else None),
            finger_count=(str(int(p3.ro2_tee_finger_count)) if p3.ro2_tee_finger_count is not None else None),
            ro_serpentine_side=p3.ro2_serpentine_side,
            ro_center_serpentine_width_mm=p3.ro2_serpentine_width_mm,
            purcell_stub_length_mm=p3.ro2_purcell_stub_length_mm,
            purcell_stub_offset_mm=p3.ro2_purcell_stub_offset_mm,
            purcell_stub_width=(f"{float(p3.ro2_purcell_stub_width_um)}um" if p3.ro2_purcell_stub_width_um is not None else None),
            purcell_stub_gap=(f"{float(p3.ro2_purcell_stub_gap_um)}um" if p3.ro2_purcell_stub_gap_um is not None else None),
        ),
        3: _layout_copy_with_overrides(
            p_q3,
            lp_dx=p3.lp3_dx_mm,
            ro_total_length=(f"{float(p3.ro3_total_length_mm)}mm" if p3.ro3_total_length_mm is not None else None),
            cap_gap=(f"{float(p3.ro3_tee_cap_gap_um)}um" if p3.ro3_tee_cap_gap_um is not None else None),
            cap_width=(f"{float(p3.ro3_tee_cap_width_um)}um" if p3.ro3_tee_cap_width_um is not None else None),
            cap_distance=(f"{float(p3.ro3_tee_cap_distance_um)}um" if p3.ro3_tee_cap_distance_um is not None else None),
            finger_length=(f"{int(p3.ro3_tee_finger_length_um)}um" if p3.ro3_tee_finger_length_um is not None else None),
            finger_count=(str(int(p3.ro3_tee_finger_count)) if p3.ro3_tee_finger_count is not None else None),
            ro_serpentine_side=p3.ro3_serpentine_side,
            ro_center_serpentine_width_mm=p3.ro2_serpentine_width_mm,
            purcell_stub_length_mm=p3.ro3_purcell_stub_length_mm,
            purcell_stub_offset_mm=p3.ro3_purcell_stub_offset_mm,
            purcell_stub_width=(f"{float(p3.ro3_purcell_stub_width_um)}um" if p3.ro3_purcell_stub_width_um is not None else None),
            purcell_stub_gap=(f"{float(p3.ro3_purcell_stub_gap_um)}um" if p3.ro3_purcell_stub_gap_um is not None else None),
        ),
    }

    design.variables.update(base.Dict(cpw_width=p.cpw_width, cpw_gap=p.cpw_gap))
    design.chips.main.size["size_x"] = p.chip_size_x
    design.chips.main.size["size_y"] = p.chip_size_y

    try:
        if hasattr(design.chips.main, "material") and isinstance(design.chips.main.material, dict):
            design.chips.main.material["tan_delta"] = float(base.TAN_DELTA_MAIN)
        elif hasattr(design.chips.main, "material"):
            setattr(design.chips.main.material, "tan_delta", float(base.TAN_DELTA_MAIN))
    except Exception:
        pass

    # Q1/Q2/Q3 use top readout pads and lower side bus pads. This keeps
    # readout routing in the upper channel and inter-qubit buses below.
    _make_transmon(
        design,
        "Q1",
        x_mm=q1_x_mm,
        y_mm=q1_y_mm,
        p=p_q1,
        lj_var=lj_vars[0],
        cj_var=cj_vars[0],
        bus_pads=[("bus_right", +1, -1)],
        bus_pad_width=p3.bus_pad_width,
        bus_pad_height=p3.bus_pad_height,
        bus_pad_gap=p3.bus_pad_gap,
        readout_loc_W=0,
        readout_loc_H=+1,
    )

    # Q2: center, bus on both left and right sides
    _make_transmon(
        design,
        "Q2",
        x_mm=q2_x_mm,
        y_mm=q2_y_mm,
        p=p_q2,
        lj_var=lj_vars[1],
        cj_var=cj_vars[1],
        bus_pads=[("bus_left", -1, -1), ("bus_right", +1, -1)],
        bus_pad_width=p3.bus_pad_width,
        bus_pad_height=p3.bus_pad_height,
        bus_pad_gap=p3.bus_pad_gap,
        readout_loc_W=0,
        readout_loc_H=+1,
    )

    # Q3: rightmost, bus on left side
    _make_transmon(
        design,
        "Q3",
        x_mm=q3_x_mm,
        y_mm=q3_y_mm,
        p=p_q3,
        lj_var=lj_vars[2],
        cj_var=cj_vars[2],
        bus_pads=[("bus_left", -1, -1)],
        bus_pad_width=p3.bus_pad_width,
        bus_pad_height=p3.bus_pad_height,
        bus_pad_gap=p3.bus_pad_gap,
        readout_loc_W=0,
        readout_loc_H=+1,
    )

    design.rebuild()

    # Add readout chains in a single aligned upper row. Q1/Q3 use shallow
    # outward readout lanes so the routing stays compact but does not overlap
    # the qubit pockets.
    tee_offset = max(float(p3.tee_offset_mm), 0.34)
    readout_lane_dx = max(abs(float(getattr(p3, "readout_lane_dx_mm", 0.03) or 0.03)), 0.03)
    q_centers = {
        1: (float(q1_x_mm), float(q1_y_mm)),
        2: (float(q2_x_mm), float(q2_y_mm)),
        3: (float(q3_x_mm), float(q3_y_mm)),
    }
    lp_dx_values = {
        1: max(abs(float(getattr(p3, "lp1_dx_mm", 0.56) or 0.56)), 0.54),
        2: max(abs(float(getattr(p3, "lp2_dx_mm", 0.56) or 0.56)), 0.54),
        3: max(abs(float(getattr(p3, "lp3_dx_mm", 0.56) or 0.56)), 0.54),
    }
    q2_lane_raw = getattr(p3, "q2_readout_lane_dx_mm", -0.15)
    q2_lane_dx = -0.15 if q2_lane_raw is None else float(q2_lane_raw)
    ro_lane_dx = {
        1: -readout_lane_dx,
        2: q2_lane_dx,
        3: readout_lane_dx,
    }
    readout_pin_y = []
    for q in ("Q1", "Q2", "Q3"):
        _px, py, _nx, _ny, _tx, _ty = _get_pin_frame(design, q, "readout")
        readout_pin_y.append(float(py))
    tee_row_y = max(readout_pin_y) + tee_offset
    lp_row_y = tee_row_y + max(lp_dx_values.values())
    for i in range(1, 4):
        q = f"Q{i}"
        _px, _py, _nx, _ny, _tx, _ty = _get_pin_frame(design, q, "readout")
        ori = "180"
        cx = q_centers[i][0] + ro_lane_dx[i]
        cy = tee_row_y
        q13_lp_outward = max(float(getattr(p3, "q13_lp_outward_offset_mm", 0.0) or 0.0), 0.0)
        if i == 1:
            lp_x = cx - q13_lp_outward
        elif i == 3:
            lp_x = cx + q13_lp_outward
        else:
            lp_x = cx
        lp_y = lp_row_y
        lp_dx = max(abs(lp_y - cy), 0.54)

        prime_pin_override = "prime_start" if float(cx) < 0 else "prime_end"

        _add_readout_chain(
            design,
            suffix=str(i),
            qname=q,
            q_pin="readout",
            coupler_x_mm=cx,
            coupler_y_mm=cy,
            p=p_ro[i],
            tee_orientation=ori,
            lp_pos_x_mm=lp_x,
            lp_pos_y_mm=lp_y,
            lp_orientation="270",
            prime_pin_override=prime_pin_override,
            lp_direction=-1,
            lp_dx_mm=lp_dx,
            swap_tee_ports=p3.swap_tee_ports,
        )

    # Bus couplers: keep the inter-qubit channel as a short, deterministic
    # lower route. The long meandered bus made the compact layout visually
    # noisy and harder to inspect before parameter tuning.
    RoutePathfinder = _load_route_pathfinder_cls()
    RoutePathfinder(
        design,
        "Bus_12",
        options=base.Dict(
            fillet=p3.bus_fillet,
            trace_width=p3.bus_width,
            trace_gap=p3.bus_gap,
            lead=base.Dict(start_straight=p3.bus_lead_start, end_straight=p3.bus_lead_end),
            pin_inputs=base.Dict(
                start_pin=base.Dict(component="Q1", pin="bus_right"),
                end_pin=base.Dict(component="Q2", pin="bus_left"),
            ),
        ),
        type="CPW",
    )

    RoutePathfinder(
        design,
        "Bus_23",
        options=base.Dict(
            fillet=p3.bus_fillet,
            trace_width=p3.bus_width,
            trace_gap=p3.bus_gap,
            lead=base.Dict(start_straight=p3.bus_lead_start, end_straight=p3.bus_lead_end),
            pin_inputs=base.Dict(
                start_pin=base.Dict(component="Q2", pin="bus_right"),
                end_pin=base.Dict(component="Q3", pin="bus_left"),
            ),
        ),
        type="CPW",
    )

    design.rebuild()
    if _env_bool("METAL_3Q_Q3D_CONNECTOR_PAD_CUTOUTS", True):
        _add_connector_pad_subtract_cutouts(design)
    if _env_bool("METAL_3Q_Q3D_FEED_MARKERS", False):
        _add_q3d_readout_feed_markers(design)


# -------------------------
# payload
# -------------------------
def _empty_readout_block() -> TDict[str, Any]:
    return {
        "f_GHz": None,
        "K_MHz": None,
        "modes_f_GHz": None,
        "picked_qubit_mode_index": None,
        "picked_res_mode_index": None,
        "Qi": None,
        "kappa_i_over_2pi_Hz": None,
        "external": {
            "Cin_fF": None,
            "pair": None,
            "res_node": None,
            "feed_node": None,
            "kappa_Hz": None,
            "kappa_over_2pi_Hz": None,
            "Qe": None,
        },
        "kappa_over_2pi_Hz": None,
        "Q_loaded": None,
        "warnings": [],
        "T1_photon_us": None,
        "kappa_i_over_kappa_e": None,
    }


def _empty_qubit_block() -> TDict[str, Any]:
    return {
        "Lj_H": None,
        "Cj_fF": None,
        "C_eff_fF": None,
        "f01_epr_GHz": None,
        "alpha_epr_MHz": None,
        "chi_MHz": None,
        "dispersive": {"chi_GHz": None, "Delta_GHz": None, "g_GHz": None},
        "energies": {"Ec_GHz": None, "Ej_GHz": None, "Ej_over_Ec": None},
        "f01_transmon_GHz": None,
        "Q_dielectric_main": None,
        "T1_dielectric_us": None,
        "T1_purcell_us": None,
        "T1_est_us": None,
        "T2_est_us": None,
    }


def build_chip_summary_3q(*, sample_id: str) -> TDict[str, Any]:
    return {
        "meta": {
            "created_utc": base.now_utc_iso(),
            "updated_utc": None,
            "sample_id": sample_id,
            "units": {"f": "GHz", "kappa": "Hz", "C": "fF", "K": "MHz", "T": "us"},
        },
        "resonators": {
            "readout1": _empty_readout_block(),
            "readout2": _empty_readout_block(),
            "readout3": _empty_readout_block(),
        },
        "qubits": {
            "Q1": _empty_qubit_block(),
            "Q2": _empty_qubit_block(),
            "Q3": _empty_qubit_block(),
        },
        "q3d": {"internal": {}, "external": {}},
        "chip": {
            "T1_qubit1_est_us": None,
            "T2_qubit1_est_us": None,
            "T1_qubit2_est_us": None,
            "T2_qubit2_est_us": None,
            "T1_qubit3_est_us": None,
            "T2_qubit3_est_us": None,
            "chi12_MHz": None,
            "chi23_MHz": None,
            "chi13_MHz": None,
            "chi1_over_kappa1": None,
            "chi2_over_kappa2": None,
            "chi3_over_kappa3": None,
        },
        "status": "init",
    }


def _estimate_T1T2_3q(payload: TDict[str, Any]) -> None:
    for i in [1, 2, 3]:
        qname = f"Q{i}"
        roname = f"readout{i}"
        ro = payload.get("resonators", {}).get(roname, {}) or {}
        q = payload.get("qubits", {}).get(qname, {}) or {}
        fq_GHz = q.get("f01_epr_GHz")

        kappa_over_2pi = ro.get("kappa_over_2pi_Hz")
        if kappa_over_2pi:
            try:
                T1_photon_s = 1.0 / (2.0 * math.pi * float(kappa_over_2pi))
                ro["T1_photon_us"] = base._safe_us(T1_photon_s)
            except Exception:
                pass

        T1_diel_s = None
        Qq = q.get("Q_dielectric_main")
        if Qq is not None and fq_GHz is not None:
            try:
                fq_hz = float(fq_GHz) * 1e9
                Qqf = float(Qq)
                if fq_hz > 0 and Qqf > 0:
                    T1_diel_s = Qqf / (2.0 * math.pi * fq_hz)
                    q["T1_dielectric_us"] = base._safe_us(T1_diel_s)
            except Exception:
                pass

        T1_purcell_s = None
        g_GHz = (q.get("dispersive") or {}).get("g_GHz")
        Delta_GHz = (q.get("dispersive") or {}).get("Delta_GHz")
        if g_GHz is not None and Delta_GHz is not None and kappa_over_2pi:
            try:
                g_hz = float(g_GHz) * 1e9
                d_hz = float(Delta_GHz) * 1e9
                k_over_2pi_raw = float(kappa_over_2pi)
                k_over_2pi = k_over_2pi_raw
                pf = ro.get("purcell_filter") if isinstance(ro.get("purcell_filter"), dict) else {}
                if pf and fq_GHz is not None:
                    try:
                        fr_hz = float(ro.get("f_GHz")) * 1e9
                        fq_hz = float(fq_GHz) * 1e9
                        notch_hz = float(pf.get("notch_frequency_GHz")) * 1e9
                        bw_hz = float(pf.get("bandwidth_GHz")) * 1e9
                        if fr_hz > 0 and fq_hz > 0 and notch_hz > 0 and bw_hz > 0:
                            readout_det = (fr_hz - notch_hz) / bw_hz
                            qubit_det = (fq_hz - notch_hz) / bw_hz
                            readout_trans = readout_det * readout_det / (1.0 + readout_det * readout_det)
                            qubit_trans = qubit_det * qubit_det / (1.0 + qubit_det * qubit_det)
                            if readout_trans > 1e-9:
                                suppression = max(1.0e-6, min(1.0, qubit_trans / readout_trans))
                                k_over_2pi = k_over_2pi_raw * suppression
                                pf["kappa_over_2pi_at_readout_Hz"] = k_over_2pi_raw
                                pf["kappa_eff_over_2pi_at_qubit_Hz"] = k_over_2pi
                                pf["suppression_at_qubit"] = suppression
                    except Exception:
                        k_over_2pi = k_over_2pi_raw
                if abs(d_hz) > 0 and g_hz > 0 and k_over_2pi > 0:
                    Gamma_p = (g_hz / d_hz) ** 2 * (2.0 * math.pi * k_over_2pi)
                    if Gamma_p > 0:
                        T1_purcell_s = 1.0 / Gamma_p
                        q["T1_purcell_us"] = base._safe_us(T1_purcell_s)
            except Exception:
                pass

        inv = 0.0
        n = 0
        for T in [T1_diel_s, T1_purcell_s]:
            if T is not None and np.isfinite(T) and T > 0:
                inv += 1.0 / float(T)
                n += 1

        if n > 0 and inv > 0:
            T1_est_s = 1.0 / inv
            q["T1_est_us"] = base._safe_us(T1_est_s)
            q["T2_est_us"] = base._safe_us(2.0 * T1_est_s)

            payload["chip"][f"T1_qubit{i}_est_us"] = q["T1_est_us"]
            payload["chip"][f"T2_qubit{i}_est_us"] = q["T2_est_us"]


def _postprocess_one_readout(ro: TDict[str, Any]) -> None:
    if ro.get("kappa_i_over_2pi_Hz") is None and ro.get("Qi") and ro.get("f_GHz"):
        ro["kappa_i_over_2pi_Hz"] = float(ro["f_GHz"]) * 1e9 / float(ro["Qi"])

    ki = ro.get("kappa_i_over_2pi_Hz")
    ke = (ro.get("external") or {}).get("kappa_over_2pi_Hz")

    if ro.get("kappa_over_2pi_Hz") is None:
        if ki is not None and ke is not None:
            ro["kappa_over_2pi_Hz"] = float(ki) + float(ke)
        elif ki is not None:
            ro["kappa_over_2pi_Hz"] = float(ki)
        elif ke is not None:
            ro["kappa_over_2pi_Hz"] = float(ke)

    if ro.get("Q_loaded") is None and ro.get("f_GHz") and ro.get("kappa_over_2pi_Hz"):
        ro["Q_loaded"] = (float(ro["f_GHz"]) * 1e9) / float(ro["kappa_over_2pi_Hz"])

    if ki is not None and ke is not None and float(ke) > 0:
        ro["kappa_i_over_kappa_e"] = float(ki) / float(ke)


def _purcell_filter_meta_for_readout_3q(layout_params: Optional[Layout3QParams], idx: int) -> TDict[str, Any]:
    if layout_params is None:
        return {}
    length_mm = float(getattr(layout_params, f"ro{idx}_purcell_stub_length_mm", 0.0) or 0.0)
    offset_mm = float(getattr(layout_params, f"ro{idx}_purcell_stub_offset_mm", 0.0) or 0.0)
    width_um = float(getattr(layout_params, f"ro{idx}_purcell_stub_width_um", 0.0) or 0.0)
    gap_um = float(getattr(layout_params, f"ro{idx}_purcell_stub_gap_um", 0.0) or 0.0)
    if length_mm <= 0.0:
        return {}
    notch_ghz = 75.0 / max(length_mm, 1.0e-9)
    bandwidth_ghz = max(0.05, 0.12 * notch_ghz)
    return {
        "model": "single_shunt_quarter_wave_notch_approx",
        "stub_length_mm": length_mm,
        "stub_offset_mm": offset_mm,
        "stub_width_um": width_um,
        "stub_gap_um": gap_um,
        "notch_frequency_GHz": notch_ghz,
        "bandwidth_GHz": bandwidth_ghz,
    }


def _postprocess_one_qubit(q: TDict[str, Any]) -> None:
    C_eff_fF = q.get("C_eff_fF")
    Lj_H = q.get("Lj_H")
    EC_hz = base.ec_over_h_hz_from_c(float(C_eff_fF) * 1e-15) if C_eff_fF is not None else None
    EJ_hz = base.ej_over_h_hz_from_lj(float(Lj_H)) if Lj_H is not None else None

    if EC_hz:
        q["energies"]["Ec_GHz"] = EC_hz / 1e9
    if EJ_hz:
        q["energies"]["Ej_GHz"] = EJ_hz / 1e9
    if EC_hz and EJ_hz:
        q["energies"]["Ej_over_Ec"] = EJ_hz / EC_hz
        f01 = base.transmon_f01_hz(EJ_hz, EC_hz)
        q["f01_transmon_GHz"] = (f01 / 1e9) if f01 else None


def postprocess_3q(payload: TDict[str, Any]) -> TDict[str, Any]:
    payload["meta"]["updated_utc"] = base.now_utc_iso()

    for roname in ["readout1", "readout2", "readout3"]:
        _postprocess_one_readout(payload["resonators"][roname])
    for qname in ["Q1", "Q2", "Q3"]:
        _postprocess_one_qubit(payload["qubits"][qname])

    # chi/kappa for each local pair
    def _chi_over_kappa(q: TDict[str, Any], ro: TDict[str, Any]) -> Optional[float]:
        chi_mhz = q.get("chi_MHz")
        kappa_over_2pi_hz = ro.get("kappa_over_2pi_Hz")
        if chi_mhz is None or not kappa_over_2pi_hz:
            return None
        try:
            return (abs(float(chi_mhz)) * 1e6) / float(kappa_over_2pi_hz)
        except Exception:
            return None

    for i in [1, 2, 3]:
        payload["chip"][f"chi{i}_over_kappa{i}"] = _chi_over_kappa(
            payload["qubits"][f"Q{i}"],
            payload["resonators"][f"readout{i}"],
        )

    _estimate_T1T2_3q(payload)
    return payload


def _as_float_or_none(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        f = float(v)
        return f if math.isfinite(f) else None
    except Exception:
        return None


def metrics_for_qubit_3q(payload: TDict[str, Any], idx: int) -> TDict[str, Optional[float]]:
    q_block = payload.get("qubits") or payload.get("qubit") or {}
    q = q_block.get(f"Q{idx}", {}) or {}
    ro = payload.get("resonators", {}).get(f"readout{idx}", {}) or {}
    q3d_ext = (((payload.get("q3d") or {}).get("external") or {}).get("resonators") or {}).get(f"readout{idx}", {})
    ext = ro.get("external") or q3d_ext or {}
    ki_hz = _as_float_or_none(ro.get("kappa_i_over_2pi_Hz"))
    ke_hz = _as_float_or_none(ext.get("kappa_over_2pi_Hz"))
    kt_hz = _as_float_or_none(ro.get("kappa_over_2pi_Hz"))
    if kt_hz is None:
        if ki_hz is not None and ke_hz is not None:
            kt_hz = ki_hz + ke_hz
        elif ki_hz is not None:
            kt_hz = ki_hz
        elif ke_hz is not None:
            kt_hz = ke_hz
    if ke_hz is None and kt_hz is not None and ki_hz is not None:
        ke_hz = max(0.0, kt_hz - ki_hz)
    chi_mhz = _as_float_or_none(q.get("chi_MHz"))
    return {
        "fq_GHz": _as_float_or_none(q.get("f01_epr_GHz")),
        "fr_GHz": _as_float_or_none(ro.get("f_GHz")),
        "chi_MHz": chi_mhz,
        "g_MHz": ((_as_float_or_none((q.get("dispersive") or {}).get("g_GHz")) or 0.0) * 1e3)
        if _as_float_or_none((q.get("dispersive") or {}).get("g_GHz")) is not None
        else None,
        "alpha_MHz": _as_float_or_none(q.get("alpha_epr_MHz")),
        "kint_MHz": (ki_hz / 1e6) if ki_hz is not None else None,
        "kext_MHz": (ke_hz / 1e6) if ke_hz is not None else None,
        "k_MHz": (kt_hz / 1e6) if kt_hz is not None else None,
        "T1_us": _as_float_or_none(q.get("T1_est_us")),
        "chi_over_k": (abs(chi_mhz) * 1e6 / kt_hz) if chi_mhz is not None and kt_hz and kt_hz > 0 else None,
    }


def validate_3q_candidate(payload: TDict[str, Any]) -> Tuple[bool, List[str], TDict[str, TDict[str, Optional[float]]]]:
    metrics = {f"Q{i}": metrics_for_qubit_3q(payload, i) for i in [1, 2, 3]}
    reasons: List[str] = []
    for qname, m in metrics.items():
        fq = m["fq_GHz"]
        fr = m["fr_GHz"]
        chi_over_k = m["chi_over_k"]
        k_mhz = m["k_MHz"]
        kext_mhz = m["kext_MHz"]
        kint_mhz = m["kint_MHz"]
        t1 = m["T1_us"]
        if fq is None or fr is None or not (3.0 < fq < fr < 8.0):
            reasons.append(f"{qname}: need 3<fq<fr<8, got fq={fq}, fr={fr}")
        if chi_over_k is None or not (chi_over_k > 0.1):
            reasons.append(f"{qname}: need chi/k>0.1, got {chi_over_k}")
        if k_mhz is None or not (k_mhz > 1.5):
            reasons.append(f"{qname}: need k>1.5MHz, got {k_mhz}")
        if kext_mhz is None or not (kext_mhz > 1.5):
            reasons.append(f"{qname}: need kext>1.5MHz, got {kext_mhz}")
        if kext_mhz is None or kint_mhz is None or not (kext_mhz > kint_mhz):
            reasons.append(f"{qname}: need kext>kint, got kext={kext_mhz}, kint={kint_mhz}")
        if t1 is None or not (t1 > 40.0):
            reasons.append(f"{qname}: need T1>40us, got {t1}")
    return (len(reasons) == 0), reasons, metrics


def _candidate_param_grid_3q() -> List[TDict[str, Any]]:
    candidates: List[TDict[str, Any]] = []
    targeted = [
        {
            "lj_nh": (10.0, 10.5, 10.0),
            "ro_l_mm": (16.90, 15.00, 16.90),
            "tee_cap_gap_um": (70.0, 85.0, 70.0),
            "tee_cap_width_um": (0.38, 0.32, 0.42),
            "tee_cap_distance_um": (170.0, 170.0, 170.0),
            "stub_length_mm": (0.0, 0.0, 0.0),
            "stub_offset_mm": (2.8, 2.8, 2.8),
            "ro_pad_w_um": (420.0, 400.0, 420.0),
            "ro_pad_h_um": (420.0, 400.0, 420.0),
            "ro_pad_gap_um": (4.0, 4.0, 4.0),
            "lp_dx_mm": (0.56, 0.56, 0.56),
            "sample_suffix": "structured_s1p48_layout0181_compact_neat_tune01_teegentle",
        },
        {
            "lj_nh": (10.0, 10.5, 10.0),
            "ro_l_mm": (16.90, 15.00, 16.90),
            "tee_cap_gap_um": (1.2, 1.2, 1.2),
            "tee_cap_width_um": (50.0, 50.0, 50.0),
            "stub_length_mm": (0.0, 0.0, 0.0),
            "stub_offset_mm": (2.8, 2.8, 2.8),
            "ro_pad_gap_um": (4.0, 4.0, 4.0),
            "lp_dx_mm": (1.20, 1.20, 1.20),
            "sample_suffix": "target_pickfix_mindet_nostub_tg1p2_tw50",
        },
        {
            "lj_nh": (10.0, 10.5, 10.0),
            "ro_l_mm": (16.90, 15.00, 14.70),
            "tee_cap_gap_um": (3.0, 3.0, 3.0),
            "tee_cap_width_um": (18.0, 18.0, 18.0),
            "stub_length_mm": (15.76, 16.73, 17.61),
            "stub_offset_mm": (2.8, 2.8, 2.8),
            "ro_pad_gap_um": (4.0, 4.0, 4.0),
            "lp_dx_mm": (1.20, 1.20, 1.20),
            "sample_suffix": "target_markerfix_strongk_tg3_tw18",
        },
        {
            "lj_nh": (10.0, 10.5, 10.0),
            "ro_l_mm": (16.90, 15.00, 14.70),
            "tee_cap_gap_um": (3.0, 8.0, 8.0),
            "tee_cap_width_um": (4.0, 6.0, 6.0),
            "stub_length_mm": (15.74, 18.62, 21.27),
            "stub_offset_mm": (2.8, 2.8, 2.8),
            "ro_pad_gap_um": (4.0, 4.0, 4.0),
            "lp_dx_mm": (1.20, 1.20, 1.20),
            "sample_suffix": "target_pickfix_lowk_tw4_6_6_tg3_8_8",
        },
        {
            "lj_nh": (13.0, 10.8, 10.0),
            "ro_l_mm": (16.90, 15.00, 14.70),
            "tee_cap_gap_um": (3.0, 8.0, 3.0),
            "tee_cap_width_um": (5.0, 18.0, 18.0),
            "stub_length_mm": (18.42, 17.07, 17.35),
            "stub_offset_mm": (2.8, 2.8, 2.8),
            "ro_pad_gap_um": (4.0, 4.0, 4.0),
            "lp_dx_mm": (1.20, 1.20, 1.20),
            "sample_suffix": "target_lj13_10p8_stub_exact18p42_17p07_17p35",
        },
        {
            "lj_nh": (13.0, 10.8, 10.0),
            "ro_l_mm": (16.90, 15.00, 14.70),
            "tee_cap_gap_um": (3.0, 8.0, 3.0),
            "tee_cap_width_um": (5.0, 18.0, 18.0),
            "stub_length_mm": (17.20, 17.35, 20.08),
            "stub_offset_mm": (2.8, 2.8, 2.8),
            "ro_pad_gap_um": (4.0, 4.0, 4.0),
            "lp_dx_mm": (1.20, 1.20, 1.20),
            "sample_suffix": "target_lj13_10p8_stub_retune_q2gap8",
        },
        {
            "lj_nh": (10.0, 10.5, 10.0),
            "ro_l_mm": (16.90, 15.00, 14.70),
            "tee_cap_gap_um": (3.0, 3.0, 3.0),
            "tee_cap_width_um": (5.0, 18.0, 18.0),
            "stub_length_mm": (17.06, 18.62, 21.27),
            "stub_offset_mm": (2.8, 2.8, 2.8),
            "ro_pad_gap_um": (4.0, 4.0, 4.0),
            "lp_dx_mm": (1.20, 1.20, 1.20),
            "sample_suffix": "target_stub_mix_q1w5_q23w18_stub17p06_18p62_21p27",
        },
        # Keep Q2 launchpad shorter to avoid the prior LP_RO2 / RO_TEE3 Q3D overlap.
        {
            "lj_nh": (10.0, 10.5, 10.0),
            "ro_l_mm": (16.90, 15.00, 13.80),
            "tee_cap_gap_um": (80.0, 100.0, 80.0),
            "tee_cap_width_um": (0.28, 0.22, 0.28),
            "stub_length_mm": (14.33, 14.95, 14.33),
            "ro_pad_gap_um": (4.0, 4.0, 4.0),
            "lp_dx_mm": (1.20, 1.20, 1.20),
            "sample_suffix": "target_stub_lp1p20_ro3short13p8_tg80_100_80_tw0p28_0p22_0p28",
        },
        {
            "lj_nh": (10.0, 10.5, 10.0),
            "ro_l_mm": (16.90, 15.00, 14.70),
            "tee_cap_gap_um": (80.0, 100.0, 80.0),
            "tee_cap_width_um": (0.28, 0.22, 0.28),
            "stub_length_mm": (14.33, 14.95, 14.33),
            "ro_pad_gap_um": (4.0, 4.0, 4.0),
            "lp_dx_mm": (1.20, 1.20, 1.20),
            "sample_suffix": "target_stub_lp1p20_ro3short14p7_tg80_100_80_tw0p28_0p22_0p28",
        },
        {
            "lj_nh": (10.0, 10.5, 10.0),
            "ro_l_mm": (16.90, 15.00, 14.70),
            "tee_cap_gap_um": (20.0, 20.0, 20.0),
            "tee_cap_width_um": (2.5, 2.5, 2.5),
            "stub_length_mm": (18.73, 19.26, 15.74),
            "ro_pad_gap_um": (4.0, 4.0, 4.0),
            "lp_dx_mm": (1.20, 1.20, 1.20),
            "sample_suffix": "target_stub_midk_lp1p20_ro3short14p7_tg20_tw2p5_stub18p73_19p26_15p74",
        },
        {
            "lj_nh": (10.0, 10.5, 10.0),
            "ro_l_mm": (16.90, 15.00, 14.70),
            "tee_cap_gap_um": (3.0, 3.0, 3.0),
            "tee_cap_width_um": (5.0, 5.0, 5.0),
            "stub_length_mm": (18.73, 19.26, 15.74),
            "ro_pad_gap_um": (4.0, 4.0, 4.0),
            "lp_dx_mm": (1.20, 1.20, 1.20),
            "sample_suffix": "target_stub_midk_lp1p20_ro3short14p7_tg3_tw5_stub18p73_19p26_15p74",
        },
        {
            "lj_nh": (10.0, 10.5, 10.0),
            "ro_l_mm": (16.90, 15.00, 14.70),
            "tee_cap_gap_um": (3.0, 3.0, 3.0),
            "tee_cap_width_um": (8.0, 8.0, 8.0),
            "stub_length_mm": (18.73, 19.26, 15.74),
            "ro_pad_gap_um": (4.0, 4.0, 4.0),
            "lp_dx_mm": (1.20, 1.20, 1.20),
            "sample_suffix": "target_stub_midk_lp1p20_ro3short14p7_tg3_tw8_stub18p73_19p26_15p74",
        },
        {
            "lj_nh": (10.0, 10.5, 10.0),
            "ro_l_mm": (16.90, 15.00, 14.70),
            "tee_cap_gap_um": (3.0, 3.0, 3.0),
            "tee_cap_width_um": (18.0, 18.0, 18.0),
            "stub_length_mm": (15.76, 16.73, 17.61),
            "ro_pad_gap_um": (4.0, 4.0, 4.0),
            "lp_dx_mm": (1.20, 1.20, 1.20),
            "sample_suffix": "target_stub_strongk_lp1p20_ro3short14p7_tg3_tw18_stub15p76_16p73_17p61",
        },
        {
            "lj_nh": (10.0, 10.5, 10.0),
            "ro_l_mm": (16.90, 15.00, 14.70),
            "tee_cap_gap_um": (3.0, 3.0, 3.0),
            "tee_cap_width_um": (10.0, 10.0, 10.0),
            "stub_length_mm": (15.74, 18.62, 21.27),
            "ro_pad_gap_um": (4.0, 4.0, 4.0),
            "lp_dx_mm": (1.20, 1.20, 1.20),
            "sample_suffix": "target_stub_midk_q3dpath_lp1p20_ro3short14p7_tg3_tw10_stub15p74_18p62_21p27",
        },
        {
            "lj_nh": (10.0, 10.5, 10.0),
            "ro_l_mm": (16.90, 15.00, 14.70),
            "tee_cap_gap_um": (3.0, 3.0, 3.0),
            "tee_cap_width_um": (5.0, 1.2, 1.5),
            "stub_length_mm": (15.80, 17.32, 16.42),
            "ro_pad_gap_um": (4.0, 4.0, 4.0),
            "lp_dx_mm": (1.20, 1.20, 1.20),
            "sample_suffix": "target_stub_balanced_q3dpath_lp1p20_ro3short14p7_tg3_tw5_1p2_1p5_stub15p80_17p32_16p42",
        },
        {
            "lj_nh": (10.0, 10.5, 10.0),
            "ro_l_mm": (16.90, 15.00, 13.80),
            "tee_cap_gap_um": (20.0, 20.0, 20.0),
            "tee_cap_width_um": (2.5, 2.5, 2.5),
            "stub_length_mm": (18.73, 19.26, 15.74),
            "ro_pad_gap_um": (4.0, 4.0, 4.0),
            "lp_dx_mm": (1.20, 1.20, 1.20),
            "sample_suffix": "target_stub_midk_lp1p20_ro3short13p8_tg20_tw2p5_stub18p73_19p26_15p74",
        },
        {
            "lj_nh": (10.0, 10.5, 10.0),
            "ro_l_mm": (16.90, 15.00, 13.80),
            "tee_cap_gap_um": (80.0, 100.0, 80.0),
            "tee_cap_width_um": (0.28, 0.22, 0.28),
            "stub_length_mm": (14.33, 14.95, 14.33),
            "ro_pad_gap_um": (4.0, 4.0, 4.0),
            "lp_dx_mm": (1.20, 1.20, 1.20),
            "sample_suffix": "target_stub_lp1p20_ro3short13p8_tg80_100_80_tw0p28_0p22_0p28",
        },
        {
            "lj_nh": (10.0, 10.5, 10.0),
            "ro_l_mm": (16.90, 15.00, 14.70),
            "tee_cap_gap_um": (80.0, 100.0, 80.0),
            "tee_cap_width_um": (0.28, 0.22, 0.28),
            "stub_length_mm": (14.33, 14.95, 14.33),
            "ro_pad_gap_um": (4.0, 4.0, 4.0),
            "lp_dx_mm": (1.20, 1.20, 1.20),
            "sample_suffix": "target_stub_lp1p20_ro3short14p7_tg80_100_80_tw0p28_0p22_0p28",
        },
    ]
    candidates.extend(targeted)
    q2_lj_values = [10.5, 10.2, 10.8]
    ro_scale_values = [1.0, 0.98, 1.02]
    tee_gap_values = [80.0, 100.0, 120.0, 150.0]
    tee_width_values = [0.28, 0.22, 0.18]
    stub_scale_values = [1.0, 0.8, 1.2, 0.0]
    ro_pad_gap_values = [4.0, 2.0, 1.0]
    for q2_lj, ro_scale, tee_gap, tee_width, stub_scale, ro_pad_gap in product(
        q2_lj_values,
        ro_scale_values,
        tee_gap_values,
        tee_width_values,
        stub_scale_values,
        ro_pad_gap_values,
    ):
        candidates.append(
            {
                "lj_nh": (10.0, float(q2_lj), 10.0),
                "ro_l_mm": tuple(float(x) * float(ro_scale) for x in (16.90, 15.00, 16.90)),
                "tee_cap_gap_um": (float(tee_gap), float(tee_gap), float(tee_gap)),
                "tee_cap_width_um": (float(tee_width), float(tee_width), float(tee_width)),
                "stub_length_mm": tuple(float(x) * float(stub_scale) for x in (14.33, 14.95, 14.33)),
                "ro_pad_gap_um": (float(ro_pad_gap), float(ro_pad_gap), float(ro_pad_gap)),
                "sample_suffix": (
                    f"lj2_{str(q2_lj).replace('.', 'p')}"
                    f"_ros{str(round(ro_scale, 3)).replace('.', 'p')}"
                    f"_tg{str(round(tee_gap, 3)).replace('.', 'p')}"
                    f"_tw{str(round(tee_width, 3)).replace('.', 'p')}"
                    f"_ss{str(round(stub_scale, 3)).replace('.', 'p')}"
                    f"_rpg{str(round(ro_pad_gap, 3)).replace('.', 'p')}"
                ),
            }
        )
    return candidates


# -------------------------
# mode picking for 3Q
# -------------------------
def _extract_pm_normed_3q(res: Optional[TDict[str, Any]]) -> Optional[pd.DataFrame]:
    if not isinstance(res, dict):
        return None
    pm = res.get("Pm_normed")
    if isinstance(pm, pd.DataFrame):
        return pm
    if pm is not None:
        try:
            arr = np.array(pm, dtype=float)
        except Exception:
            arr = None
        if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[0] > 0 and arr.shape[1] > 0:
            cols = None
            for key in ("Ljs", "Cjs"):
                obj = res.get(key)
                if isinstance(obj, pd.Series) and len(obj.index) == int(arr.shape[1]):
                    cols = [str(x) for x in list(obj.index)]
                    break
            if cols is None:
                cols = [str(i) for i in range(int(arr.shape[1]))]
            return pd.DataFrame(arr, columns=pd.Index(cols))
    for key in ("Pm_norm", "Pm", "pm_normed"):
        obj = res.get(key)
        if isinstance(obj, pd.DataFrame):
            return obj
    return None


def _find_pm_column_3q(pm: pd.DataFrame, preferred: str) -> Optional[str]:
    pref = str(preferred).strip().lower()
    if not pref:
        return None
    for c in list(pm.columns):
        cs = str(c).strip().lower()
        if cs == pref:
            return str(c)
    for c in list(pm.columns):
        cs = str(c).strip().lower()
        if pref in cs:
            return str(c)
    return None


def _pick_three_qubits_and_three_resonators(
    chi_MHz: np.ndarray,
    *,
    freqs_GHz: Optional[np.ndarray] = None,
    res: Optional[TDict[str, Any]] = None,
) -> Tuple[int, int, int, int, int, int, TDict[str, Any]]:
    """Heuristic mode picker for 3Q system.

    Returns: (idx_q1, idx_q2, idx_q3, idx_r1, idx_r2, idx_r3, debug).
    """
    chi = np.array(chi_MHz, dtype=float)
    n = chi.shape[0]
    diag = np.abs(np.diag(chi))
    if n < 6:
        raise ValueError(f"Need >=6 modes for 3Q+3R picking, got n={n}")

    debug: TDict[str, Any] = {
        "method": "self_kerr_fallback",
        "warnings": [],
    }

    q_sorted = [int(i) for i in list(np.argsort(-diag))]
    idx_q1, idx_q2, idx_q3 = int(q_sorted[0]), int(q_sorted[1]), int(q_sorted[2])

    pm = _extract_pm_normed_3q(res)
    if pm is not None and len(pm.index) >= n and len(pm.columns) >= 3:
        jj_prefs = [
            os.environ.get("METAL_3Q_PICK_JJ1", "jj1"),
            os.environ.get("METAL_3Q_PICK_JJ2", "jj2"),
            os.environ.get("METAL_3Q_PICK_JJ3", "jj3"),
        ]
        cols = [_find_pm_column_3q(pm, pref) for pref in jj_prefs]
        if any(c is None for c in cols):
            cols_all = [str(c) for c in list(pm.columns)]
            if len(cols_all) >= 3:
                cols = cols_all[:3]
                debug["warnings"].append(f"Pm columns did not match {jj_prefs}; using {cols}")

        if all(c is not None for c in cols) and len(set(str(c) for c in cols)) == 3:
            pm_min = _env_float("METAL_3Q_PICK_PM_MIN", 1e-3, lo=0.0, hi=1.0)

            def pm_val(mode_idx: int, col: str) -> float:
                try:
                    v = float(pm.loc[:, str(col)].iloc[int(mode_idx)])
                    return float(v) if np.isfinite(v) else 0.0
                except Exception:
                    return 0.0

            top_by_col: TDict[str, Any] = {}
            for col in cols:
                ranked = sorted(
                    [(pm_val(i, str(col)), int(i)) for i in range(n)],
                    key=lambda item: item[0],
                    reverse=True,
                )
                top_by_col[str(col)] = [
                    {
                        "idx": int(idx),
                        "p": float(pv),
                        "f_GHz": float(freqs_GHz[int(idx)]) if freqs_GHz is not None else None,
                        "self_kerr_MHz": float(chi[int(idx), int(idx)]),
                    }
                    for pv, idx in ranked[:6]
                ]

            best: Optional[Tuple[float, int, int, int]] = None
            for i in range(n):
                for j in range(n):
                    if int(j) == int(i):
                        continue
                    for k in range(n):
                        if int(k) in {int(i), int(j)}:
                            continue
                        p11 = pm_val(i, str(cols[0]))
                        p22 = pm_val(j, str(cols[1]))
                        p33 = pm_val(k, str(cols[2]))
                        cross = (
                            pm_val(i, str(cols[1])) + pm_val(i, str(cols[2]))
                            + pm_val(j, str(cols[0])) + pm_val(j, str(cols[2]))
                            + pm_val(k, str(cols[0])) + pm_val(k, str(cols[1]))
                        )
                        if min(p11, p22, p33) < float(pm_min):
                            continue
                        # Primary junction participation must dominate.  In 3Q layouts
                        # adjacent modes can share noticeable participation, so using a
                        # large cross penalty can incorrectly skip the true middle-qubit
                        # mode in favor of a weak high-frequency trace mode.
                        score = 10.0 * float(p11 + p22 + p33)
                        score -= 0.05 * float(cross)
                        score += 0.001 * float(diag[int(i)] + diag[int(j)] + diag[int(k)])
                        if best is None or float(score) > float(best[0]):
                            best = (float(score), int(i), int(j), int(k))

            debug.update({"pm_cols": [str(c) for c in cols], "top_pm_by_col": top_by_col, "pm_min": float(pm_min)})
            if best is not None:
                idx_q1, idx_q2, idx_q3 = int(best[1]), int(best[2]), int(best[3])
                debug["method"] = "junction_participation"
                debug["pm_selected"] = {
                    "Q1": {"idx": int(idx_q1), "p": pm_val(idx_q1, str(cols[0])), "col": str(cols[0])},
                    "Q2": {"idx": int(idx_q2), "p": pm_val(idx_q2, str(cols[1])), "col": str(cols[1])},
                    "Q3": {"idx": int(idx_q3), "p": pm_val(idx_q3, str(cols[2])), "col": str(cols[2])},
                }
            else:
                debug["warnings"].append("No distinct PM-selected 3Q mode tuple met pm_min; using self-Kerr fallback")

    remaining = [i for i in range(n) if i not in (idx_q1, idx_q2, idx_q3)]

    pm_readout_max = _env_float("METAL_3Q_PICK_READOUT_PM_MAX", 0.05, lo=0.0, hi=1.0)
    readout_self_k_max = _env_float("METAL_3Q_PICK_READOUT_SELF_K_MAX_MHZ", 0.5, lo=0.0, hi=1000.0)
    readout_min_detuning = _env_float("METAL_3Q_PICK_READOUT_MIN_DETUNING_GHZ", 0.25, lo=0.0, hi=10.0)
    readout_min_freq = _env_float("METAL_3Q_PICK_READOUT_MIN_FREQ_GHZ", 3.0, lo=0.0, hi=100.0)
    readout_max_freq = _env_float("METAL_3Q_PICK_READOUT_MAX_FREQ_GHZ", 8.0, lo=0.0, hi=100.0)

    def readout_pm_sum(mode_idx: int) -> Optional[float]:
        if pm is None:
            return None
        try:
            row = pm.iloc[int(mode_idx)]
            vals = [abs(float(v)) for v in list(row.values) if np.isfinite(float(v))]
            return float(sum(vals))
        except Exception:
            return None

    def score_res(qi: int, ri: int) -> float:
        fq = float(freqs_GHz[int(qi)]) if freqs_GHz is not None else None
        fr = float(freqs_GHz[int(ri)]) if freqs_GHz is not None else None
        chi_qr = abs(float(chi[qi, ri]))
        self_k = abs(float(chi[ri, ri]))
        pm_sum = readout_pm_sum(int(ri))
        if self_k > readout_self_k_max:
            return -1.0e9
        if pm_sum is not None and pm_sum > pm_readout_max:
            return -1.0e9
        score = chi_qr
        if fq is not None and fr is not None:
            if fr <= fq:
                return -1.0e9
            if (fr - fq) < readout_min_detuning:
                return -1.0e9
            if not (readout_min_freq < fr < readout_max_freq):
                return -1.0e9
            score += 0.20
            if not (3.0 < fq < fr < 8.0):
                score -= 2.0
        score -= 0.10 * self_k
        if pm_sum is not None:
            score -= 0.05 * pm_sum
        return float(score)

    best_r: Optional[Tuple[float, int, int, int]] = None
    for r1_c in remaining:
        for r2_c in remaining:
            if int(r2_c) == int(r1_c):
                continue
            for r3_c in remaining:
                if int(r3_c) in {int(r1_c), int(r2_c)}:
                    continue
                score = (
                    score_res(idx_q1, int(r1_c))
                    + score_res(idx_q2, int(r2_c))
                    + score_res(idx_q3, int(r3_c))
                )
                if best_r is None or float(score) > float(best_r[0]):
                    best_r = (float(score), int(r1_c), int(r2_c), int(r3_c))
    if best_r is None:
        raise ValueError("Failed to assign 3 readout modes")
    r1, r2, r3 = int(best_r[1]), int(best_r[2]), int(best_r[3])
    debug["readout_assignment_score"] = float(best_r[0])
    debug["readout_constraints"] = {
        "pm_sum_max": float(pm_readout_max),
        "self_k_max_MHz": float(readout_self_k_max),
        "min_detuning_GHz": float(readout_min_detuning),
        "readout_min_freq_GHz": float(readout_min_freq),
        "readout_max_freq_GHz": float(readout_max_freq),
        "readout_pm_sums": {
            "R1": readout_pm_sum(r1),
            "R2": readout_pm_sum(r2),
            "R3": readout_pm_sum(r3),
        },
        "readout_self_k_MHz": {
            "R1": float(diag[int(r1)]),
            "R2": float(diag[int(r2)]),
            "R3": float(diag[int(r3)]),
        },
    }

    debug["selected_indices"] = {
        "Q1": int(idx_q1),
        "Q2": int(idx_q2),
        "Q3": int(idx_q3),
        "R1": int(r1),
        "R2": int(r2),
        "R3": int(r3),
    }
    if freqs_GHz is not None:
        debug["selected_freqs_GHz"] = {
            "Q1": float(freqs_GHz[int(idx_q1)]),
            "Q2": float(freqs_GHz[int(idx_q2)]),
            "Q3": float(freqs_GHz[int(idx_q3)]),
            "R1": float(freqs_GHz[int(r1)]),
            "R2": float(freqs_GHz[int(r2)]),
            "R3": float(freqs_GHz[int(r3)]),
        }

    return idx_q1, idx_q2, idx_q3, int(r1), int(r2), int(r3), debug


def _calc_g_from_chi(*, chi_MHz: float, Delta_GHz: float, alpha_MHz: float) -> Optional[float]:
    try:
        Delta_MHz = float(Delta_GHz) * 1e3
        alpha_MHz = float(alpha_MHz)
        chi_MHz = float(chi_MHz)
        if abs(alpha_MHz) <= 1e-9:
            return None
        g_MHz = math.sqrt(abs(chi_MHz * Delta_MHz * (Delta_MHz + alpha_MHz) / alpha_MHz))
        return g_MHz / 1e3
    except Exception:
        return None


def _qi_for_mode(res: dict, idx_mode: int, *, warnings: List[str]) -> Optional[float]:
    Qi_val = None

    p = base.extract_participation(res, idx_mode, kind="dielectrics_bulk", name="main")
    Qi_from_p = base.qi_from_p_tandelta(p, base.TAN_DELTA_MAIN)
    if Qi_from_p is not None:
        Qi_val = float(Qi_from_p)
        warnings.append(
            f"Qi from p*tan_delta: p={float(p):.3g}, tan_delta={float(base.TAN_DELTA_MAIN):.3g}, Qi={Qi_val:.3g}"
        )

    if Qi_val is None:
        Qi_raw = base.extract_qdielectric_main(res, idx_mode)
        if Qi_raw is not None and np.isfinite(Qi_raw) and Qi_raw > 0:
            p_inferred = 1.0 / (float(Qi_raw) * float(base.EPR_TAN_DELTA_ASSUMED))
            if 0.0001 <= p_inferred <= 0.99:
                Qi_val = 1.0 / (p_inferred * float(base.TAN_DELTA_MAIN))
                warnings.append(
                    "Qi recalculated from inferred p="
                    f"{p_inferred:.3g}: tan_delta={float(base.TAN_DELTA_MAIN):.3g} "
                    f"(assumed EPR tan_delta={float(base.EPR_TAN_DELTA_ASSUMED):.3g}), Qi={Qi_val:.3g}"
                )
            else:
                Qi_val = float(Qi_raw)
                warnings.append(f"Qi from EPR Qdielectric_main (tan_delta may differ): {Qi_val:.3g}")

    if Qi_val is None and base.USE_QI_FALLBACK:
        Qi_val = float(base.QI_FALLBACK)
        warnings.append(f"Qi fallback assumed: {base.QI_FALLBACK:g}")

    if (
        Qi_val is not None
        and base.QI_CLAMP_TO_FALLBACK
        and base.USE_QI_FALLBACK
        and Qi_val > float(base.QI_FALLBACK)
    ):
        warnings.append(f"Qi clamped to {base.QI_FALLBACK:g} (raw {Qi_val:g})")
        Qi_val = float(base.QI_FALLBACK)

    return float(Qi_val) if Qi_val is not None else None


def _qdielectric_for_mode(res: dict, idx_mode: int) -> Optional[float]:
    p = base.extract_participation(res, idx_mode, kind="dielectrics_bulk", name="main")
    q_from_p = base.qi_from_p_tandelta(p, base.TAN_DELTA_MAIN)
    if q_from_p is not None:
        return float(q_from_p)

    q_raw = base.extract_qdielectric_main(res, idx_mode)
    if q_raw is not None and np.isfinite(q_raw) and q_raw > 0:
        return float(q_raw)
    return None


def _configure_three_junctions_best_effort(eig, *, lj_vars: Tuple[str, str, str], cj_vars: Tuple[str, str, str]) -> None:
    """Try to ensure all three junctions are wired correctly."""
    try:
        pinfo = eig.sim.renderer.pinfo
    except Exception:
        return

    try:
        if hasattr(eig, "del_junction") and hasattr(eig, "add_junction"):
            eig.del_junction()

            def _try_add(jj_name: str, *, Lj: str, Cj: str, qname: str) -> bool:
                patterns = [
                    (f"JJ_rect_Lj_{qname}_rect_jj", f"JJ_Lj_{qname}_rect_jj_"),
                    (f"JJ_rect_{Lj}_{qname}_rect_jj", f"JJ_{Lj}_{qname}_rect_jj_"),
                ]
                for rect, line in patterns:
                    try:
                        eig.add_junction(jj_name, Lj, Cj, rect=rect, line=line)
                        try:
                            pinfo.validate_junction_info()
                        except Exception:
                            pass
                        return True
                    except Exception:
                        continue
                return False

            _try_add("jj1", Lj=lj_vars[0], Cj=cj_vars[0], qname="Q1")
            _try_add("jj2", Lj=lj_vars[1], Cj=cj_vars[1], qname="Q2")
            _try_add("jj3", Lj=lj_vars[2], Cj=cj_vars[2], qname="Q3")
    except Exception:
        pass


# -------------------------
# analysis
# -------------------------
def run_analysis_pipeline_3q(
    design,
    sample_id: str,
    *,
    root_path: str,
    do_hfss: bool,
    do_q3d: bool,
    lj_nh: Tuple[float, float, float],
    cj_fF_user: Tuple[float, float, float],
    inputs_sweep: Optional[TDict[str, Any]] = None,
    inputs_qubits: Optional[TDict[str, Any]] = None,
    gds_subdir: str = "gds",
    json_subdir: str = "json",
    layout_params: Optional[Layout3QParams] = None,
) -> TDict[str, Any]:
    root = Path(root_path)
    gds_dir = root / str(gds_subdir)
    json_dir = root / str(json_subdir)
    gds_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    payload = build_chip_summary_3q(sample_id=sample_id)
    payload["status"] = "running"

    payload["inputs"] = {
        "sweep": dict(inputs_sweep or {}),
        "qubits": dict(inputs_qubits or {}),
        "junction": {
            "Lj_nH": [float(x) for x in lj_nh],
            "Cj_fF": [float(x) for x in cj_fF_user],
        },
    }

    payload["meta"]["filename_base"] = sample_id
    sim_tag = base.short_hash_tag(sample_id, n=8)
    payload["meta"]["ansys_sim_tag"] = sim_tag

    lj_vars = ("Lj1", "Lj2", "Lj3")
    cj_vars = ("Cj1", "Cj2", "Cj3")

    # 1) GDS
    try:
        gds_path = gds_dir / f"{sample_id}.gds"
        abs_gds_path = str(gds_path.resolve())
        info = base.export_gds_robust(design, abs_gds_path)
        payload["meta"]["gds_path"] = abs_gds_path
        payload["meta"]["gds_export_mode"] = info["mode"]
        payload["meta"]["gds_error"] = info.get("error") or ""
    except Exception as e:
        payload["meta"]["gds_error"] = str(e)

    # 2) HFSS/EPR
    if do_hfss:
        for attempt in [1, 2]:
            eig = None
            try:
                eig = base.EPRanalysis(design, "hfss")
                hfss = eig.sim.renderer

                if hfss is None:
                    raise RuntimeError("HFSS renderer is not available.")

                workdir = base._ansys_short_workdir(root)
                target = workdir / f"EIG3Q_{sim_tag}_a{attempt}.aedt"

                print(f"[HFSS] attempt={attempt} Starting HFSS...", flush=True)
                base._ansys_best_effort_reset()
                hfss.start()
                time.sleep(0.8)
                base._ansys_wait_ready(hfss, timeout_s=45.0)

                base._ansys_prepare_project(hfss, target)

                eig.sim.setup.vars = base.Dict(
                    Lj1=f"{float(lj_nh[0])} nH",
                    Cj1=f"{float(cj_fF_user[0])} fF",
                    Lj2=f"{float(lj_nh[1])} nH",
                    Cj2=f"{float(cj_fF_user[1])} fF",
                    Lj3=f"{float(lj_nh[2])} nH",
                    Cj3=f"{float(cj_fF_user[2])} fF",
                )
                hfss_mesh_options = {
                    "max_mesh_length_jj": os.environ.get("METAL_3Q_HFSS_MAX_MESH_LENGTH_JJ", "7um"),
                    "max_mesh_length_port": os.environ.get("METAL_3Q_HFSS_MAX_MESH_LENGTH_PORT", "7um"),
                }
                try:
                    for _mesh_key, _mesh_val in hfss_mesh_options.items():
                        hfss.options[_mesh_key] = _mesh_val
                except Exception as _mesh_e:
                    payload["meta"]["hfss_mesh_options_error"] = str(_mesh_e)
                payload["meta"]["hfss_mesh_options"] = base.to_jsonable(hfss_mesh_options)

                hfss_setup_updates = {
                    "n_modes": _env_int("METAL_3Q_HFSS_N_MODES", 12, lo=4, hi=30),
                    "max_passes": _env_int("METAL_3Q_HFSS_MAX_PASSES", 6, lo=1, hi=30),
                    "min_passes": _env_int("METAL_3Q_HFSS_MIN_PASSES", 6, lo=1, hi=30),
                    "min_converged": _env_int("METAL_3Q_HFSS_MIN_CONVERGED", 2, lo=1, hi=5),
                    "min_freq_ghz": _env_float("METAL_3Q_HFSS_MIN_FREQ_GHZ", 3.0, lo=0.1, hi=20.0),
                    "max_delta_f": _env_float("METAL_3Q_HFSS_MAX_DELTA_F_GHZ", 0.1, lo=0.001, hi=1.0),
                    "pct_refinement": _env_int("METAL_3Q_HFSS_PCT_REFINEMENT", 30, lo=1, hi=100),
                }
                try:
                    eig.sim.setup_update(**hfss_setup_updates)
                except Exception:
                    for k, v in hfss_setup_updates.items():
                        try:
                            setattr(eig.sim.setup, k, v)
                        except Exception:
                            pass
                payload["meta"]["hfss_setup_used"] = base.to_jsonable(hfss_setup_updates)

                all_components = list(design.components.keys())
                print(f"[HFSS] Running simulation with components: {all_components}", flush=True)
                eig.sim.run(
                    name=f"Eig3Q_{sim_tag}_a{attempt}",
                    components=all_components,
                    open_terminations=[],
                    box_plus_buffer=True,
                )

                _configure_three_junctions_best_effort(eig, lj_vars=lj_vars, cj_vars=cj_vars)

                eig.setup.dissipatives = {"dielectrics_bulk": ["main"]}
                try:
                    if hasattr(eig.setup, "dissipative"):
                        eig.setup.dissipative = {"dielectrics_bulk": {"main": base.TAN_DELTA_MAIN}}
                except Exception:
                    pass

                # Avoid memory-heavy numerical diagonalization; use analytic chi_O1.
                try:
                    eig.setup.cos_trunc = None
                    eig.setup.fock_trunc = None
                except Exception:
                    pass

                try:
                    eig.run_epr()
                except KeyError as e:
                    qa_tmp = getattr(eig.sim.renderer, "epr_quantum_analysis", None)
                    if str(e) == "'_Lj'" and qa_tmp is not None and getattr(qa_tmp, "results", None):
                        payload["meta"]["epr_report_warning"] = "KeyError('_Lj') in report_hamiltonian ignored"
                    else:
                        raise

                freqs = eig.get_frequencies().iloc[:, 0].values
                for roname in ["readout1", "readout2", "readout3"]:
                    payload["resonators"][roname]["modes_f_GHz"] = base.to_jsonable(freqs)

                qa = getattr(eig.sim.renderer, "epr_quantum_analysis", None)
                if qa is None or not getattr(qa, "results", None):
                    raise RuntimeError("EPR quantum_analysis not available or results empty.")
                res_key = list(qa.results.keys())[0]
                res = qa.results[res_key]

                chi_nd = res.get("chi_ND", None)
                chi_o1 = res.get("chi_O1", None)

                chi_mat = None
                if chi_nd is not None and (not hasattr(chi_nd, "empty") or not chi_nd.empty):
                    chi_mat = chi_nd
                elif chi_o1 is not None and (not hasattr(chi_o1, "empty") or not chi_o1.empty):
                    chi_mat = chi_o1
                if chi_mat is None:
                    raise RuntimeError("No chi matrix found in EPR results.")

                chi_matrix = np.array(chi_mat.values, dtype=float)

                idx_q1, idx_q2, idx_q3, idx_r1, idx_r2, idx_r3, picker_debug = _pick_three_qubits_and_three_resonators(
                    chi_matrix,
                    freqs_GHz=np.array(freqs, dtype=float),
                    res=res,
                )
                payload["meta"]["mode_picker_debug"] = base.to_jsonable(picker_debug)

                f_vals = {
                    "q1": float(freqs[idx_q1]),
                    "q2": float(freqs[idx_q2]),
                    "q3": float(freqs[idx_q3]),
                    "r1": float(freqs[idx_r1]),
                    "r2": float(freqs[idx_r2]),
                    "r3": float(freqs[idx_r3]),
                }

                alpha = {
                    "q1": -abs(float(chi_matrix[idx_q1, idx_q1])),
                    "q2": -abs(float(chi_matrix[idx_q2, idx_q2])),
                    "q3": -abs(float(chi_matrix[idx_q3, idx_q3])),
                }
                K = {
                    "r1": float(chi_matrix[idx_r1, idx_r1]),
                    "r2": float(chi_matrix[idx_r2, idx_r2]),
                    "r3": float(chi_matrix[idx_r3, idx_r3]),
                }
                chi_qr = {
                    "q1r1": float(chi_matrix[idx_q1, idx_r1]),
                    "q2r2": float(chi_matrix[idx_q2, idx_r2]),
                    "q3r3": float(chi_matrix[idx_q3, idx_r3]),
                }
                chi_qq = {
                    "12": float(chi_matrix[idx_q1, idx_q2]),
                    "23": float(chi_matrix[idx_q2, idx_q3]),
                    "13": float(chi_matrix[idx_q1, idx_q3]),
                }

                # Store in payload
                for i, (qname, roname, qi, ri) in enumerate([
                    ("Q1", "readout1", "q1", "r1"),
                    ("Q2", "readout2", "q2", "r2"),
                    ("Q3", "readout3", "q3", "r3"),
                ]):
                    ro = payload["resonators"][roname]
                    q = payload["qubits"][qname]
                    idx_q = [idx_q1, idx_q2, idx_q3][i]
                    idx_r = [idx_r1, idx_r2, idx_r3][i]

                    ro["f_GHz"] = f_vals[ri]
                    ro["K_MHz"] = K[ri]
                    ro["picked_qubit_mode_index"] = idx_q
                    ro["picked_res_mode_index"] = idx_r

                    Delta = f_vals[qi] - f_vals[ri]
                    chi_val = chi_qr[f"{qi}{ri}"]
                    g_GHz = _calc_g_from_chi(chi_MHz=chi_val, Delta_GHz=Delta, alpha_MHz=alpha[qi])

                    q["f01_epr_GHz"] = f_vals[qi]
                    q["alpha_epr_MHz"] = alpha[qi]
                    q["chi_MHz"] = chi_val
                    q["dispersive"]["chi_GHz"] = chi_val / 1e3
                    q["dispersive"]["Delta_GHz"] = Delta
                    q["dispersive"]["g_GHz"] = g_GHz

                    # Qi
                    ro_w = ro.get("warnings") or []
                    Qi = _qi_for_mode(res, idx_r, warnings=ro_w)
                    ro["Qi"] = Qi
                    ro["kappa_i_over_2pi_Hz"] = (f_vals[ri] * 1e9) / float(Qi) if Qi else None

                    # qubit dielectric Q
                    qQ = _qdielectric_for_mode(res, idx_q)
                    if qQ is not None:
                        q["Q_dielectric_main"] = float(qQ)

                # store sol columns for debugging participation extraction
                try:
                    sol = res.get("sol", None)
                    if isinstance(sol, pd.DataFrame):
                        payload["meta"]["epr_sol_columns"] = [str(c) for c in sol.columns]
                except Exception:
                    pass

                payload["chip"]["chi12_MHz"] = chi_qq["12"]
                payload["chip"]["chi23_MHz"] = chi_qq["23"]
                payload["chip"]["chi13_MHz"] = chi_qq["13"]

                payload["coupling"] = base.build_coupling_pairs(
                    chi_matrix_MHz=chi_matrix,
                    freqs_GHz=freqs,
                    idx_qubits=[idx_q1, idx_q2, idx_q3],
                    idx_readouts=[idx_r1, idx_r2, idx_r3],
                    qubit_names=["Q1", "Q2", "Q3"],
                    readout_names=["readout1", "readout2", "readout3"],
                )

                break

            except Exception as e:
                payload["meta"]["hfss_error"] = str(e)
                print(f"[HFSS] attempt={attempt} Error: {e}", flush=True)
                try:
                    if hfss is not None:
                        base._print_ansys_messages(hfss, "HFSS", level=2)
                except Exception:
                    pass
                if attempt == 1:
                    try:
                        if eig is not None:
                            eig.sim.close()
                    except Exception:
                        pass
                    base._ansys_best_effort_reset()
                    time.sleep(2.0)
                    gc.collect()
                    continue
            finally:
                try:
                    if eig is not None:
                        eig.sim.close()
                except Exception:
                    pass

    # 3) Q3D
    Ceff = {1: None, 2: None, 3: None}
    cj_fF = {1: float(cj_fF_user[0]), 2: float(cj_fF_user[1]), 3: float(cj_fF_user[2])}

    if do_q3d:
        for attempt in [1, 2]:
            lom = None
            q3d = None
            try:
                lom = base.LOManalysis(design, "q3d")
                q3d = lom.sim.renderer

                if q3d is None:
                    raise RuntimeError("Q3D renderer is not available.")

                workdir = base._ansys_short_workdir(root)
                target = workdir / f"Q3D3Q_{sim_tag}_a{attempt}.aedt"

                print(f"[Q3D] attempt={attempt} Starting Q3D...", flush=True)
                base._ansys_best_effort_reset()
                q3d.start()
                time.sleep(0.8)
                base._ansys_wait_ready(q3d, timeout_s=45.0)

                base._ansys_prepare_project(q3d, target)

                try:
                    lom.sim.renderer_initialized = True
                except Exception:
                    pass

                lom.sim.setup.freq_ghz = 5.0
                lom.sim.setup.max_passes = _env_int("METAL_3Q_Q3D_MAX_PASSES", 4, lo=1, hi=30)
                payload["meta"]["q3d_setup_used"] = {
                    "freq_ghz": float(lom.sim.setup.freq_ghz),
                    "max_passes": int(lom.sim.setup.max_passes),
                }
                print(f"[Q3D] Running simulation...", flush=True)
                q3d_open_terminations = []
                if _env_bool("METAL_3Q_Q3D_USE_OPEN_TERMINATIONS", False):
                    q3d_open_terminations = [
                        (f"LP_RO{i}", "tie") for i in (1, 2, 3) if f"LP_RO{i}" in design.components
                    ]
                    q3d_open_terminations.extend(
                        [(f"RO_TEE{i}", "prime_start") for i in (1, 2, 3) if f"RO_TEE{i}" in design.components]
                    )
                q3d_components = [
                    name
                    for name in list(design.components.keys())
                    if not (
                        name.startswith("RO_PSTUB")
                    )
                ]
                q3d_open_terminations = [
                    (comp, pin)
                    for comp, pin in q3d_open_terminations
                    if comp in set(q3d_components)
                ]
                q3d_run_kwargs = {
                    "name": f"Q3D3Q_{sim_tag}_a{attempt}",
                    "components": q3d_components,
                    "box_plus_buffer": False,
                }
                if q3d_open_terminations:
                    q3d_run_kwargs["open_terminations"] = q3d_open_terminations
                lom.sim.run(**q3d_run_kwargs)

                Cmat = lom.sim.capacitance_matrix
                full_run_name = f"Q3D3Q_{sim_tag}_a{attempt}"
                payload["q3d"].setdefault("runs", [])
                payload["q3d"]["runs"].append(
                    {
                        "name": full_run_name,
                        "purpose": "full_chip",
                        "components": list(q3d_components),
                        "excluded_components": [
                            name for name in list(design.components.keys()) if name not in set(q3d_components)
                        ],
                    }
                )

                def _ceff_for_qubit(qname: str, cj_fF_local: float) -> Optional[float]:
                    pt = f"pad_top_{qname}"
                    pb = f"pad_bot_{qname}"
                    if pt in Cmat.index and pb in Cmat.index:
                        C2 = Cmat.loc[[pt, pb], [pt, pb]].values.astype(float)
                        C_mode_fF = base.diff_mode_ceff_from_2x2_maxwell_fF(C2)
                        if C_mode_fF is not None and np.isfinite(C_mode_fF):
                            return float(C_mode_fF) + float(cj_fF_local)
                    return None

                Ceff[1] = _ceff_for_qubit("Q1", cj_fF[1])
                Ceff[2] = _ceff_for_qubit("Q2", cj_fF[2])
                Ceff[3] = _ceff_for_qubit("Q3", cj_fF[3])

                payload["q3d"]["internal"]["capacitances_fF"] = {
                    "units": "fF",
                    "nodes": list(Cmat.index),
                    "capacitance_matrix_fF": base.to_jsonable(Cmat),
                }

                def _annotate_cin_full_chip(cin_pick: TDict[str, Any], *, tee_name: str) -> TDict[str, Any]:
                    a = f"cap_body_0_{tee_name}"
                    b = f"cap_body_1_{tee_name}"
                    marker_prefix = f"feed_marker_{tee_name}"
                    marker = next((str(x) for x in Cmat.index if str(x).startswith(marker_prefix)), None)
                    if a in Cmat.index and marker in Cmat.index:
                        cin_pick = {
                            "Cin_fF": abs(float(Cmat.loc[a, marker])),
                            "res_node": a,
                            "feed_node": marker,
                            "pair": f"{a} <-> {marker}",
                            "method": "q3d_feed_marker",
                        }
                    missing: List[str] = []
                    if a not in Cmat.index:
                        missing.append(a)
                    if b not in Cmat.index:
                        missing.append(b)

                    cin_pick["source_run"] = full_run_name
                    if cin_pick.get("method") == "q3d_feed_marker":
                        if missing:
                            cin_pick["full_chip_missing_nodes"] = missing
                    elif not missing:
                        cin_pick["method"] = "q3d_full_chip"
                    elif a in Cmat.index:
                        cin_pick["method"] = "q3d_full_chip_fallback"
                        cin_pick["full_chip_missing_nodes"] = missing
                    else:
                        cin_pick["method"] = "q3d_full_chip_missing"
                        cin_pick["full_chip_missing_nodes"] = missing
                    return cin_pick

                def _cin_components_for_suffix(suffix: str) -> List[str]:
                    out: List[str] = []
                    for n in (
                        f"RO_FILTER_TEE{suffix}",
                        f"RO_FILTER_LINK{suffix}",
                        f"RO_TEE{suffix}",
                        f"RO_RES{suffix}",
                        f"RO_FEED{suffix}",
                        f"LP_RO{suffix}",
                    ):
                        if n in design.components:
                            out.append(n)
                    return out

                def _refine_cin_if_missing_plate(
                    cin_pick: TDict[str, Any],
                    *,
                    tee_name: str,
                    suffix: str,
                ) -> TDict[str, Any]:
                    # Full-chip Q3D sometimes merges/aliases one tee capacitor plate.
                    # A readout-chain-only run restores the local cap_body_0/cap_body_1 pair.
                    a = f"cap_body_0_{tee_name}"
                    b = f"cap_body_1_{tee_name}"
                    force_chain = _env_bool("METAL_3Q_Q3D_FORCE_CHAIN_CIN", False)
                    if a in Cmat.index and b in Cmat.index and not force_chain:
                        return cin_pick

                    missing: List[str] = []
                    if a not in Cmat.index:
                        missing.append(a)
                    if b not in Cmat.index:
                        missing.append(b)

                    pair = str(cin_pick.get("pair") or "")
                    res_node = str(cin_pick.get("res_node") or "")
                    feed_node = str(cin_pick.get("feed_node") or "")
                    other_te_names = [f"RO_TEE{j}" for j in (1, 2, 3) if str(j) != str(suffix)]

                    suspicious = bool(missing)
                    if any(other in pair or other in res_node or other in feed_node for other in other_te_names):
                        suspicious = True
                    try:
                        if float(cin_pick.get("Cin_fF") or 0.0) <= 0.0:
                            suspicious = True
                    except Exception:
                        suspicious = True
                    if not suspicious and not force_chain:
                        return cin_pick

                    chain = _cin_components_for_suffix(suffix)
                    if not chain:
                        return cin_pick

                    try:
                        run_name = f"Q3D3Q_{sim_tag}_a{attempt}_cin{suffix}"
                        print(
                            f"[Q3D] Refining Cin for {tee_name}: missing={missing}, components={chain}",
                            flush=True,
                        )
                        chain_open = []
                        if _env_bool("METAL_3Q_Q3D_USE_OPEN_TERMINATIONS", False):
                            if f"LP_RO{suffix}" in design.components:
                                chain_open.append((f"LP_RO{suffix}", "tie"))
                            if tee_name in design.components:
                                chain_open.extend([(tee_name, "prime_start"), (tee_name, "prime_end")])
                        chain_run_kwargs = {
                            "name": run_name,
                            "components": chain,
                            "box_plus_buffer": False,
                        }
                        if chain_open:
                            chain_run_kwargs["open_terminations"] = chain_open
                        lom.sim.run(
                            **chain_run_kwargs,
                        )
                        Cmat_local = lom.sim.capacitance_matrix
                    except Exception as refine_e:
                        cin_pick["refine_error"] = str(refine_e)
                        return cin_pick

                    payload["q3d"].setdefault("runs", [])
                    payload["q3d"]["runs"].append(
                        {
                            "name": run_name,
                            "purpose": f"cin_refine_readout{suffix}",
                            "components": list(chain),
                        }
                    )
                    payload["q3d"].setdefault("refinements", {})
                    payload["q3d"]["refinements"][f"readout{suffix}"] = {
                        "source_run": run_name,
                        "full_chip_missing_nodes": list(missing),
                        "components": list(chain),
                        "units": "fF",
                        "nodes": list(Cmat_local.index),
                        "capacitance_matrix_fF": base.to_jsonable(Cmat_local),
                    }

                    if a in Cmat_local.index and b in Cmat_local.index:
                        refined = base.pick_cin_prefer_tee_bodies(Cmat_local, tee_name=tee_name)
                        refined["method"] = "q3d_chain_refine" if missing else "q3d_chain_only"
                        refined["source_run"] = run_name
                        refined["full_chip_missing_nodes"] = list(missing)
                        return refined

                    if not _env_bool("METAL_3Q_Q3D_TEE_ONLY_CIN", True):
                        return cin_pick

                    try:
                        tee_run_name = f"Q3D3Q_{sim_tag}_a{attempt}_tee{suffix}"
                        print(
                            f"[Q3D] Refining Cin for {tee_name}: tee-only components={[tee_name]}",
                            flush=True,
                        )
                        lom.sim.run(
                            name=tee_run_name,
                            components=[tee_name],
                            box_plus_buffer=False,
                        )
                        Cmat_tee = lom.sim.capacitance_matrix
                    except Exception as tee_refine_e:
                        cin_pick["tee_refine_error"] = str(tee_refine_e)
                        return cin_pick

                    payload["q3d"]["runs"].append(
                        {
                            "name": tee_run_name,
                            "purpose": f"cin_refine_readout{suffix}_tee_only",
                            "components": [tee_name],
                        }
                    )
                    payload["q3d"]["refinements"][f"readout{suffix}_tee_only"] = {
                        "source_run": tee_run_name,
                        "full_chip_missing_nodes": list(missing),
                        "components": [tee_name],
                        "units": "fF",
                        "nodes": list(Cmat_tee.index),
                        "capacitance_matrix_fF": base.to_jsonable(Cmat_tee),
                    }

                    if a in Cmat_tee.index and b in Cmat_tee.index:
                        refined = base.pick_cin_prefer_tee_bodies(Cmat_tee, tee_name=tee_name)
                        refined["method"] = "q3d_tee_only_refine"
                        refined["source_run"] = tee_run_name
                        refined["full_chip_missing_nodes"] = list(missing)
                        return refined
                    return cin_pick

                def _fallback_cin_from_self_cap_if_missing_plate(
                    cin_pick: TDict[str, Any],
                    *,
                    tee_name: str,
                ) -> TDict[str, Any]:
                    a = f"cap_body_0_{tee_name}"
                    b = f"cap_body_1_{tee_name}"
                    if not _env_bool("METAL_3Q_Q3D_SELF_CAP_FALLBACK", False):
                        return cin_pick
                    if a not in Cmat.index or b in Cmat.index:
                        return cin_pick
                    try:
                        current = float(cin_pick.get("Cin_fF") or 0.0)
                    except Exception:
                        current = 0.0
                    try:
                        self_cap = abs(float(Cmat.loc[a, a]))
                    except Exception:
                        self_cap = 0.0
                    if not np.isfinite(self_cap) or self_cap <= 0.0:
                        return cin_pick
                    if current > 0.0 and str(cin_pick.get("method") or "") not in {
                        "q3d_full_chip_fallback",
                        "q3d_full_chip_missing",
                    }:
                        return cin_pick
                    out = dict(cin_pick)
                    out.update(
                        {
                            "Cin_fF": float(self_cap),
                            "res_node": a,
                            "feed_node": b,
                            "pair": f"{a} self-cap fallback; missing {b}",
                            "method": "q3d_self_cap_fallback_missing_feed_plate",
                            "source_run": full_run_name,
                            "full_chip_missing_nodes": [b],
                            "fallback_warning": (
                                "Q3D merged the tee feed-side plate into the connected feed/readout metal; "
                                "using the exposed resonator-side tee plate diagonal self capacitance as an "
                                "effective Cin estimate for sweep ranking."
                            ),
                        }
                    )
                    return out

                # Cin per tee; refine readout chain when full-chip matrix misses one capacitor plate.
                cin = {}
                for i in [1, 2, 3]:
                    tee_name = f"RO_TEE{i}"
                    cin[i] = _annotate_cin_full_chip(
                        base.pick_cin_prefer_tee_bodies(Cmat, tee_name=tee_name),
                        tee_name=tee_name,
                    )
                    cin[i] = _refine_cin_if_missing_plate(cin[i], tee_name=tee_name, suffix=str(i))
                    cin[i] = _fallback_cin_from_self_cap_if_missing_plate(cin[i], tee_name=tee_name)
                    ro_i = payload["resonators"][f"readout{i}"]
                    ro_i["external"].update(cin[i])
                    if cin[i].get("method") == "q3d_chain_refine":
                        ro_i["warnings"].append(
                            "Cin refined via chain-only Q3D run="
                            f"{cin[i].get('source_run')} missing_full_chip_nodes={cin[i].get('full_chip_missing_nodes')}"
                        )
                    if cin[i].get("method") == "q3d_self_cap_fallback_missing_feed_plate":
                        ro_i["warnings"].append(str(cin[i].get("fallback_warning") or "Q3D self-cap Cin fallback used."))

                payload["q3d"]["external"] = {
                    "units": "fF",
                    "readout1": cin[1],
                    "readout2": cin[2],
                    "readout3": cin[3],
                }

                # kappa_e estimates
                from qiskit_metal.analyses.em.kappa_calculation import kappa_in

                for i in [1, 2, 3]:
                    ro = payload["resonators"][f"readout{i}"]
                    if cin[i].get("Cin_fF") and ro.get("f_GHz"):
                        fr_hz = float(ro["f_GHz"]) * 1e9
                        Cin_F = float(cin[i]["Cin_fF"]) * 1e-15
                        ke_hz = float(kappa_in(fr_hz, Cin_F, fr_hz))
                        ke_over_2pi_hz = ke_hz / (2.0 * math.pi) if ke_hz > 0 else None
                        ro["external"]["kappa_over_2pi_Hz"] = ke_over_2pi_hz
                        ro["external"]["kappa_Hz"] = ke_hz
                        ro["external"]["Qe"] = fr_hz / ke_hz if ke_hz > 0 else None
                    pf = _purcell_filter_meta_for_readout_3q(layout_params, i)
                    if pf:
                        ro["purcell_filter"] = pf

                break

            except Exception as e:
                payload["meta"]["q3d_error"] = str(e)
                print(f"[Q3D] attempt={attempt} Error: {e}", flush=True)
                try:
                    if q3d is not None:
                        base._print_ansys_messages(q3d, "Q3D", level=2)
                except Exception:
                    pass
                if attempt == 1:
                    try:
                        if lom is not None:
                            lom.sim.close()
                    except Exception:
                        pass
                    base._ansys_best_effort_reset()
                    time.sleep(2.0)
                    gc.collect()
                    continue
            finally:
                try:
                    if lom is not None:
                        lom.sim.close()
                except Exception:
                    pass

    # Finalize
    for i, qname in enumerate(["Q1", "Q2", "Q3"], 1):
        payload["qubits"][qname]["Lj_H"] = float(lj_nh[i-1]) * 1e-9
        payload["qubits"][qname]["Cj_fF"] = float(cj_fF[i])
        payload["qubits"][qname]["C_eff_fF"] = Ceff[i]

    payload = postprocess_3q(payload)
    payload["status"] = "completed"

    json_path = json_dir / f"{sample_id}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(base.to_jsonable(base.export_payload_v3(payload)), f, indent=2, ensure_ascii=False)

    return payload


# -------------------------
# main
# -------------------------
if __name__ == "__main__":
    dataset_root = Path("./data/sqchip_em_3q")
    dataset_root.mkdir(parents=True, exist_ok=True)

    # Junction params aligned to the current 2Q baseline; Q3 mirrors Q1.
    LJ_NH = (10.0, 10.5, 10.0)
    # Match Qiskit Metal / pyEPR common default: ignore JJ capacitance in EM (use Cj=0).
    CJ_FF = (0.0, 0.0, 0.0)

    # Q1/Q3 mirror the current 2Q Q1 readout; Q2 mirrors the current 2Q Q2 readout.
    RO_L_MM = (16.90, 15.00, 16.90)
    TEE_FINGER_LENGTH_UM = (10, 10, 10)
    TEE_FINGER_COUNT = (1, 1, 1)
    TEE_CAP_GAP_UM = (80.0, 100.0, 80.0)
    TEE_CAP_WIDTH_UM = (0.28, 0.22, 0.28)
    TEE_CAP_DISTANCE_UM = (170.0, 170.0, 170.0)
    Q_RO_PAD_W_UM = (420.0, 400.0, 420.0)
    Q_RO_PAD_H_UM = (420.0, 400.0, 420.0)
    Q_RO_PAD_GAP_UM = (4.0, 4.0, 4.0)
    PURCELL_STUB_LENGTH_MM = (14.33, 14.95, 14.33)
    PURCELL_STUB_OFFSET_MM = (2.8, 2.8, 2.8)
    PURCELL_STUB_WIDTH_UM = (10.0, 10.0, 10.0)
    PURCELL_STUB_GAP_UM = (6.0, 6.0, 6.0)
    BUS_PAD_W_UM = 270.0
    BUS_PAD_H_UM = 270.0
    BUS_PAD_GAP_UM = 3.0
    LP_DX_MM = (0.56, 0.56, 0.56)

    RUN_ONE_EXAMPLE = _env_bool("METAL_3Q_RUN_ONE_EXAMPLE", False)
    MAX_SAMPLES = _env_int("METAL_3Q_BATCH_SIZE", 1 if RUN_ONE_EXAMPLE else 1, lo=1, hi=20)
    RESUME_FROM_SUMMARY = False if RUN_ONE_EXAMPLE else True
    TARGET_ABS_CHI_MHZ = 2.0
    DO_HFSS = _env_bool("METAL_3Q_DO_HFSS", True)
    DO_Q3D = _env_bool("METAL_3Q_DO_Q3D", True)

    os.environ.setdefault("METAL_3Q_HFSS_N_MODES", "12")
    os.environ.setdefault("METAL_3Q_HFSS_MAX_PASSES", "6")
    os.environ.setdefault("METAL_3Q_HFSS_MIN_PASSES", "6")
    os.environ.setdefault("METAL_3Q_HFSS_MIN_CONVERGED", "2")
    os.environ.setdefault("METAL_3Q_HFSS_MIN_FREQ_GHZ", "3.0")
    os.environ.setdefault("METAL_3Q_HFSS_MAX_DELTA_F_GHZ", "0.1")
    os.environ.setdefault("METAL_3Q_HFSS_PCT_REFINEMENT", "30")
    os.environ.setdefault("METAL_3Q_HFSS_MAX_MESH_LENGTH_JJ", "7um")
    os.environ.setdefault("METAL_3Q_HFSS_MAX_MESH_LENGTH_PORT", "7um")
    os.environ.setdefault("METAL_3Q_PICK_READOUT_PM_MAX", "0.05")
    os.environ.setdefault("METAL_3Q_PICK_READOUT_SELF_K_MAX_MHZ", "0.5")
    os.environ.setdefault("METAL_3Q_PICK_READOUT_MIN_DETUNING_GHZ", "0.25")
    os.environ.setdefault("METAL_3Q_Q3D_FEED_MARKERS", "0")
    os.environ.setdefault("METAL_3Q_Q3D_SELF_CAP_FALLBACK", "0")
    os.environ.setdefault("METAL_3Q_Q3D_MAX_PASSES", "4")
    os.environ.setdefault("METAL_3Q_USE_QI_FALLBACK", "1")
    os.environ.setdefault("METAL_3Q_QI_FALLBACK", "4000000")
    base.USE_QI_FALLBACK = _env_bool("METAL_3Q_USE_QI_FALLBACK", True)
    base.QI_FALLBACK = _env_float("METAL_3Q_QI_FALLBACK", 4000000.0, lo=1.0, hi=1.0e12)

    csv_headers = [
        "sample_id",
        "q1_x_mm", "q1_y_mm",
        "q2_x_mm", "q2_y_mm",
        "q3_x_mm", "q3_y_mm",
        "fq1_GHz", "fq2_GHz", "fq3_GHz",
        "fr1_GHz", "fr2_GHz", "fr3_GHz",
        "chi12_MHz", "chi23_MHz", "chi13_MHz",
        "q1_chi_over_k", "q2_chi_over_k", "q3_chi_over_k",
        "q1_k_MHz", "q2_k_MHz", "q3_k_MHz",
        "q1_kext_MHz", "q2_kext_MHz", "q3_kext_MHz",
        "q1_kint_MHz", "q2_kint_MHz", "q3_kint_MHz",
        "q1_T1_us", "q2_T1_us", "q3_T1_us",
        "valid", "invalid_reason",
        "status", "error",
    ]

    summary_csv_path = dataset_root / "summary_3q_sweep.csv"
    if not summary_csv_path.exists():
        with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(csv_headers)

    existing_ids = set()
    if RESUME_FROM_SUMMARY and summary_csv_path.exists():
        try:
            with open(summary_csv_path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    sid = row.get("sample_id")
                    if sid:
                        existing_ids.add(str(sid))
        except Exception:
            existing_ids = set()

    def write_row(row_dict: TDict[str, Any]) -> None:
        row = [row_dict.get(h, "") for h in csv_headers]
        with open(summary_csv_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)

    # Layout: Q1 -- Q2 -- Q3 in a row. Keep a structure-first routing:
    # buses stay in the lower channel, readout tees and launchpads use one
    # tighter upper row, and the readout meanders sit above the qubit pads.
    QUBIT_SPACING_MM = 1.48
    CHIP_SIZE_X_MM = 4.30
    CHIP_SIZE_Y_MM = 2.28

    Q1_X = -QUBIT_SPACING_MM
    Q2_X = 0.0
    Q3_X = QUBIT_SPACING_MM
    Q_Y = -0.70

    lp_shared = base.LayoutParams(
        chip_size_x=f"{CHIP_SIZE_X_MM:.2f}mm",
        chip_size_y=f"{CHIP_SIZE_Y_MM:.2f}mm",
        cpw_width="10um",
        cpw_gap="6um",
        feed_fillet="0um",
        ro_fillet="0um",
        pad_width="425um",
        pocket_height="650um",
        prime_width="14um",
        prime_gap="6um",
        second_width="10um",
        second_gap="6um",
        feed_total_length="2.6mm",
        feed_spacing="500um",
        feed_lead_start="0um",
        feed_lead_end="0um",
        ro_spacing="24um",
        ro_lead_start="0um",
        ro_lead_end="200um",
        ro_pad_w=f"{Q_RO_PAD_W_UM[0]}um",
        ro_pad_h=f"{Q_RO_PAD_H_UM[0]}um",
        ro_pad_gap=f"{Q_RO_PAD_GAP_UM[0]}um",
        lp_dx=float(LP_DX_MM[0]),
        ro_total_length=f"{RO_L_MM[0]}mm",
        finger_length=f"{TEE_FINGER_LENGTH_UM[0]}um",
        finger_count=str(int(TEE_FINGER_COUNT[0])),
        cap_gap=f"{TEE_CAP_GAP_UM[0]}um",
        cap_width=f"{TEE_CAP_WIDTH_UM[0]}um",
        cap_distance=f"{TEE_CAP_DISTANCE_UM[0]}um",
    )

    # Match the current 2Q inter-qubit bus baseline.
    bus_length_mm = 5.2

    p3 = Layout3QParams(
        shared=lp_shared,
        bus_total_length=f"{bus_length_mm:.2f}mm",
        bus_width="10um",
        bus_gap="6um",
        bus_lead_start="130um",
        bus_lead_end="130um",
        bus_fillet="24um",
        bus_spacing="145um",
        bus_pad_width=f"{BUS_PAD_W_UM}um",
        bus_pad_height=f"{BUS_PAD_H_UM}um",
        bus_pad_gap=f"{BUS_PAD_GAP_UM}um",
        tee_offset_mm=0.59,
        readout_lane_dx_mm=0.028,
        q2_readout_lane_dx_mm=-0.14,
        q13_lp_outward_offset_mm=0.0,
        swap_tee_ports=True,
        ro1_serpentine_side=-1.0,
        ro2_serpentine_side=1.0,
        ro3_serpentine_side=1.0,
        ro2_serpentine_width_mm=1.12,
        q1_ro_pad_w_um=Q_RO_PAD_W_UM[0],
        q2_ro_pad_w_um=Q_RO_PAD_W_UM[1],
        q3_ro_pad_w_um=Q_RO_PAD_W_UM[2],
        q1_ro_pad_h_um=Q_RO_PAD_H_UM[0],
        q2_ro_pad_h_um=Q_RO_PAD_H_UM[1],
        q3_ro_pad_h_um=Q_RO_PAD_H_UM[2],
        q1_ro_pad_gap_um=Q_RO_PAD_GAP_UM[0],
        q2_ro_pad_gap_um=Q_RO_PAD_GAP_UM[1],
        q3_ro_pad_gap_um=Q_RO_PAD_GAP_UM[2],
        lp1_dx_mm=LP_DX_MM[0],
        lp2_dx_mm=LP_DX_MM[1],
        lp3_dx_mm=LP_DX_MM[2],
        ro1_total_length_mm=RO_L_MM[0],
        ro2_total_length_mm=RO_L_MM[1],
        ro3_total_length_mm=RO_L_MM[2],
        ro1_tee_cap_gap_um=TEE_CAP_GAP_UM[0],
        ro2_tee_cap_gap_um=TEE_CAP_GAP_UM[1],
        ro3_tee_cap_gap_um=TEE_CAP_GAP_UM[2],
        ro1_tee_cap_width_um=TEE_CAP_WIDTH_UM[0],
        ro2_tee_cap_width_um=TEE_CAP_WIDTH_UM[1],
        ro3_tee_cap_width_um=TEE_CAP_WIDTH_UM[2],
        ro1_tee_cap_distance_um=TEE_CAP_DISTANCE_UM[0],
        ro2_tee_cap_distance_um=TEE_CAP_DISTANCE_UM[1],
        ro3_tee_cap_distance_um=TEE_CAP_DISTANCE_UM[2],
        ro1_tee_finger_length_um=TEE_FINGER_LENGTH_UM[0],
        ro2_tee_finger_length_um=TEE_FINGER_LENGTH_UM[1],
        ro3_tee_finger_length_um=TEE_FINGER_LENGTH_UM[2],
        ro1_tee_finger_count=TEE_FINGER_COUNT[0],
        ro2_tee_finger_count=TEE_FINGER_COUNT[1],
        ro3_tee_finger_count=TEE_FINGER_COUNT[2],
        ro1_purcell_stub_length_mm=PURCELL_STUB_LENGTH_MM[0],
        ro2_purcell_stub_length_mm=PURCELL_STUB_LENGTH_MM[1],
        ro3_purcell_stub_length_mm=PURCELL_STUB_LENGTH_MM[2],
        ro1_purcell_stub_offset_mm=PURCELL_STUB_OFFSET_MM[0],
        ro2_purcell_stub_offset_mm=PURCELL_STUB_OFFSET_MM[1],
        ro3_purcell_stub_offset_mm=PURCELL_STUB_OFFSET_MM[2],
        ro1_purcell_stub_width_um=PURCELL_STUB_WIDTH_UM[0],
        ro2_purcell_stub_width_um=PURCELL_STUB_WIDTH_UM[1],
        ro3_purcell_stub_width_um=PURCELL_STUB_WIDTH_UM[2],
        ro1_purcell_stub_gap_um=PURCELL_STUB_GAP_UM[0],
        ro2_purcell_stub_gap_um=PURCELL_STUB_GAP_UM[1],
        ro3_purcell_stub_gap_um=PURCELL_STUB_GAP_UM[2],
    )

    candidates = _candidate_param_grid_3q()
    processed_new = 0
    found_valid = False
    for cand_idx, cand in enumerate(candidates, 1):
        if processed_new >= int(MAX_SAMPLES):
            break

        lj_run = tuple(float(x) for x in cand.get("lj_nh", LJ_NH))
        ro_l_run = tuple(float(x) for x in cand.get("ro_l_mm", RO_L_MM))
        tee_gap_run = tuple(float(x) for x in cand.get("tee_cap_gap_um", TEE_CAP_GAP_UM))
        tee_width_run = tuple(float(x) for x in cand.get("tee_cap_width_um", TEE_CAP_WIDTH_UM))
        tee_distance_run = tuple(float(x) for x in cand.get("tee_cap_distance_um", TEE_CAP_DISTANCE_UM))
        stub_len_run = tuple(float(x) for x in cand.get("stub_length_mm", PURCELL_STUB_LENGTH_MM))
        stub_offset_run = tuple(float(x) for x in cand.get("stub_offset_mm", PURCELL_STUB_OFFSET_MM))
        ro_pad_w_run = tuple(float(x) for x in cand.get("ro_pad_w_um", Q_RO_PAD_W_UM))
        ro_pad_h_run = tuple(float(x) for x in cand.get("ro_pad_h_um", Q_RO_PAD_H_UM))
        ro_pad_gap_run = tuple(float(x) for x in cand.get("ro_pad_gap_um", Q_RO_PAD_GAP_UM))
        lp_dx_run = tuple(float(x) for x in cand.get("lp_dx_mm", LP_DX_MM))

        sample_id = (
            f"chip_3q_sweep_{cand_idx:04d}_"
            f"{cand['sample_suffix']}_"
            f"m{os.environ.get('METAL_3Q_HFSS_N_MODES', '12')}"
            f"_p{os.environ.get('METAL_3Q_HFSS_MAX_PASSES', '6')}"
            f"_fmin{str(os.environ.get('METAL_3Q_HFSS_MIN_FREQ_GHZ', '3.0')).replace('.', 'p')}"
            f"_df{str(os.environ.get('METAL_3Q_HFSS_MAX_DELTA_F_GHZ', '0.1')).replace('.', 'p')}"
        )

        if sample_id in existing_ids:
            continue

        p3_run = copy.deepcopy(p3)
        p3_run.ro1_total_length_mm = ro_l_run[0]
        p3_run.ro2_total_length_mm = ro_l_run[1]
        p3_run.ro3_total_length_mm = ro_l_run[2]
        p3_run.ro1_tee_cap_gap_um = tee_gap_run[0]
        p3_run.ro2_tee_cap_gap_um = tee_gap_run[1]
        p3_run.ro3_tee_cap_gap_um = tee_gap_run[2]
        p3_run.ro1_tee_cap_width_um = tee_width_run[0]
        p3_run.ro2_tee_cap_width_um = tee_width_run[1]
        p3_run.ro3_tee_cap_width_um = tee_width_run[2]
        p3_run.ro1_tee_cap_distance_um = tee_distance_run[0]
        p3_run.ro2_tee_cap_distance_um = tee_distance_run[1]
        p3_run.ro3_tee_cap_distance_um = tee_distance_run[2]
        p3_run.ro1_purcell_stub_length_mm = stub_len_run[0]
        p3_run.ro2_purcell_stub_length_mm = stub_len_run[1]
        p3_run.ro3_purcell_stub_length_mm = stub_len_run[2]
        p3_run.ro1_purcell_stub_offset_mm = stub_offset_run[0]
        p3_run.ro2_purcell_stub_offset_mm = stub_offset_run[1]
        p3_run.ro3_purcell_stub_offset_mm = stub_offset_run[2]
        p3_run.q1_ro_pad_w_um = ro_pad_w_run[0]
        p3_run.q2_ro_pad_w_um = ro_pad_w_run[1]
        p3_run.q3_ro_pad_w_um = ro_pad_w_run[2]
        p3_run.q1_ro_pad_h_um = ro_pad_h_run[0]
        p3_run.q2_ro_pad_h_um = ro_pad_h_run[1]
        p3_run.q3_ro_pad_h_um = ro_pad_h_run[2]
        p3_run.q1_ro_pad_gap_um = ro_pad_gap_run[0]
        p3_run.q2_ro_pad_gap_um = ro_pad_gap_run[1]
        p3_run.q3_ro_pad_gap_um = ro_pad_gap_run[2]
        p3_run.lp1_dx_mm = lp_dx_run[0]
        p3_run.lp2_dx_mm = lp_dx_run[1]
        p3_run.lp3_dx_mm = lp_dx_run[2]

        print(f"[3Q SWEEP] running {sample_id}", flush=True)
        try:
            design = base.designs.DesignPlanar({}, True)
            print(f"[3Q SWEEP] populate design {sample_id}", flush=True)
            populate_design_3qubit(
                design,
                q1_x_mm=Q1_X,
                q1_y_mm=Q_Y,
                q2_x_mm=Q2_X,
                q2_y_mm=Q_Y,
                q3_x_mm=Q3_X,
                q3_y_mm=Q_Y,
                p3=p3_run,
            )
            print(f"[3Q SWEEP] run analysis do_hfss={DO_HFSS} do_q3d={DO_Q3D}", flush=True)

            result = run_analysis_pipeline_3q(
                design,
                sample_id,
                root_path=str(dataset_root),
                do_hfss=DO_HFSS,
                do_q3d=DO_Q3D,
                lj_nh=lj_run,
                cj_fF_user=CJ_FF,
                inputs_sweep={
                    "candidate_index": int(cand_idx),
                    "spacing_mm": float(QUBIT_SPACING_MM),
                    "q_y_mm": float(Q_Y),
                    "chip_size_x_mm": float(CHIP_SIZE_X_MM),
                    "chip_size_y_mm": float(CHIP_SIZE_Y_MM),
                    "Lj_nH": list(lj_run),
                    "ro_l_mm": list(ro_l_run),
                    "tee_cap_gap_um": list(tee_gap_run),
                    "tee_cap_width_um": list(tee_width_run),
                    "tee_cap_distance_um": list(tee_distance_run),
                    "purcell_stub_length_mm": list(stub_len_run),
                    "purcell_stub_offset_mm": list(stub_offset_run),
                    "ro_pad_w_um": list(ro_pad_w_run),
                    "ro_pad_h_um": list(ro_pad_h_run),
                    "ro_pad_gap_um": list(ro_pad_gap_run),
                    "lp_dx_mm": list(lp_dx_run),
                    "tee_offset_mm": float(p3_run.tee_offset_mm),
                    "readout_lane_dx_mm": float(p3_run.readout_lane_dx_mm),
                    "q2_readout_lane_dx_mm": float(p3_run.q2_readout_lane_dx_mm),
                    "q13_lp_outward_offset_mm": float(p3_run.q13_lp_outward_offset_mm),
                    "ro_serpentine_side": [
                        float(p3_run.ro1_serpentine_side),
                        float(p3_run.ro2_serpentine_side),
                        float(p3_run.ro3_serpentine_side),
                    ],
                    "ro2_serpentine_width_mm": float(p3_run.ro2_serpentine_width_mm),
                    "manual_ro": {
                        "edge_margin_mm": _env_float("METAL_3Q_MANUAL_RO_EDGE_MARGIN_MM", 0.05, lo=0.04, hi=0.6),
                        "inner_offset_mm": _env_float("METAL_3Q_MANUAL_RO_INNER_OFFSET_MM", 0.070, lo=0.05, hi=0.7),
                        "q2_inner_offset_mm": _env_float("METAL_3Q_MANUAL_RO_Q2_INNER_OFFSET_MM", 0.070, lo=0.05, hi=0.7),
                        "bottom_clearance_mm": _env_float("METAL_3Q_MANUAL_RO_BOTTOM_CLEARANCE_MM", 0.24, lo=0.04, hi=0.5),
                        "spacing_mm": min(
                            float(base._value_to_mm(getattr(p3_run.shared, "ro_spacing", "28um")) or 0.028),
                            _env_float("METAL_3Q_MANUAL_RO_SPACING_MM", 0.030, lo=0.024, hi=0.18),
                        ),
                    },
                    "hfss_n_modes": _env_int("METAL_3Q_HFSS_N_MODES", 12, lo=4, hi=30),
                    "hfss_max_passes": _env_int("METAL_3Q_HFSS_MAX_PASSES", 6, lo=1, hi=30),
                    "hfss_max_delta_f": _env_float("METAL_3Q_HFSS_MAX_DELTA_F_GHZ", 0.1, lo=0.001, hi=1.0),
                },
                inputs_qubits={
                    "Q1": {"x_mm": float(Q1_X), "y_mm": float(Q_Y)},
                    "Q2": {"x_mm": float(Q2_X), "y_mm": float(Q_Y)},
                    "Q3": {"x_mm": float(Q3_X), "y_mm": float(Q_Y)},
                },
                gds_subdir="gds_tmp",
                json_subdir="json_tmp",
                layout_params=p3_run,
            )

            chip = result.get("chip", {})
            q1 = result.get("qubits", {}).get("Q1", {})
            q2 = result.get("qubits", {}).get("Q2", {})
            q3 = result.get("qubits", {}).get("Q3", {})
            ro1 = result.get("resonators", {}).get("readout1", {})
            ro2 = result.get("resonators", {}).get("readout2", {})
            ro3 = result.get("resonators", {}).get("readout3", {})
            valid, reasons, metrics = validate_3q_candidate(result)

            out = dict(
                sample_id=sample_id,
                q1_x_mm=Q1_X,
                q1_y_mm=Q_Y,
                q2_x_mm=Q2_X,
                q2_y_mm=Q_Y,
                q3_x_mm=Q3_X,
                q3_y_mm=Q_Y,
                fq1_GHz=q1.get("f01_epr_GHz"),
                fq2_GHz=q2.get("f01_epr_GHz"),
                fq3_GHz=q3.get("f01_epr_GHz"),
                fr1_GHz=ro1.get("f_GHz"),
                fr2_GHz=ro2.get("f_GHz"),
                fr3_GHz=ro3.get("f_GHz"),
                chi12_MHz=chip.get("chi12_MHz"),
                chi23_MHz=chip.get("chi23_MHz"),
                chi13_MHz=chip.get("chi13_MHz"),
                q1_chi_over_k=metrics["Q1"].get("chi_over_k"),
                q2_chi_over_k=metrics["Q2"].get("chi_over_k"),
                q3_chi_over_k=metrics["Q3"].get("chi_over_k"),
                q1_k_MHz=metrics["Q1"].get("k_MHz"),
                q2_k_MHz=metrics["Q2"].get("k_MHz"),
                q3_k_MHz=metrics["Q3"].get("k_MHz"),
                q1_kext_MHz=metrics["Q1"].get("kext_MHz"),
                q2_kext_MHz=metrics["Q2"].get("kext_MHz"),
                q3_kext_MHz=metrics["Q3"].get("kext_MHz"),
                q1_kint_MHz=metrics["Q1"].get("kint_MHz"),
                q2_kint_MHz=metrics["Q2"].get("kint_MHz"),
                q3_kint_MHz=metrics["Q3"].get("kint_MHz"),
                q1_T1_us=metrics["Q1"].get("T1_us"),
                q2_T1_us=metrics["Q2"].get("T1_us"),
                q3_T1_us=metrics["Q3"].get("T1_us"),
                valid=bool(valid),
                invalid_reason="; ".join(reasons),
                status=result.get("status", ""),
                error=(
                    result.get("meta", {}).get("hfss_error", "")
                    or result.get("meta", {}).get("q3d_error", "")
                    or result.get("meta", {}).get("gds_error", "")
                    or ""
                ),
            )
            write_row(out)
            processed_new += 1
            found_valid = bool(valid)
            print(f"[3Q SWEEP] valid={valid} reason={'; '.join(reasons) if reasons else 'ok'}", flush=True)
            if found_valid:
                break

        except Exception as e:
            out = dict(
                sample_id=sample_id,
                q1_x_mm=Q1_X, q1_y_mm=Q_Y,
                q2_x_mm=Q2_X, q2_y_mm=Q_Y,
                q3_x_mm=Q3_X, q3_y_mm=Q_Y,
                valid=False,
                invalid_reason=str(e),
                status="ERROR",
                error=str(e),
            )
            write_row(out)
            processed_new += 1
        finally:
            base._mpl_cleanup()

    print(f"[Done] 3Q sweep processed_new={processed_new} found_valid={found_valid}")
