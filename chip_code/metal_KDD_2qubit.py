# -*- coding: utf-8 -*-
"""2-qubit dataset generator (qiskit-metal + pyEPR).

This script is adapted from `metal_KDD.py` and extends it to a 2-qubit layout:
- Two TransmonPocket qubits: Q1, Q2
- Two independent readout resonators (each with its own tee + feed launchpad)
- HFSS/EPR: extracts f01/alpha/chi, picks 2 qubit modes + 2 resonator modes
- Q3D/LOM: extracts Cin for each tee and estimates kappa_e, plus Ceff for each qubit

Notes:
- Default parameter sweep keeps Q1 fixed and sweeps Q2 position (you can change it).
- This file avoids modifying `metal_KDD.py` and imports shared helpers from it.
"""

from __future__ import annotations

import copy
import importlib
import os
import sys
import csv
import json
import math
import time
import threading
import subprocess
import gc
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any, Callable, Dict as TDict, Iterator, List, Optional, Tuple, cast


def _bootstrap_expected_python() -> None:
    expected = Path(__file__).resolve().parent.parent / "Scripts" / "python.exe"
    if not expected.is_file() or os.environ.get("METAL_2Q_BOOTSTRAPPED_PYTHON") == "1":
        return
    current = Path(sys.executable).resolve()
    if os.path.normcase(str(current)) == os.path.normcase(str(expected.resolve())):
        return
    os.environ["METAL_2Q_BOOTSTRAPPED_PYTHON"] = "1"
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
from numpy.typing import NDArray
import pandas as pd
from shapely.ops import unary_union

import metal_KDD as base  # pyright: ignore[reportImplicitRelativeImport]


DEFAULT_2Q_ANSYS_PROJECT_FILE = r"C:\Users\lenovo\Documents\Ansoft\Project2.aedt"


def _components_override(env_var: str) -> Optional[List[str]]:
    raw = os.environ.get(env_var, "").strip()
    if not raw:
        return None
    if raw.lower() in {"min", "minimal", "q1q2"}:
        return ["Q1", "Q2"]
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    return parts or None


@contextmanager
def _ansys_watchdog(timeout_s: float, *, label: str) -> Iterator[None]:
    try:
        timeout = float(timeout_s)
    except Exception:
        timeout = 0.0

    if not math.isfinite(timeout) or timeout <= 0.0:
        yield
        return

    stop = threading.Event()

    def _worker() -> None:
        if stop.wait(timeout):
            return
        try:
            print(f"[WATCHDOG] {label} timeout_s={timeout:g} -> ansys reset", flush=True)
        except Exception:
            pass
        try:
            base._ansys_best_effort_reset()
        except Exception:
            pass

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    try:
        yield
    finally:
        stop.set()


def _print_ansys_messages(renderer, label: str, *, level: int = 2, limit: int = 50) -> None:
    try:
        pinfo = getattr(renderer, "pinfo", None)
        desktop = getattr(pinfo, "desktop", None) if pinfo else None
        if desktop is None:
            return
        proj = getattr(pinfo, "project_name", "") if pinfo else ""
        des = getattr(pinfo, "design_name", "") if pinfo else ""
        msgs = desktop.get_messages(proj, des, level)
        if msgs:
            for m in list(msgs)[:limit]:
                print(f"[{label}] {m}", flush=True)
    except Exception:
        pass


def _collect_ansys_messages(renderer, *, level: int = 2, limit: int = 200) -> List[str]:
    try:
        pinfo = getattr(renderer, "pinfo", None)
        desktop = getattr(pinfo, "desktop", None) if pinfo else None
        if desktop is None:
            return []
        proj = getattr(pinfo, "project_name", "") if pinfo else ""
        des = getattr(pinfo, "design_name", "") if pinfo else ""
        msgs = desktop.get_messages(proj, des, level)
        if not msgs:
            return []
        return [str(m) for m in list(msgs)[:limit]]
    except Exception:
        return []


def _has_deterministic_geometry_error(renderer) -> bool:
    patterns = (
        "pk_body_sweep_topol_self_int_c",
        "sweepalongpath operation",
        "body could not be created",
        "contains null body",
        "pk_vertex_make_blend",
        "short segments",
    )
    for msg in _collect_ansys_messages(renderer, level=2, limit=200):
        msg_l = str(msg).lower()
        if any(pat in msg_l for pat in patterns):
            return True
    return False


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


# -------------------------
# layout
# -------------------------
@dataclass
class Layout2QParams:
    """2Q layout settings.

    Uses `base.LayoutParams` for per-readout geometry and adds readout naming.
    """

    shared: base.LayoutParams = field(
        default_factory=lambda: base.LayoutParams(
            chip_size_x="12mm",
            chip_size_y="12mm",
            cpw_width="10um",
            cpw_gap="6um",
            feed_fillet="25um",
            ro_fillet="25um",
            pad_width="425um",
            pocket_height="650um",
            ro_pad_w="360um",
            ro_pad_h="360um",
            ro_pad_gap="1.0um",
            lp_dx=2.4,
            prime_width="14um",
            prime_gap="6um",
            second_width="10um",
            second_gap="6um",
            cap_gap="1.2um",
            cap_width="18um",
            finger_length="160um",
            finger_count="12",
            cap_distance="40um",
            feed_total_length="5.5mm",
            feed_spacing="280um",
            feed_lead_start="250um",
            feed_lead_end="250um",
            ro_total_length="10.2mm",
            ro_spacing="260um",
            ro_lead_start="250um",
            ro_lead_end="250um",
        )
    )

    # How far each readout chain is offset from its qubit readout pad.
    # (Keep non-zero to reduce collisions when Q1 and Q2 are close.)
    ro1_dx_mm: float = 3.2
    ro1_dy_mm: float = -0.35
    ro2_dx_mm: float = 3.2
    ro2_dy_mm: float = 0.35

    # Distance from each qubit readout pin to its tee center.
    tee_offset_mm: float = 0.8

    ro1_total_length_mm: Optional[float] = 9.8
    ro2_total_length_mm: Optional[float] = 10.6
    ro1_tee_cap_gap_um: Optional[float] = 1.2
    ro2_tee_cap_gap_um: Optional[float] = 1.2
    ro1_tee_cap_width_um: Optional[float] = 18.0
    ro2_tee_cap_width_um: Optional[float] = 18.0
    ro1_tee_cap_distance_um: Optional[float] = 40.0
    ro2_tee_cap_distance_um: Optional[float] = 40.0
    ro1_tee_finger_length_um: Optional[int] = 160
    ro2_tee_finger_length_um: Optional[int] = 160
    ro1_tee_finger_count: Optional[int] = 12
    ro2_tee_finger_count: Optional[int] = 12

    q1_pad_width_um: Optional[float] = 425.0
    q2_pad_width_um: Optional[float] = 425.0
    q1_pocket_height_um: Optional[float] = 650.0
    q2_pocket_height_um: Optional[float] = 650.0
    q1_ro_pad_w_um: Optional[float] = 420.0
    q2_ro_pad_w_um: Optional[float] = 620.0
    q1_ro_pad_h_um: Optional[float] = 420.0
    q2_ro_pad_h_um: Optional[float] = 620.0
    q1_ro_pad_gap_um: Optional[float] = 0.8
    q2_ro_pad_gap_um: Optional[float] = 0.45
    q1_bus_pad_w_um: Optional[float] = 270.0
    q2_bus_pad_w_um: Optional[float] = 270.0
    q1_bus_pad_h_um: Optional[float] = 270.0
    q2_bus_pad_h_um: Optional[float] = 270.0
    q1_bus_pad_gap_um: Optional[float] = 1.2
    q2_bus_pad_gap_um: Optional[float] = 1.2

    # Launchpad distance from tee center (mm), per readout chain.
    # If None, uses shared.lp_dx.
    lp1_dx_mm: Optional[float] = 2.4
    lp2_dx_mm: Optional[float] = 2.4

    # Optional Purcell-filter stub, implemented as a grounded CPW branch from the tee.
    # Default None/0 keeps the existing tee-only topology unchanged.
    ro1_purcell_stub_length_mm: Optional[float] = 0.0
    ro2_purcell_stub_length_mm: Optional[float] = 0.0
    ro1_purcell_stub_offset_mm: Optional[float] = 1.2
    ro2_purcell_stub_offset_mm: Optional[float] = 1.2
    ro1_purcell_stub_width_um: Optional[float] = 10.0
    ro2_purcell_stub_width_um: Optional[float] = 10.0
    ro1_purcell_stub_gap_um: Optional[float] = 6.0
    ro2_purcell_stub_gap_um: Optional[float] = 6.0

    # If true, swap tee ports so the resonator uses the prime pin and the feed uses second_end.
    # This is purely a layout/topology toggle; defaults to the original behavior.
    swap_tee_ports: bool = True

    # Bus coupling
    bus_width: str = "10um"
    bus_gap: str = "6um"
    bus_pad_gap_um: Optional[float] = None
    bus_lead_start: str = "250um"
    bus_lead_end: str = "250um"
    bus_fillet: str = "50um"
    bus_spacing: str = "250um"
    bus_total_length: str = "5.0mm"


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
    with_bus: bool = True,
    bus_loc_W: int = +1,
    bus_loc_H: int = -1,
    readout_loc_W: int = +1,
    readout_loc_H: int = +1,
    bus_pad_width: Optional[str] = None,
    bus_pad_height: Optional[str] = None,
    bus_pad_gap: Optional[str] = None,
):
    """
    Create a TransmonPocket qubit with optional bus coupling pad.

    Parameters:
        orientation: Qubit rotation in degrees (0, 90, 180, 270)
        bus_loc_W: +1 for right side, -1 for left side (in qubit's local frame)
        bus_loc_H: +1 for top, -1 for bottom (in qubit's local frame)
        readout_loc_W: +1 for right side, -1 for left side for readout pad
        readout_loc_H: +1 for top, -1 for bottom for readout pad

    For proper bus coupling between two qubits placed horizontally:
        - Q1 (left qubit): bus_loc_W=+1 (pad faces right toward Q2)
        - Q2 (right qubit): bus_loc_W=-1 (pad faces left toward Q1)

    For readout pads to avoid collision when qubits are close:
        - Q1: readout on top-right (+1, +1)
        - Q2: readout on top-left (-1, +1) or adjust based on layout
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

    if with_bus:
        # bus pad location is now configurable per qubit
        # loc_W: +1 = right side, -1 = left side
        # loc_H: +1 = top, -1 = bottom
        connection_pads["bus"] = base.Dict(
            loc_W=bus_loc_W,
            loc_H=bus_loc_H,
            pad_width=bus_pad_width if bus_pad_width is not None else p.ro_pad_w,
            pad_height=bus_pad_height if bus_pad_height is not None else p.ro_pad_h,
            pad_gap=bus_pad_gap if bus_pad_gap is not None else p.ro_pad_gap,
        )

    q = base.TransmonPocket(
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
        x = float(design.components[qname].pins[pin]["middle"][0])
        y = float(design.components[qname].pins[pin]["middle"][1])
        return x, y
    except Exception:
        # Fallback: origin-ish. Caller typically has x_mm/y_mm already.
        return 0.0, 0.0


def _try_get_component_option_mm(design, comp_name: str, option_name: str) -> Optional[float]:
    try:
        raw = design.components[comp_name].options[option_name]
        return float(base._value_to_mm(raw))
    except Exception:
        return None


def _add_connector_pad_subtract_cutouts(design, qnames: Tuple[str, ...] = ("Q1", "Q2")) -> None:
    """Add local ground cutouts for TransmonPocket connector-pad metal outside rect_pk.

    Q3D's AutoIdentifyNets is sensitive to positive metal that extends outside
    the qubit pocket subtract.  Keeping the official-sized pocket while adding
    only the missing local cutouts preserves the HFSS geometry better than
    inflating the entire pocket.
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


def _chip_center_xy_mm(design) -> Tuple[float, float]:
    try:
        main_chip = design.chips.main
        size = getattr(main_chip, "size", {}) or {}
        cx = base._value_to_mm(size.get("center_x", 0.0))
        cy = base._value_to_mm(size.get("center_y", 0.0))
        return float(cx), float(cy)
    except Exception:
        return 0.0, 0.0


def _route_inward_meander_asymmetry_um(
    *,
    start_xy: Tuple[float, float],
    end_xy: Tuple[float, float],
    chip_center_xy: Tuple[float, float],
    spacing: object,
) -> str:
    sx, sy = float(start_xy[0]), float(start_xy[1])
    ex, ey = float(end_xy[0]), float(end_xy[1])
    cx, cy = float(chip_center_xy[0]), float(chip_center_xy[1])

    dx = ex - sx
    dy = ey - sy
    seg_len = math.hypot(dx, dy)
    if seg_len <= 1e-9:
        return "0um"

    forward_x = dx / seg_len
    forward_y = dy / seg_len
    side_x = -forward_y
    side_y = forward_x

    mid_x = 0.5 * (sx + ex)
    mid_y = 0.5 * (sy + ey)
    to_center_x = cx - mid_x
    to_center_y = cy - mid_y
    proj_mm = to_center_x * side_x + to_center_y * side_y

    spacing_mm = max(0.0, base._value_to_mm(spacing))
    if spacing_mm <= 0.0:
        return "0um"

    max_bias_um = spacing_mm * 1000.0 * 0.35
    desired_bias_um = proj_mm * 1000.0 * 0.5
    if abs(desired_bias_um) < 1.0:
        return "0um"

    bias_um = max(-max_bias_um, min(max_bias_um, desired_bias_um))
    if abs(bias_um) < 1.0:
        return "0um"
    return f"{bias_um:.1f}um"


def _pick_launchpad_params(
    *,
    coupler_x_mm: float,
    coupler_y_mm: float,
    tee_orientation: str,
    p: base.LayoutParams,
    edge_margin_mm: float = 0.5,
) -> Tuple[int, float]:
    ori = int(float(tee_orientation)) % 360
    lp_dx = float(getattr(p, "lp_dx", 0.0) or 0.0)

    chip_x = base._value_to_mm(getattr(p, "chip_size_x", 0.0))
    chip_y = base._value_to_mm(getattr(p, "chip_size_y", 0.0))
    half_x = (chip_x / 2.0) if chip_x else 0.0
    half_y = (chip_y / 2.0) if chip_y else 0.0

    if ori in (0, 180):
        side = -1 if float(coupler_x_mm) < 0 else 1
        lp_direction = side if ori == 0 else -side
        if half_x > 0:
            if ori == 0:
                max_dx = (
                    (half_x - edge_margin_mm) - coupler_x_mm
                    if lp_direction > 0
                    else coupler_x_mm - (-half_x + edge_margin_mm)
                )
            else:
                max_dx = (
                    coupler_x_mm - (-half_x + edge_margin_mm)
                    if lp_direction > 0
                    else (half_x - edge_margin_mm) - coupler_x_mm
                )
            if max_dx > 0:
                lp_dx = min(lp_dx, float(max_dx))
    else:
        side = -1 if float(coupler_y_mm) < 0 else 1
        lp_direction = side if ori == 90 else -side
        if half_y > 0:
            if ori == 90:
                max_dx = (
                    (half_y - edge_margin_mm) - coupler_y_mm
                    if lp_direction > 0
                    else coupler_y_mm - (-half_y + edge_margin_mm)
                )
            else:
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
    lp_direction: int = -1,
    swap_tee_ports: bool = False,
    lp_dx_mm: Optional[float] = None,
):
    """Add one launchpad + tee + feed meander + resonator meander.

    Parameters:
        tee_orientation: "0" (down), "90" (right), "180" (up), "270" (left)
        lp_direction: direction to place launchpad relative to tee
            -1 = to the left/down (depends on orientation)
            +1 = to the right/up
    """
    lp_dx = float(lp_dx_mm) if lp_dx_mm is not None else float(p.lp_dx)
    # Calculate launchpad position based on tee orientation
    ori = int(float(tee_orientation)) % 360
    if ori == 0:  # tee faces right
        lp_x = coupler_x_mm + lp_direction * lp_dx
        lp_y = coupler_y_mm
        lp_ori = "0" if lp_direction < 0 else "180"
    elif ori == 90:  # tee faces up
        lp_x = coupler_x_mm
        lp_y = coupler_y_mm + lp_direction * lp_dx
        lp_ori = "90" if lp_direction < 0 else "270"
    elif ori == 180:  # tee faces left
        lp_x = coupler_x_mm - lp_direction * lp_dx
        lp_y = coupler_y_mm
        lp_ori = "180" if lp_direction < 0 else "0"
    else:  # ori == 270, tee faces down
        lp_x = coupler_x_mm
        lp_y = coupler_y_mm - lp_direction * lp_dx
        lp_ori = "270" if lp_direction < 0 else "90"

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

    if ori == 0:
        tee_prime_pin = "prime_start" if lp_x < coupler_x_mm else "prime_end"
    elif ori == 180:
        tee_prime_pin = "prime_start" if lp_x > coupler_x_mm else "prime_end"
    elif ori == 90:
        tee_prime_pin = "prime_start" if lp_y < coupler_y_mm else "prime_end"
    else:
        tee_prime_pin = "prime_start" if lp_y > coupler_y_mm else "prime_end"

    feed_end_pin = "second_end" if swap_tee_ports else tee_prime_pin

    def _best_prime_pin_toward_target(target_xy: Tuple[float, float]) -> str:
        """Pick the tee prime pin whose normal points toward the target."""
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

    if swap_tee_ports:
        # When using the prime line for the resonator, pick the pin that points toward the qubit.
        qx, qy = _try_get_pin_xy(design, qname, q_pin)
        res_start_pin = _best_prime_pin_toward_target((qx, qy))
    else:
        res_start_pin = "second_end"

    purcell_stub_length_mm = float(getattr(p, "purcell_stub_length_mm", 0.0) or 0.0)
    purcell_stub_offset_mm = float(getattr(p, "purcell_stub_offset_mm", max(0.8, 0.6 * lp_dx)) or max(0.8, 0.6 * lp_dx))
    purcell_stub_width = getattr(p, "purcell_stub_width", p.cpw_width)
    purcell_stub_gap = getattr(p, "purcell_stub_gap", p.cpw_gap)
    filter_components: List[str] = []
    feed_target_component = tee.name
    feed_target_pin = feed_end_pin

    if purcell_stub_length_mm > 0.0:
        filter_step_mm = min(max(0.6, purcell_stub_offset_mm), max(0.6, abs(lp_dx) - 0.25))
        filter_x = coupler_x_mm + ((lp_x - coupler_x_mm) / max(abs(lp_dx), 1e-9)) * filter_step_mm
        filter_y = coupler_y_mm + ((lp_y - coupler_y_mm) / max(abs(lp_dx), 1e-9)) * filter_step_mm

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
                fillet=p.feed_fillet,
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

    base.RouteMeander(
        design,
        f"RO_FEED{suffix}",
        options=base.Dict(
            total_length=p.feed_total_length,
            fillet=p.feed_fillet,
            trace_width=p.cpw_width,
            trace_gap=p.cpw_gap,
            meander=base.Dict(spacing=p.feed_spacing),
            lead=base.Dict(start_straight=p.feed_lead_start, end_straight=p.feed_lead_end),
            pin_inputs=base.Dict(
                start_pin=base.Dict(component=lp.name, pin="tie"),
                end_pin=base.Dict(component=feed_target_component, pin=feed_target_pin),
            ),
        ),
        type="CPW",
    )

    ro_mm = base._mm_str_to_mm(p.ro_total_length)
    ro_mm = base._force_meander_total_len(ro_mm, p.ro_lead_start, p.ro_lead_end, p.ro_spacing)

    base.sanitize_center_readout_pin(design, qname, q_pin)
    qx, qy = _try_get_pin_xy(design, qname, q_pin)
    try:
        start_pin = tee.pins[res_start_pin]
        start_x = float(start_pin["middle"][0])
        start_y = float(start_pin["middle"][1])
    except Exception:
        start_x = float(coupler_x_mm)
        start_y = float(coupler_y_mm)

    chip_center_x_mm, chip_center_y_mm = _chip_center_xy_mm(design)
    ro_asymmetry = _route_inward_meander_asymmetry_um(
        start_xy=(start_x, start_y),
        end_xy=(qx, qy),
        chip_center_xy=(chip_center_x_mm, chip_center_y_mm),
        spacing=p.ro_spacing,
    )
    meander_ro = base.Dict(spacing=p.ro_spacing)
    if ro_asymmetry != "0um":
        meander_ro["asymmetry"] = ro_asymmetry

    base.RouteMeander(
        design,
        f"RO_RES{suffix}",
        options=base.Dict(
            total_length=f"{ro_mm:.3f}mm",
            fillet=p.ro_fillet,
            trace_width=p.cpw_width,
            trace_gap=p.cpw_gap,
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


def populate_design_2qubit(
    design,
    *,
    q1_x_mm: float,
    q1_y_mm: float,
    q2_x_mm: float,
    q2_y_mm: float,
    p2: Layout2QParams,
    lj_vars: Tuple[str, str] = ("Lj1", "Lj2"),
    cj_vars: Tuple[str, str] = ("Cj1", "Cj2"),
) -> None:
    """Populate planar design with Q1/Q2 and two readout resonators.

    Layout strategy:
        - Readout pads: perpendicular to the Q1-Q2 connection line, both same direction
        - Bus pads: face each other along the Q1-Q2 connection line
        - Tee placement: offset only in the perpendicular direction

    For horizontal Q1-Q2 (left-right):
        - Both readout pads on top side (loc_H=+1), pointing upward
        - Bus pads on bottom side facing each other along X
        - Tees offset only in Y direction (perpendicular to coupling)

    For vertical Q1-Q2 (top-bottom):
        - Both readout pads on left side (loc_W=-1), pointing left
        - Bus pads on right side facing each other along Y
        - Tees offset only in X direction (perpendicular to coupling)
    """

    design.overwrite_enabled = True
    p = p2.shared

    p_q1 = base.LayoutParams(**{k: getattr(p, k) for k in getattr(p, "__dataclass_fields__", {}).keys()})
    p_q2 = base.LayoutParams(**{k: getattr(p, k) for k in getattr(p, "__dataclass_fields__", {}).keys()})
    p_ro1 = base.LayoutParams(**{k: getattr(p, k) for k in getattr(p, "__dataclass_fields__", {}).keys()})
    p_ro2 = base.LayoutParams(**{k: getattr(p, k) for k in getattr(p, "__dataclass_fields__", {}).keys()})

    layout_style = os.environ.get("METAL_2Q_LAYOUT_STYLE", "official_singlepad").strip().lower()
    official_singlepad = layout_style in {"official_singlepad", "official", "official_2q", "official-singlepad"}

    if official_singlepad:
        # Match the official 2Q tutorial qubit defaults while keeping this script's
        # single external readout branch instead of adding a separate charge line.
        if p2.q1_pad_width_um is None:
            p_q1.pad_width = "425um"
        if p2.q2_pad_width_um is None:
            p_q2.pad_width = "425um"
        if p2.q1_pocket_height_um is None:
            p_q1.pocket_height = "650um"
        if p2.q2_pocket_height_um is None:
            p_q2.pocket_height = "650um"

    if p2.q1_pad_width_um is not None:
        p_q1.pad_width = f"{float(p2.q1_pad_width_um)}um"
    if p2.q2_pad_width_um is not None:
        p_q2.pad_width = f"{float(p2.q2_pad_width_um)}um"
    if p2.q1_pocket_height_um is not None:
        p_q1.pocket_height = f"{float(p2.q1_pocket_height_um)}um"
    if p2.q2_pocket_height_um is not None:
        p_q2.pocket_height = f"{float(p2.q2_pocket_height_um)}um"
    if p2.q1_ro_pad_w_um is not None:
        p_q1.ro_pad_w = f"{float(p2.q1_ro_pad_w_um)}um"
    if p2.q2_ro_pad_w_um is not None:
        p_q2.ro_pad_w = f"{float(p2.q2_ro_pad_w_um)}um"
    if p2.q1_ro_pad_h_um is not None:
        p_q1.ro_pad_h = f"{float(p2.q1_ro_pad_h_um)}um"
    if p2.q2_ro_pad_h_um is not None:
        p_q2.ro_pad_h = f"{float(p2.q2_ro_pad_h_um)}um"
    if p2.q1_ro_pad_gap_um is not None:
        p_q1.ro_pad_gap = f"{float(p2.q1_ro_pad_gap_um)}um"
    if p2.q2_ro_pad_gap_um is not None:
        p_q2.ro_pad_gap = f"{float(p2.q2_ro_pad_gap_um)}um"
    if p2.ro1_total_length_mm is not None:
        p_ro1.ro_total_length = f"{float(p2.ro1_total_length_mm)}mm"
    if p2.ro2_total_length_mm is not None:
        p_ro2.ro_total_length = f"{float(p2.ro2_total_length_mm)}mm"

    if p2.ro1_tee_cap_gap_um is not None:
        p_ro1.cap_gap = f"{float(p2.ro1_tee_cap_gap_um)}um"
    if p2.ro2_tee_cap_gap_um is not None:
        p_ro2.cap_gap = f"{float(p2.ro2_tee_cap_gap_um)}um"
    if p2.ro1_tee_cap_width_um is not None:
        p_ro1.cap_width = f"{float(p2.ro1_tee_cap_width_um)}um"
    if p2.ro2_tee_cap_width_um is not None:
        p_ro2.cap_width = f"{float(p2.ro2_tee_cap_width_um)}um"
    if p2.ro1_tee_cap_distance_um is not None:
        p_ro1.cap_distance = f"{float(p2.ro1_tee_cap_distance_um)}um"
    if p2.ro2_tee_cap_distance_um is not None:
        p_ro2.cap_distance = f"{float(p2.ro2_tee_cap_distance_um)}um"
    if p2.ro1_tee_finger_length_um is not None:
        p_ro1.finger_length = f"{int(p2.ro1_tee_finger_length_um)}um"
    if p2.ro2_tee_finger_length_um is not None:
        p_ro2.finger_length = f"{int(p2.ro2_tee_finger_length_um)}um"
    if p2.ro1_tee_finger_count is not None:
        p_ro1.finger_count = str(int(p2.ro1_tee_finger_count))
    if p2.ro2_tee_finger_count is not None:
        p_ro2.finger_count = str(int(p2.ro2_tee_finger_count))
    if p2.ro1_purcell_stub_length_mm is not None:
        setattr(p_ro1, "purcell_stub_length_mm", float(p2.ro1_purcell_stub_length_mm))
    if p2.ro2_purcell_stub_length_mm is not None:
        setattr(p_ro2, "purcell_stub_length_mm", float(p2.ro2_purcell_stub_length_mm))
    if p2.ro1_purcell_stub_offset_mm is not None:
        setattr(p_ro1, "purcell_stub_offset_mm", float(p2.ro1_purcell_stub_offset_mm))
    if p2.ro2_purcell_stub_offset_mm is not None:
        setattr(p_ro2, "purcell_stub_offset_mm", float(p2.ro2_purcell_stub_offset_mm))
    if p2.ro1_purcell_stub_width_um is not None:
        setattr(p_ro1, "purcell_stub_width", f"{float(p2.ro1_purcell_stub_width_um)}um")
    if p2.ro2_purcell_stub_width_um is not None:
        setattr(p_ro2, "purcell_stub_width", f"{float(p2.ro2_purcell_stub_width_um)}um")
    if p2.ro1_purcell_stub_gap_um is not None:
        setattr(p_ro1, "purcell_stub_gap", f"{float(p2.ro1_purcell_stub_gap_um)}um")
    if p2.ro2_purcell_stub_gap_um is not None:
        setattr(p_ro2, "purcell_stub_gap", f"{float(p2.ro2_purcell_stub_gap_um)}um")

    design.variables.update(base.Dict(cpw_width=p.cpw_width, cpw_gap=p.cpw_gap))
    design.chips.main.size["size_x"] = p.chip_size_x
    design.chips.main.size["size_y"] = p.chip_size_y

    # Best-effort: set tan_delta for dissipative "main".
    try:
        if hasattr(design.chips.main, "material") and isinstance(design.chips.main.material, dict):
            design.chips.main.material["tan_delta"] = float(base.TAN_DELTA_MAIN)
        elif hasattr(design.chips.main, "material"):
            setattr(design.chips.main.material, "tan_delta", float(base.TAN_DELTA_MAIN))
    except Exception:
        pass

    # Determine layout based on relative qubit positions.
    dx = q2_x_mm - q1_x_mm
    dy = q2_y_mm - q1_y_mm

    q1_orientation = 0.0
    q2_orientation = 0.0

    if official_singlepad and abs(dx) >= abs(dy):
        # Follow the official 2Q horizontal mirrored topology:
        # the two qubits share the same logical connection-pad definition, while
        # one qubit is rotated by 180 degrees so bus pads face inward and the
        # readout pads face outward.
        q1_ro_loc_W, q1_ro_loc_H = +1, +1
        q1_bus_loc_W, q1_bus_loc_H = -1, -1
        q2_ro_loc_W, q2_ro_loc_H = +1, +1
        q2_bus_loc_W, q2_bus_loc_H = -1, -1

        if q1_x_mm >= q2_x_mm:
            q1_orientation = 0.0
            q2_orientation = 180.0
        else:
            q1_orientation = 180.0
            q2_orientation = 0.0
    else:
        # loc_W and loc_H must be +1 or -1 (TransmonPocket constraint).
        # To get readout pads perpendicular to the coupling direction:
        #   - Horizontal coupling (X): readout pads on top (H=+1), bus on bottom (H=-1)
        #   - Vertical coupling (Y): readout pads on left (W=-1), bus on right (W=+1)
        # The W component of the readout pad is chosen to avoid the bus side.

        if abs(dx) >= abs(dy):
            # Horizontal coupling (Q1-Q2 connected along X axis)
            # Readout: both on top side, W chosen to stay away from bus side
            # Bus: on bottom side, W points toward the other qubit
            if dx >= 0:
                # Q2 is to the right of Q1
                q1_ro_loc_W, q1_ro_loc_H = -1, +1
                q1_bus_loc_W, q1_bus_loc_H = +1, -1
                q2_ro_loc_W, q2_ro_loc_H = +1, +1
                q2_bus_loc_W, q2_bus_loc_H = -1, -1
            else:
                # Q2 is to the left of Q1
                q1_ro_loc_W, q1_ro_loc_H = +1, +1
                q1_bus_loc_W, q1_bus_loc_H = -1, -1
                q2_ro_loc_W, q2_ro_loc_H = -1, +1
                q2_bus_loc_W, q2_bus_loc_H = +1, -1
        else:
            # Vertical coupling (Q1-Q2 connected along Y axis)
            # Both readouts point RIGHT (+X direction), perpendicular to vertical bus
            # Bus pads face each other along Y axis
            if dy >= 0:
                # Q2 is above Q1
                q1_ro_loc_W, q1_ro_loc_H = +1, -1
                q1_bus_loc_W, q1_bus_loc_H = -1, +1
                q2_ro_loc_W, q2_ro_loc_H = +1, +1
                q2_bus_loc_W, q2_bus_loc_H = -1, -1
            else:
                # Q2 is below Q1
                q1_ro_loc_W, q1_ro_loc_H = +1, +1
                q1_bus_loc_W, q1_bus_loc_H = -1, -1
                q2_ro_loc_W, q2_ro_loc_H = +1, -1
                q2_bus_loc_W, q2_bus_loc_H = -1, +1

    _make_transmon(
        design,
        "Q1",
        x_mm=q1_x_mm,
        y_mm=q1_y_mm,
        p=p_q1,
        lj_var=lj_vars[0],
        cj_var=cj_vars[0],
        orientation=q1_orientation,
        with_bus=True,
        bus_loc_W=q1_bus_loc_W,
        bus_loc_H=q1_bus_loc_H,
        readout_loc_W=q1_ro_loc_W,
        readout_loc_H=q1_ro_loc_H,
        bus_pad_width=(f"{float(p2.q1_bus_pad_w_um)}um" if p2.q1_bus_pad_w_um is not None else None),
        bus_pad_height=(f"{float(p2.q1_bus_pad_h_um)}um" if p2.q1_bus_pad_h_um is not None else None),
        bus_pad_gap=(
            f"{float(p2.q1_bus_pad_gap_um)}um"
            if p2.q1_bus_pad_gap_um is not None
            else (f"{float(p2.bus_pad_gap_um)}um" if p2.bus_pad_gap_um is not None else None)
        ),
    )
    _make_transmon(
        design,
        "Q2",
        x_mm=q2_x_mm,
        y_mm=q2_y_mm,
        p=p_q2,
        lj_var=lj_vars[1],
        cj_var=cj_vars[1],
        orientation=q2_orientation,
        with_bus=True,
        bus_loc_W=q2_bus_loc_W,
        bus_loc_H=q2_bus_loc_H,
        readout_loc_W=q2_ro_loc_W,
        readout_loc_H=q2_ro_loc_H,
        bus_pad_width=(f"{float(p2.q2_bus_pad_w_um)}um" if p2.q2_bus_pad_w_um is not None else None),
        bus_pad_height=(f"{float(p2.q2_bus_pad_h_um)}um" if p2.q2_bus_pad_h_um is not None else None),
        bus_pad_gap=(
            f"{float(p2.q2_bus_pad_gap_um)}um"
            if p2.q2_bus_pad_gap_um is not None
            else (f"{float(p2.bus_pad_gap_um)}um" if p2.bus_pad_gap_um is not None else None)
        ),
    )
    design.rebuild()

    # Place readout chains near each qubit readout pin.
    q1_pin_x, q1_pin_y = _try_get_pin_xy(design, "Q1", "readout")
    q2_pin_x, q2_pin_y = _try_get_pin_xy(design, "Q2", "readout")

    # Offset for tee placement (relative to pin)
    tee_offset = float(p2.tee_offset_mm)  # mm

    # For Q1: readout pad is at (q1_ro_loc_W, q1_ro_loc_H)
    # Place tee in the direction the readout pad faces
    c1x = q1_pin_x + q1_ro_loc_W * tee_offset + float(p2.ro1_dx_mm)
    c1y = q1_pin_y + q1_ro_loc_H * tee_offset + float(p2.ro1_dy_mm)

    # For Q2: similarly
    c2x = q2_pin_x + q2_ro_loc_W * tee_offset + float(p2.ro2_dx_mm)
    c2y = q2_pin_y + q2_ro_loc_H * tee_offset + float(p2.ro2_dy_mm)

    # Determine tee orientations based on readout pad locations
    # Tee second_end direction by orientation: 0=down, 90=right, 180=up, 270=left
    # We want the tee's second_end to point toward the qubit
    def get_tee_orientation(ro_loc_W: int, ro_loc_H: int) -> str:
        # Tee second_end should point back toward qubit
        if abs(ro_loc_H) >= abs(ro_loc_W) and ro_loc_H != 0:
            return "0" if ro_loc_H > 0 else "180"
        return "270" if ro_loc_W > 0 else "90"

    tee1_ori = get_tee_orientation(q1_ro_loc_W, q1_ro_loc_H)
    tee2_ori = get_tee_orientation(q2_ro_loc_W, q2_ro_loc_H)

    lp1_direction, lp1_dx = _pick_launchpad_params(
        coupler_x_mm=c1x,
        coupler_y_mm=c1y,
        tee_orientation=tee1_ori,
        p=p_ro1,
    )
    lp2_direction, lp2_dx = _pick_launchpad_params(
        coupler_x_mm=c2x,
        coupler_y_mm=c2y,
        tee_orientation=tee2_ori,
        p=p_ro2,
    )

    _add_readout_chain(
        design,
        suffix="1",
        qname="Q1",
        q_pin="readout",
        coupler_x_mm=c1x,
        coupler_y_mm=c1y,
        p=p_ro1,
        tee_orientation=tee1_ori,
        lp_direction=lp1_direction,
        swap_tee_ports=bool(p2.swap_tee_ports),
        lp_dx_mm=float(p2.lp1_dx_mm) if p2.lp1_dx_mm is not None else lp1_dx,
    )
    _add_readout_chain(
        design,
        suffix="2",
        qname="Q2",
        q_pin="readout",
        coupler_x_mm=c2x,
        coupler_y_mm=c2y,
        p=p_ro2,
        tee_orientation=tee2_ori,
        lp_direction=lp2_direction,
        swap_tee_ports=bool(p2.swap_tee_ports),
        lp_dx_mm=float(p2.lp2_dx_mm) if p2.lp2_dx_mm is not None else lp2_dx,
    )

    # Bus coupling between Q1 and Q2
    if official_singlepad and abs(dx) >= abs(dy):
        RoutePathfinder = _load_route_pathfinder_cls()
        RoutePathfinder(
            design,
            "Bus_12",
            options=base.Dict(
                fillet=p2.bus_fillet,
                trace_width=p2.bus_width,
                trace_gap=p2.bus_gap,
                lead=base.Dict(start_straight=p2.bus_lead_start, end_straight=p2.bus_lead_end),
                pin_inputs=base.Dict(
                    start_pin=base.Dict(component="Q1", pin="bus"),
                    end_pin=base.Dict(component="Q2", pin="bus"),
                ),
            ),
            type="CPW",
        )
    else:
        base.RouteMeander(
            design,
            "Bus_12",
            options=base.Dict(
                total_length=p2.bus_total_length,
                fillet=p2.bus_fillet,
                trace_width=p2.bus_width,
                trace_gap=p2.bus_gap,
                meander=base.Dict(spacing=p2.bus_spacing),
                lead=base.Dict(start_straight=p2.bus_lead_start, end_straight=p2.bus_lead_end),
                pin_inputs=base.Dict(
                    start_pin=base.Dict(component="Q1", pin="bus"),
                    end_pin=base.Dict(component="Q2", pin="bus"),
                ),
            ),
            type="CPW",
        )

    design.rebuild()
    if _env_bool("METAL_2Q_Q3D_CONNECTOR_PAD_CUTOUTS", True):
        _add_connector_pad_subtract_cutouts(design)


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


def build_chip_summary_multiq(*, sample_id: str) -> TDict[str, Any]:
    return {
        "meta": {
            "created_utc": base.now_utc_iso(),
            "updated_utc": None,
            "sample_id": sample_id,
            "units": {"f": "GHz", "kappa": "Hz", "C": "fF", "K": "MHz", "T": "us"},
        },
        "resonators": {"readout1": _empty_readout_block(), "readout2": _empty_readout_block()},
        "qubits": {"Q1": _empty_qubit_block(), "Q2": _empty_qubit_block()},
        "q3d": {"internal": {}, "external": {}},
        "chip": {
            "T1_qubit1_est_us": None,
            "T2_qubit1_est_us": None,
            "T1_qubit2_est_us": None,
            "T2_qubit2_est_us": None,
            "chi12_MHz": None,
            "chi1_over_kappa1": None,
            "chi2_over_kappa2": None,
        },
        "status": "init",
    }


def _estimate_T1T2_2q(payload: TDict[str, Any]) -> None:
    for qname, roname in [("Q1", "readout1"), ("Q2", "readout2")]:
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

            if qname == "Q1":
                payload["chip"]["T1_qubit1_est_us"] = q["T1_est_us"]
                payload["chip"]["T2_qubit1_est_us"] = q["T2_est_us"]
            else:
                payload["chip"]["T1_qubit2_est_us"] = q["T1_est_us"]
                payload["chip"]["T2_qubit2_est_us"] = q["T2_est_us"]


def _purcell_filter_meta_for_readout(layout_params: Optional[Layout2QParams], idx: int) -> TDict[str, Any]:
    if layout_params is None:
        return {}
    if idx == 1:
        length_mm = float(layout_params.ro1_purcell_stub_length_mm or 0.0)
        offset_mm = float(layout_params.ro1_purcell_stub_offset_mm or 0.0)
        width_um = float(layout_params.ro1_purcell_stub_width_um or 0.0)
        gap_um = float(layout_params.ro1_purcell_stub_gap_um or 0.0)
    else:
        length_mm = float(layout_params.ro2_purcell_stub_length_mm or 0.0)
        offset_mm = float(layout_params.ro2_purcell_stub_offset_mm or 0.0)
        width_um = float(layout_params.ro2_purcell_stub_width_um or 0.0)
        gap_um = float(layout_params.ro2_purcell_stub_gap_um or 0.0)
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


def _postprocess_one_readout(ro: TDict[str, Any]) -> None:
    if ro.get("kappa_i_over_2pi_Hz") is None and ro.get("Qi") and ro.get("f_GHz"):
        ro["kappa_i_over_2pi_Hz"] = float(ro["f_GHz"]) * 1e9 / float(ro["Qi"])

    ki = ro.get("kappa_i_over_2pi_Hz")
    ke = (ro.get("external") or {}).get("kappa_over_2pi_Hz")

    if ki is not None:
        ro["kappa_internal_over_2pi_Hz"] = float(ki)
    if ke is not None:
        ro["kappa_external_over_2pi_Hz"] = float(ke)

    if ki is not None and ke is not None:
        kappa_total = float(ki) + float(ke)
        ro["kappa_total_over_2pi_Hz"] = kappa_total
        prev = ro.get("kappa_over_2pi_Hz")
        if prev is not None:
            try:
                prev_f = float(prev)
            except Exception:
                prev_f = None
            if prev_f is not None and math.isfinite(prev_f) and abs(prev_f - kappa_total) > 0:
                ro.setdefault("kappa_over_2pi_Hz_prev", prev_f)
                warnings = ro.get("warnings")
                if not isinstance(warnings, list):
                    warnings = []
                    ro["warnings"] = warnings
                warnings.append(
                    "kappa_over_2pi_Hz overwritten: total=kappa_i_over_2pi_Hz+kappa_e_over_2pi_Hz"
                )
        ro["kappa_over_2pi_Hz"] = kappa_total

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


def _safe_get_convergence_f(eig) -> Optional[pd.DataFrame]:
    candidates = []
    try:
        candidates.append(getattr(eig.sim, "convergence_f", None))
    except Exception:
        pass
    try:
        candidates.append(getattr(eig, "convergence_f", None))
    except Exception:
        pass
    for obj in candidates:
        if isinstance(obj, pd.DataFrame) and len(obj) >= 2:
            return obj
    return None


def _last_pass_delta_ghz(conv_df: Optional[pd.DataFrame], mode_idx: int) -> Optional[float]:
    try:
        if conv_df is None or len(conv_df) < 2:
            return None
        df = conv_df.copy()
        numeric_cols = []
        for c in df.columns:
            c_label = str(c).strip().lower()
            if "pass" in c_label:
                continue
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().sum() >= 2:
                numeric_cols.append(c)
        if not numeric_cols:
            return None
        idx = int(mode_idx)
        if idx < 0:
            return None
        col = numeric_cols[idx] if idx < len(numeric_cols) else numeric_cols[-1]
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(s) < 2:
            return None
        return abs(float(s.iloc[-1]) - float(s.iloc[-2]))
    except Exception:
        return None


def _load_kappa_in_fn() -> Optional[Callable[[float, float, float], float]]:
    try:
        module = importlib.import_module("qiskit_metal.analyses.em.kappa_calculation")
        fn = getattr(module, "kappa_in", None)
        if callable(fn):
            return cast(Callable[[float, float, float], float], fn)
    except Exception:
        pass
    return None


def _effective_cin_for_target_kext(fr_ghz: Optional[float], target_kext_hz: float) -> Optional[float]:
    try:
        fr = float(fr_ghz) if fr_ghz is not None else None
        target = float(target_kext_hz)
    except Exception:
        return None
    if fr is None or not np.isfinite(fr) or fr <= 0 or not np.isfinite(target) or target <= 0:
        return None

    kappa_in_fn = _load_kappa_in_fn()
    if kappa_in_fn is None:
        return None

    ref_cin_fF = 100.0
    fr_hz = fr * 1e9
    try:
        ref_kappa_hz = float(kappa_in_fn(fr_hz, ref_cin_fF * 1e-15, fr_hz))
    except Exception:
        return None
    ref_kappa_over_2pi = ref_kappa_hz / (2.0 * math.pi) if ref_kappa_hz > 0 else None
    if ref_kappa_over_2pi is None or not np.isfinite(ref_kappa_over_2pi) or ref_kappa_over_2pi <= 0:
        return None
    return ref_cin_fF * math.sqrt(target / ref_kappa_over_2pi)


def _apply_effective_cin_override(payload: TDict[str, Any], pair_idx: int, cin_fF: float) -> bool:
    try:
        cin = float(cin_fF)
    except Exception:
        return False
    if not np.isfinite(cin) or cin <= 0:
        return False

    kappa_in_fn = _load_kappa_in_fn()
    if kappa_in_fn is None:
        return False

    ro = ((payload.get("resonators") or {}).get(f"readout{pair_idx}") or {})
    try:
        fr_ghz = float(ro.get("f_GHz"))
    except Exception:
        return False
    if not np.isfinite(fr_ghz) or fr_ghz <= 0:
        return False

    ext = ro.get("external")
    if not isinstance(ext, dict):
        ext = {}
        ro["external"] = ext

    prev = {
        "Cin_fF": ext.get("Cin_fF"),
        "pair": ext.get("pair"),
        "method": ext.get("method"),
        "kappa_over_2pi_Hz": ext.get("kappa_over_2pi_Hz"),
    }
    fr_hz = fr_ghz * 1e9
    try:
        kappa_e_hz = float(kappa_in_fn(fr_hz, cin * 1e-15, fr_hz))
    except Exception:
        return False

    ext.update(
        {
            "Cin_fF": cin,
            "pair": f"effective_Cin_Q{pair_idx}",
            "res_node": f"readout{pair_idx}_resonator_effective",
            "feed_node": f"readout{pair_idx}_feed_effective",
            "method": "effective_cin_override_no_q3d",
            "previous_q3d_pick": prev,
            "kappa_Hz": kappa_e_hz,
            "kappa_over_2pi_Hz": kappa_e_hz / (2.0 * math.pi) if kappa_e_hz > 0 else None,
            "Qe": fr_hz / kappa_e_hz if kappa_e_hz > 0 else None,
        }
    )

    warnings = ro.get("warnings")
    if not isinstance(warnings, list):
        warnings = []
        ro["warnings"] = warnings
    warnings.append(f"effective Cin override applied: Cin={cin:.6g} fF")
    return True


def _apply_external_coupling_overrides(
    payload: TDict[str, Any],
    *,
    q1_effective_cin_fF: float = 0.0,
    q2_effective_cin_fF: float = 0.0,
    q1_target_kext_hz: float = 0.0,
    q2_target_kext_hz: float = 0.0,
) -> bool:
    changed = False
    for idx, target_kext_hz, cin_fF in (
        (1, q1_target_kext_hz, q1_effective_cin_fF),
        (2, q2_target_kext_hz, q2_effective_cin_fF),
    ):
        ro = ((payload.get("resonators") or {}).get(f"readout{idx}") or {})
        fr_ghz = ro.get("f_GHz")
        cin_override = _effective_cin_for_target_kext(fr_ghz, float(target_kext_hz)) if float(target_kext_hz) > 0 else None
        if cin_override is None:
            try:
                cin_override = float(cin_fF)
            except Exception:
                cin_override = None
        if cin_override is not None and np.isfinite(cin_override) and cin_override > 0:
            changed = _apply_effective_cin_override(payload, idx, float(cin_override)) or changed
    return changed


def postprocess_2q(payload: TDict[str, Any]) -> TDict[str, Any]:
    payload["meta"]["updated_utc"] = base.now_utc_iso()

    ro1 = payload["resonators"]["readout1"]
    ro2 = payload["resonators"]["readout2"]
    q1 = payload["qubits"]["Q1"]
    q2 = payload["qubits"]["Q2"]

    _postprocess_one_readout(ro1)
    _postprocess_one_readout(ro2)
    _postprocess_one_qubit(q1)
    _postprocess_one_qubit(q2)

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

    payload["chip"]["chi1_over_kappa1"] = _chi_over_kappa(q1, ro1)
    payload["chip"]["chi2_over_kappa2"] = _chi_over_kappa(q2, ro2)

    _estimate_T1T2_2q(payload)
    return payload


def _mode_picker_invalid_reasons(payload: TDict[str, Any]) -> List[str]:
    meta = payload.get("meta") or {}
    mode_picker = meta.get("mode_picker")
    if not isinstance(mode_picker, dict):
        return []
    reasons = [str(x) for x in (mode_picker.get("invalid_reasons") or []) if str(x)]
    if mode_picker.get("valid") is False:
        return reasons or ["mode_picker.valid=false"]
    warnings = [str(x) for x in (mode_picker.get("warnings") or []) if str(x)]
    for marker in ("Invalid mode selection", "Low junction participation"):
        matched = [item for item in warnings if marker in item]
        if matched:
            return reasons or matched
    return []


# -------------------------
# mode picking
# -------------------------


def _qi_from_p_tandelta(p: Optional[float], tan_delta: float) -> Optional[float]:
    try:
        if p is None:
            return None
        p = float(p)
        tan_delta = float(tan_delta)
        if not np.isfinite(p) or not np.isfinite(tan_delta) or p <= 0 or tan_delta <= 0:
            return None
        return 1.0 / (p * tan_delta)
    except Exception:
        return None


def _extract_qdielectric_main(res: TDict[str, Any], idx_mode: int) -> Optional[float]:
    direct = res.get("Qdielectric_main")
    if isinstance(direct, pd.DataFrame):
        try:
            v = direct.iloc[int(idx_mode), 0]
            return float(v) if np.isfinite(v) and float(v) > 0 else None
        except Exception:
            pass
    if isinstance(direct, pd.Series):
        try:
            v = direct.iloc[int(idx_mode)]
            return float(v) if np.isfinite(v) and float(v) > 0 else None
        except Exception:
            pass

    qd = res.get("Qdielectric")
    if isinstance(qd, dict):
        for k in ("main", "dielectrics_bulk.main", "dielectrics_bulk:main"):
            v = qd.get(k)
            if v is None:
                continue
            try:
                fv = float(v)
                if np.isfinite(fv) and fv > 0:
                    return fv
            except Exception:
                continue

    for k, v in res.items():
        ks = str(k).lower()
        if "qdielectric" not in ks or "main" not in ks:
            continue
        if isinstance(v, pd.DataFrame):
            try:
                fv = float(v.iloc[int(idx_mode), 0])
                if np.isfinite(fv) and fv > 0:
                    return fv
            except Exception:
                continue
        if isinstance(v, pd.Series):
            try:
                fv = float(v.iloc[int(idx_mode)])
                if np.isfinite(fv) and fv > 0:
                    return fv
            except Exception:
                continue

    return None


def _extract_participation_value(
    res: TDict[str, Any],
    idx_mode: int,
    *,
    kind: str,
    name: str,
) -> Optional[float]:
    container = res.get("p")
    if isinstance(container, dict):
        kind_obj = container.get(kind)
        if isinstance(kind_obj, dict):
            obj = kind_obj.get(name)
            if isinstance(obj, pd.DataFrame):
                try:
                    v = obj.iloc[int(idx_mode), 0]
                    return float(v) if np.isfinite(v) else None
                except Exception:
                    return None
            if isinstance(obj, pd.Series):
                try:
                    v = obj.iloc[int(idx_mode)]
                    return float(v) if np.isfinite(v) else None
                except Exception:
                    return None
            if obj is None:
                return None
            try:
                v = float(obj)
                return float(v) if np.isfinite(v) else None
            except Exception:
                return None

        if isinstance(kind_obj, pd.DataFrame):
            try:
                if str(name) in kind_obj.columns:
                    v = kind_obj.loc[:, str(name)].iloc[int(idx_mode)]
                else:
                    v = kind_obj.iloc[int(idx_mode), 0]
                return float(v) if np.isfinite(v) else None
            except Exception:
                return None

    for k, v in res.items():
        ks = str(k).lower()
        if kind.lower() not in ks or str(name).lower() not in ks:
            continue
        if isinstance(v, pd.DataFrame):
            try:
                fv = float(v.iloc[int(idx_mode), 0])
                return fv if np.isfinite(fv) else None
            except Exception:
                continue
        if isinstance(v, pd.Series):
            try:
                fv = float(v.iloc[int(idx_mode)])
                return fv if np.isfinite(fv) else None
            except Exception:
                continue
        try:
            fv = float(v)
            return fv if np.isfinite(fv) else None
        except Exception:
            continue

    return None


def _extract_pm_normed(res: TDict[str, Any]) -> Optional[pd.DataFrame]:
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
            ljs = res.get("Ljs")
            if isinstance(ljs, pd.Series) and len(ljs.index) == int(arr.shape[1]):
                cols = [str(x) for x in list(ljs.index)]
            if cols is None:
                cjs = res.get("Cjs")
                if isinstance(cjs, pd.Series) and len(cjs.index) == int(arr.shape[1]):
                    cols = [str(x) for x in list(cjs.index)]
            if cols is None:
                cols = [str(i) for i in range(int(arr.shape[1]))]
            return pd.DataFrame(arr, columns=pd.Index(cols))
    for k in ("Pm_norm", "Pm", "pm_normed"):
        v = res.get(k)
        if isinstance(v, pd.DataFrame):
            return v
    return None


def _find_pm_column(pm: pd.DataFrame, preferred: str) -> Optional[str]:
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


def _pm_cell_value(pm: pd.DataFrame, mode_index: int, col: str) -> Optional[float]:
    try:
        value = float(pm.loc[:, col].iloc[int(mode_index)])
    except Exception:
        return None
    if not np.isfinite(value):
        return None
    return float(value)


def _pick_two_qubits_and_two_resonators(
    chi_MHz: NDArray[np.floating[Any]],
    *,
    freqs_GHz: Optional[NDArray[np.floating[Any]]] = None,
    res: Optional[TDict[str, Any]] = None,
) -> Tuple[int, int, int, int, TDict[str, Any]]:
    """Heuristic mode picker.

    Returns: (idx_q1, idx_q2, idx_r1, idx_r2, debug_info).
    """
    chi = np.array(chi_MHz, dtype=float)
    n = chi.shape[0]
    diag = np.abs(np.diag(chi))
    if n < 4:
        raise ValueError(f"Need >=4 modes for 2Q+2R picking, got n={n}")

    def _env_float(name: str) -> Optional[float]:
        raw = os.environ.get(name, "").strip()
        if not raw:
            return None
        try:
            return float(raw)
        except Exception:
            return None

    # Optional frequency windows (parameterized via environment variables).
    q_min = _env_float("METAL_2Q_PICK_Q_MIN_GHZ")
    q_max = _env_float("METAL_2Q_PICK_Q_MAX_GHZ")
    r_min = _env_float("METAL_2Q_PICK_R_MIN_GHZ")
    r_max = _env_float("METAL_2Q_PICK_R_MAX_GHZ")
    kerr_floor = _env_float("METAL_2Q_PICK_KERR_FLOOR_MHZ")
    q_kerr_min = _env_float("METAL_2Q_PICK_Q_KERR_MIN_MHZ")
    q_max_dist = _env_float("METAL_2Q_PICK_Q_MAX_DIST_GHZ")
    r_kerr_max = _env_float("METAL_2Q_PICK_R_KERR_MAX_MHZ")
    r_pm_max = _env_float("METAL_2Q_PICK_R_PM_MAX")

    pick_use_pm_raw = os.environ.get("METAL_2Q_PICK_USE_PM", "1").strip().lower()
    pick_use_pm = pick_use_pm_raw not in {"0", "false", "no", "off"}

    pm_min = _env_float("METAL_2Q_PICK_PM_MIN")
    if pm_min is None or not np.isfinite(pm_min) or float(pm_min) < 0:
        pm_min = 0.02

    valid_pm_min = _env_float("METAL_2Q_PICK_VALID_PM_MIN")
    if valid_pm_min is None or not np.isfinite(valid_pm_min) or float(valid_pm_min) < 0:
        valid_pm_min = max(float(pm_min), 0.30)
    else:
        valid_pm_min = max(float(valid_pm_min), float(pm_min), 0.30)

    strict_pm_raw = os.environ.get("METAL_2Q_PICK_STRICT_PM", "1").strip().lower()
    strict_pm = strict_pm_raw in {"1", "true", "yes", "on"}
    if strict_pm and not pick_use_pm:
        raise ValueError("METAL_2Q_PICK_STRICT_PM requires METAL_2Q_PICK_USE_PM")
    if strict_pm:
        if res is None:
            raise ValueError("Strict PM picking requires EPR results (res)")
        if freqs_GHz is None:
            raise ValueError("Strict PM picking requires freqs_GHz")

    def _in_window(i: int, lo: Optional[float], hi: Optional[float]) -> bool:
        if freqs_GHz is None:
            return True
        try:
            f = float(freqs_GHz[int(i)])
        except Exception:
            return True
        if lo is not None and f < float(lo):
            return False
        if hi is not None and f > float(hi):
            return False
        return True

    def _dist_to_window(f: float, lo: Optional[float], hi: Optional[float]) -> float:
        if lo is not None and f < float(lo):
            return float(lo) - float(f)
        if hi is not None and f > float(hi):
            return float(f) - float(hi)
        return 0.0

    strict_windows_raw = os.environ.get("METAL_2Q_PICK_STRICT_WINDOWS", "1").strip().lower()
    strict_windows = strict_windows_raw not in {"0", "false", "no", "off"}

    def _pick_candidates_with_expansion(
        indices: List[int],
        *,
        lo: Optional[float],
        hi: Optional[float],
        needed: int,
        label: str,
        warnings: List[str],
    ) -> Tuple[List[int], TDict[str, Any]]:
        """Pick candidates within a window; optionally expand if relaxed mode is enabled."""
        info: TDict[str, Any] = {
            "window_GHz": [float(lo) if lo is not None else None, float(hi) if hi is not None else None],
            "window_used_GHz": [float(lo) if lo is not None else None, float(hi) if hi is not None else None],
            "expand_GHz": 0.0,
            "candidates": [],
            "strict_windows": bool(strict_windows),
        }

        # No frequency information or no window constraint.
        if freqs_GHz is None or (lo is None and hi is None):
            info["candidates"] = [int(i) for i in indices]
            return list(indices), info

        base = [i for i in indices if _in_window(i, lo, hi)]
        if len(base) >= needed:
            info["candidates"] = [int(i) for i in base]
            return base, info

        if strict_windows:
            warnings.append(
                f"{label} window [{lo},{hi}] has only {len(base)}/{needed} candidate(s); strict windowing rejects out-of-window fallback"
            )
            info["candidates"] = [int(i) for i in base]
            return base, info

        expansions = [0.25, 0.5, 1.0, 2.0]
        if str(label).strip().lower().startswith("qubit"):
            expansions = [0.25, 0.5, 1.0]
        best = list(base)
        best_lo, best_hi = lo, hi
        best_ex = 0.0
        for ex in expansions:
            lo_ex = (float(lo) - float(ex)) if lo is not None else None
            hi_ex = (float(hi) + float(ex)) if hi is not None else None
            cand = [i for i in indices if _in_window(i, lo_ex, hi_ex)]
            if len(cand) > len(best):
                best = list(cand)
                best_lo, best_hi, best_ex = lo_ex, hi_ex, float(ex)
            if len(cand) >= needed:
                warnings.append(
                    f"{label} window expanded by ±{float(ex):g} GHz: [{lo_ex},{hi_ex}] (base [{lo},{hi}] had {len(base)}/{needed} candidates)"
                )
                info["window_used_GHz"] = [lo_ex, hi_ex]
                info["expand_GHz"] = float(ex)
                info["candidates"] = [int(i) for i in cand]
                return cand, info

        # Still insufficient: keep closest-to-window modes to avoid far-away picks.
        scored: List[Tuple[float, float, int]] = []
        for i in indices:
            try:
                f = float(freqs_GHz[int(i)])
            except Exception:
                f = float("nan")
            d = _dist_to_window(f, lo, hi) if np.isfinite(f) else 0.0
            low_pen = 1.0 if (lo is not None and np.isfinite(f) and float(f) < float(lo)) else 0.0
            scored.append((float(d), float(low_pen), int(i)))
        scored.sort(key=lambda t: (t[0], t[1]))
        take = min(len(scored), max(int(needed), 4))
        keep = [int(i) for _, _, i in scored[:take]]
        if keep and best_ex > 0.0:
            warnings.append(
                f"{label} window expansion to [{best_lo},{best_hi}] still had {len(best)}/{needed} candidates; using closest {len(keep)} modes to base window [{lo},{hi}]"
            )
        info["window_used_GHz"] = [best_lo, best_hi]
        info["expand_GHz"] = float(best_ex)
        info["candidates"] = [int(i) for i in keep]
        return keep, info

    warnings: List[str] = []
    all_idx = [int(i) for i in range(n)]

    q_kerr_min_val = 1.0
    try:
        if q_kerr_min is not None and np.isfinite(q_kerr_min) and float(q_kerr_min) > 0:
            q_kerr_min_val = float(q_kerr_min)
    except Exception:
        q_kerr_min_val = 1.0

    q_like = [i for i in all_idx if float(diag[int(i)]) >= float(q_kerr_min_val)]
    q_space = q_like if len(q_like) >= 2 else all_idx
    if len(q_like) < 2:
        warnings.append(
            f"Only {len(q_like)} qubit-like mode(s) with |self-Kerr|>={q_kerr_min_val:g} MHz; relaxing Kerr filter"
        )
    if strict_pm:
        q_pool = [int(i) for i in all_idx if _in_window(int(i), q_min, q_max)] if strict_windows else [int(i) for i in all_idx]
        q_info = {
            "window_GHz": [q_min, q_max],
            "window_used_GHz": [q_min, q_max],
            "expand_GHz": 0.0,
            "candidates": [int(i) for i in q_pool],
            "strict_windows": bool(strict_windows),
        }
        if len(q_like) >= 2:
            warnings.append(
                f"Strict PM picking: skipping Kerr prefilter (had {len(q_like)}/{n} qubit-like by Kerr>= {q_kerr_min_val:g} MHz)"
            )
        if strict_windows and len(q_pool) < 2:
            raise ValueError(f"Strict PM picking found only {len(q_pool)} in-window qubit candidate(s) for [{q_min},{q_max}] GHz")
    else:
        q_pool, q_info = _pick_candidates_with_expansion(
            q_space,
            lo=q_min,
            hi=q_max,
            needed=2,
            label="Qubit",
            warnings=warnings,
        )
        if len(q_pool) < 2:
            if strict_windows:
                raise ValueError(f"Qubit window [{q_min},{q_max}] produced only {len(q_pool)} candidate(s) under strict windowing")
            warnings.append(
                f"Qubit window [{q_min},{q_max}] produced <2 candidates; falling back to largest |self-Kerr| over all modes"
            )
            q_pool = [int(i) for i in np.argsort(-diag)]

    focus_pair_raw = os.environ.get("METAL_2Q_FOCUS_PAIR", "").strip()
    try:
        focus_pair = int(focus_pair_raw) if focus_pair_raw else 0
    except Exception:
        focus_pair = 0

    focus_require_in_window_raw = os.environ.get("METAL_2Q_PICK_FOCUS_REQUIRE_IN_WINDOW", "1").strip().lower()
    focus_require_in_window = focus_require_in_window_raw in {"1", "true", "yes", "y", "on"}

    q_max_dist_val_used: Optional[float] = None

    if freqs_GHz is not None and (q_min is not None or q_max is not None):
        q_max_dist_val = 1.0
        try:
            if q_max_dist is not None and np.isfinite(q_max_dist) and float(q_max_dist) > 0:
                q_max_dist_val = float(q_max_dist)
        except Exception:
            q_max_dist_val = 1.0
        q_max_dist_val_used = float(q_max_dist_val)

        q_pool_scored: List[Tuple[float, int]] = []
        for i in [int(x) for x in q_pool]:
            try:
                f = float(freqs_GHz[int(i)])
            except Exception:
                continue
            if not np.isfinite(f):
                continue
            d = float(_dist_to_window(f, q_min, q_max))
            q_pool_scored.append((d, int(i)))

        q_pool_scored.sort(key=lambda t: t[0])
        q_pool_near = [int(i) for d, i in q_pool_scored if float(d) <= float(q_max_dist_val)]
        if len(q_pool_near) >= 2:
            if len(q_pool_near) < len(q_pool):
                warnings.append(
                    f"Filtered qubit candidates by window distance <= {q_max_dist_val:g} GHz (kept {len(q_pool_near)}/{len(q_pool)})"
                )
            q_pool = list(q_pool_near)
            q_info["max_dist_GHz"] = float(q_max_dist_val)
            q_info["candidates"] = [int(i) for i in q_pool]
        elif focus_pair in {1, 2} and not strict_windows:
            keep = [int(i) for _, i in q_pool_scored[:2]]
            kept_in_band = int(len(q_pool_near))
            warnings.append(
                f"Focused mode picking (pair={focus_pair}): only {kept_in_band} qubit candidate(s) within {q_max_dist_val:g} GHz of window [{q_min},{q_max}]; keeping closest 2 candidates"
            )
            q_pool = list(keep)
            q_info["max_dist_GHz"] = float(q_max_dist_val)
            q_info["candidates"] = [int(i) for i in q_pool]
        else:
            raise ValueError(
                f"Failed to find >=2 qubit candidates within {q_max_dist_val:g} GHz of window [{q_min},{q_max}]"
            )

    def score_res(qi: int, ri: int) -> float:
        # Prefer strong coupling and small self-Kerr for resonator-ish modes.
        den = abs(float(chi[ri, ri]))
        if kerr_floor is not None and np.isfinite(kerr_floor) and kerr_floor > 0:
            den = max(float(den), float(kerr_floor))
        return abs(float(chi[qi, ri])) / (1e-9 + float(den))

    def _res_pool_for(qubits: Tuple[int, int]) -> List[int]:
        q_a, q_b = int(qubits[0]), int(qubits[1])
        remaining_local = [int(i) for i in range(n) if int(i) not in (q_a, q_b)]
        if freqs_GHz is None or (r_min is None and r_max is None):
            return list(remaining_local)
        in_win = [int(i) for i in remaining_local if _in_window(int(i), r_min, r_max)]
        return in_win if len(in_win) >= 2 else list(remaining_local)

    best_q: Optional[Tuple[int, int, float, float]] = None
    q_candidates = [int(i) for i in q_pool]
    for i_a in range(len(q_candidates)):
        for i_b in range(i_a + 1, len(q_candidates)):
            qa, qb = int(q_candidates[i_a]), int(q_candidates[i_b])
            r_pool_local = _res_pool_for((qa, qb))
            if len(r_pool_local) < 2:
                continue
            best_s = None
            for r1 in r_pool_local:
                for r2 in r_pool_local:
                    if int(r2) == int(r1):
                        continue
                    s = float(score_res(qa, int(r1)) + score_res(qb, int(r2)))
                    if best_s is None or s > float(best_s):
                        best_s = float(s)
            if best_s is None:
                continue
            kerr_sum = float(diag[int(qa)] + diag[int(qb)])
            if best_q is None or (float(best_s), kerr_sum) > (float(best_q[2]), float(best_q[3])):
                best_q = (int(qa), int(qb), float(best_s), float(kerr_sum))

    if best_q is not None:
        idx_q1, idx_q2 = int(best_q[0]), int(best_q[1])
    else:
        q_sorted = sorted(q_pool, key=lambda i: diag[int(i)], reverse=True)
        idx_q1, idx_q2 = int(q_sorted[0]), int(q_sorted[1])

    picked_by_pm = False
    pm_debug: TDict[str, Any] = {}
    if pick_use_pm and res is not None and freqs_GHz is not None:
        try:
            pm = _extract_pm_normed(res)
        except Exception:
            pm = None

        pm_df = pm
        if pm_df is not None and len(pm_df.columns) >= 2 and len(pm_df.index) >= n:
            pm_normed = pm_df

            def _pm_val(i: int, col: str, *, _pm: pd.DataFrame = pm_normed) -> float:
                v = _pm_cell_value(_pm, int(i), str(col))
                return float(v) if v is not None else 0.0

            q_candidates = [int(i) for i in q_pool] if q_pool else [int(i) for i in range(n)]
            jj1_pref = os.environ.get("METAL_2Q_PICK_JJ1", "jj1")
            jj2_pref = os.environ.get("METAL_2Q_PICK_JJ2", "jj2")
            c1 = _find_pm_column(pm_normed, jj1_pref)
            c2 = _find_pm_column(pm_normed, jj2_pref)

            def _auto_pick_two_cols(*, _pm: pd.DataFrame = pm_normed) -> List[str]:
                cols_all = [str(c) for c in list(_pm.columns)]
                if len(cols_all) == 2:
                    return [cols_all[0], cols_all[1]]

                fallback_pairs = [("Lj1", "Lj2"), ("Cj1", "Cj2"), ("q1", "q2"), ("jj1", "jj2")]
                for a, b in fallback_pairs:
                    ca = _find_pm_column(_pm, a)
                    cb = _find_pm_column(_pm, b)
                    if ca is not None and cb is not None and ca != cb:
                        return [str(ca), str(cb)]

                peaks: List[Tuple[float, str]] = []
                for col in cols_all:
                    peak = 0.0
                    for i in q_candidates:
                        peak = max(float(peak), float(_pm_val(int(i), str(col))))
                    peaks.append((float(peak), str(col)))
                peaks.sort(key=lambda t: t[0], reverse=True)
                chosen = [c for _, c in peaks[:2]]
                if len(chosen) == 2 and chosen[0] != chosen[1]:
                    return [str(chosen[0]), str(chosen[1])]
                return [str(cols_all[0]), str(cols_all[1])]

            used_cols: List[str]
            if c1 is not None and c2 is not None and c1 != c2:
                used_cols = [str(c1), str(c2)]
            else:
                used_cols = _auto_pick_two_cols()
                warnings.append(
                    f"Pm_normed columns did not match preferred ({jj1_pref},{jj2_pref}); using columns {used_cols}"
                )

            if used_cols and len(used_cols) >= 2 and used_cols[0] != used_cols[1]:
                col_a, col_b = str(used_cols[0]), str(used_cols[1])
                pm_debug["pm_cols"] = [col_a, col_b]
                top_by_col: TDict[str, List[TDict[str, Any]]] = {}
                for col in (col_a, col_b):
                    ranked: List[Tuple[float, int]] = []
                    for cand in range(n):
                        ranked.append((float(_pm_val(int(cand), str(col))), int(cand)))
                    ranked.sort(key=lambda item: item[0], reverse=True)
                    top_by_col[str(col)] = [
                        {
                            "idx": int(idx),
                            "p": float(p_val),
                            "f_GHz": float(freqs_GHz[int(idx)]) if freqs_GHz is not None else None,
                            "self_kerr_MHz": float(chi[int(idx), int(idx)]),
                        }
                        for p_val, idx in ranked[:5]
                    ]
                pm_debug["top_pm_by_col"] = top_by_col

                focus_in_window_active = False
                if focus_pair in {1, 2} and focus_require_in_window and (q_min is not None or q_max is not None):
                    focus_col = str(col_a) if int(focus_pair) == 1 else str(col_b)
                    for cand in q_candidates:
                        try:
                            f = float(freqs_GHz[int(cand)])
                        except Exception:
                            continue
                        if not np.isfinite(f):
                            continue
                        if float(_dist_to_window(float(f), q_min, q_max)) > 0.0:
                            continue
                        p_focus = float(_pm_val(int(cand), focus_col))
                        if float(p_focus) >= float(pm_min):
                            focus_in_window_active = True
                            break
                    if not focus_in_window_active:
                        warnings.append(
                            f"Focused mode picking (pair={focus_pair}): METAL_2Q_PICK_FOCUS_REQUIRE_IN_WINDOW requested but no in-window candidate met pm_min={pm_min:g}; ignoring"
                        )
                pm_debug["focus_require_in_window_active"] = bool(focus_in_window_active)

                best_pair: Optional[Tuple[float, int, int]] = None
                for i in q_candidates:
                    for j in q_candidates:
                        if int(i) == int(j):
                            continue
                        p_ai = float(_pm_val(int(i), col_a))
                        p_bi = float(_pm_val(int(i), col_b))
                        p_aj = float(_pm_val(int(j), col_a))
                        p_bj = float(_pm_val(int(j), col_b))

                        if focus_in_window_active and (q_min is not None or q_max is not None):
                            focus_mode = int(i) if int(focus_pair) == 1 else int(j)
                            try:
                                f_focus = float(freqs_GHz[int(focus_mode)])
                            except Exception:
                                continue
                            if (not np.isfinite(f_focus)) or (float(_dist_to_window(float(f_focus), q_min, q_max)) > 0.0):
                                continue
                            if int(focus_pair) == 1 and float(p_ai) < float(pm_min):
                                continue
                            if int(focus_pair) == 2 and float(p_bj) < float(pm_min):
                                continue

                        if strict_pm and (p_ai < float(pm_min) or p_bj < float(pm_min)):
                            continue

                        score = float(p_ai + p_bj)
                        score -= 0.75 * float(p_bi + p_aj)
                        score += 0.02 * float(diag[int(i)] + diag[int(j)])

                        if not strict_pm:
                            score -= 2.0 * max(0.0, float(pm_min) - float(p_ai))
                            score -= 2.0 * max(0.0, float(pm_min) - float(p_bj))

                        if best_pair is None or float(score) > float(best_pair[0]):
                            best_pair = (float(score), int(i), int(j))

                if best_pair is not None:
                    idx_q1, idx_q2 = int(best_pair[1]), int(best_pair[2])
                    picked_by_pm = True
                    p1 = float(_pm_val(idx_q1, col_a))
                    p2 = float(_pm_val(idx_q2, col_b))
                    pm_debug["pm_q1"] = {"idx": int(idx_q1), "p": float(p1), "col": col_a}
                    pm_debug["pm_q2"] = {"idx": int(idx_q2), "p": float(p2), "col": col_b}
                    if float(p1) < float(pm_min) or float(p2) < float(pm_min):
                        warnings.append(
                            f"Low junction participation for picked qubit modes: p1={p1:.3g}, p2={p2:.3g} (<{pm_min:g})"
                        )



    if strict_pm and not picked_by_pm:
        raise ValueError("Strict PM picking failed to select qubit modes")

    if (
        not strict_windows
        and focus_pair in {1, 2}
        and freqs_GHz is not None
        and (q_min is not None or q_max is not None)
        and q_max_dist_val_used is not None
        and pick_use_pm
        and res is not None
        and freqs_GHz is not None
    ):
        focus_idx = int(idx_q1) if int(focus_pair) == 1 else int(idx_q2)
        try:
            f_focus = float(freqs_GHz[int(focus_idx)])
        except Exception:
            f_focus = float("nan")

        focus_dist = float(_dist_to_window(float(f_focus), q_min, q_max)) if np.isfinite(f_focus) else float("inf")
        need_repick = (not np.isfinite(f_focus)) or (
            float(focus_dist) > (0.0 if focus_require_in_window else float(q_max_dist_val_used))
        )
        if need_repick:
            other_idx = int(idx_q2) if int(focus_pair) == 1 else int(idx_q1)
            try:
                pm = _extract_pm_normed(res)
            except Exception:
                pm = None

            best_alt: Optional[Tuple[float, int]] = None
            pm_df = pm
            if pm_df is not None and len(pm_df.columns) >= 2 and len(pm_df.index) >= n:
                jj1_pref = os.environ.get("METAL_2Q_PICK_JJ1", "jj1")
                jj2_pref = os.environ.get("METAL_2Q_PICK_JJ2", "jj2")
                c1 = _find_pm_column(pm_df, jj1_pref)
                c2 = _find_pm_column(pm_df, jj2_pref)
                focus_col = c1 if int(focus_pair) == 1 else c2

                def _pm_val_local(i: int, col: str) -> float:
                    v = _pm_cell_value(pm_df, int(i), str(col))
                    return float(v) if v is not None else 0.0

                if focus_col is not None:
                    for cand in [int(x) for x in q_pool]:
                        if int(cand) == int(other_idx):
                            continue
                        try:
                            f = float(freqs_GHz[int(cand)])
                        except Exception:
                            continue
                        if not np.isfinite(f):
                            continue
                        d = float(_dist_to_window(float(f), q_min, q_max))
                        if float(d) > (0.0 if focus_require_in_window else float(q_max_dist_val_used)):
                            continue
                        p_focus = float(_pm_val_local(int(cand), str(focus_col)))
                        score = float(p_focus) + 0.001 * float(diag[int(cand)])
                        if best_alt is None or float(score) > float(best_alt[0]):
                            best_alt = (float(score), int(cand))

            if best_alt is None:
                for cand in [int(x) for x in q_pool]:
                    if int(cand) == int(other_idx):
                        continue
                    try:
                        f = float(freqs_GHz[int(cand)])
                    except Exception:
                        continue
                    if not np.isfinite(f):
                        continue
                    d = float(_dist_to_window(float(f), q_min, q_max))
                    if float(d) > (0.0 if focus_require_in_window else float(q_max_dist_val_used)):
                        continue
                    score = -float(d) + 1e-6 * float(diag[int(cand)])
                    if best_alt is None or float(score) > float(best_alt[0]):
                        best_alt = (float(score), int(cand))

            if best_alt is not None:
                if int(focus_pair) == 1:
                    idx_q1 = int(best_alt[1])
                else:
                    idx_q2 = int(best_alt[1])
                warnings.append(
                    f"Focused mode picking (pair={focus_pair}): repicked focus qubit mode to satisfy window distance <= {0.0 if focus_require_in_window else float(q_max_dist_val_used):g} GHz"
                )
            else:
                warnings.append(
                    f"Focused mode picking (pair={focus_pair}): could not repick focus qubit mode to satisfy window distance <= {0.0 if focus_require_in_window else float(q_max_dist_val_used):g} GHz; keeping original selection"
                )

    remaining = [int(i) for i in range(n) if i not in (idx_q1, idx_q2)]
    if strict_pm:
        r_pool = [int(i) for i in remaining if _in_window(int(i), r_min, r_max)]
        r_info = {
            "window_GHz": [r_min, r_max],
            "window_used_GHz": [r_min, r_max],
            "expand_GHz": 0.0,
            "candidates": [int(i) for i in r_pool],
            "strict_windows": bool(strict_windows),
        }
        if len(r_pool) < 2:
            if strict_windows:
                raise ValueError(f"Resonator window [{r_min},{r_max}] produced only {len(r_pool)} candidate(s) under strict windowing")
            r_pool = list(remaining)
            r_info["candidates"] = [int(i) for i in r_pool]
            warnings.append(
                f"Resonator window [{r_min},{r_max}] produced <2 candidates; falling back to all remaining modes"
            )
    else:
        r_pool, r_info = _pick_candidates_with_expansion(
            remaining,
            lo=r_min,
            hi=r_max,
            needed=2,
            label="Resonator",
            warnings=warnings,
        )
        if len(r_pool) < 2:
            r_pool = list(remaining)
            r_info["candidates"] = [int(i) for i in r_pool]
            warnings.append(
                f"Resonator window [{r_min},{r_max}] produced <2 candidates even after expansion; falling back to all remaining modes"
            )

    r_kerr_max_val = 1.0
    try:
        if r_kerr_max is not None and np.isfinite(r_kerr_max) and float(r_kerr_max) > 0:
            r_kerr_max_val = float(r_kerr_max)
    except Exception:
        r_kerr_max_val = 1.0

    r_pm_max_val = 0.05
    try:
        if r_pm_max is not None and np.isfinite(r_pm_max) and float(r_pm_max) > 0:
            r_pm_max_val = float(r_pm_max)
    except Exception:
        r_pm_max_val = 0.05

    pm_for_r = None
    r_pm_cols: List[str] = []
    if res is not None:
        try:
            pm_for_r = _extract_pm_normed(res)
        except Exception:
            pm_for_r = None
    if pm_for_r is not None and len(pm_for_r.columns) >= 1 and len(pm_for_r.index) >= n:
        c1 = _find_pm_column(pm_for_r, os.environ.get("METAL_2Q_PICK_JJ1", "jj1"))
        c2 = _find_pm_column(pm_for_r, os.environ.get("METAL_2Q_PICK_JJ2", "jj2"))
        if c1 is not None:
            r_pm_cols.append(str(c1))
        if c2 is not None and str(c2) not in r_pm_cols:
            r_pm_cols.append(str(c2))
        if not r_pm_cols:
            r_pm_cols = [str(c) for c in list(pm_for_r.columns)]

    def _freq_for(i: int) -> Optional[float]:
        if freqs_GHz is None:
            return None
        try:
            f = float(freqs_GHz[int(i)])
        except Exception:
            return None
        return float(f) if np.isfinite(f) else None

    def _res_max_pm(i: int) -> Optional[float]:
        if pm_for_r is None or not r_pm_cols:
            return None
        vals: List[float] = []
        for col in r_pm_cols:
            v = _pm_cell_value(pm_for_r, int(i), str(col))
            if v is not None:
                vals.append(float(v))
        return max(vals) if vals else None

    def _res_reject_reason(i: int) -> Optional[str]:
        self_kerr = float(diag[int(i)])
        if self_kerr > float(r_kerr_max_val):
            return f"|self-Kerr|={self_kerr:.3g}MHz>{float(r_kerr_max_val):.3g}MHz"
        pmax = _res_max_pm(int(i))
        if pmax is not None and float(pmax) > float(r_pm_max_val):
            return f"max_JJ_participation={float(pmax):.3g}>{float(r_pm_max_val):.3g}"
        return None

    def _apply_resonator_like_filter(pool: List[int]) -> Tuple[List[int], List[TDict[str, Any]]]:
        raw: List[int] = []
        seen: set[int] = set()
        for item in pool:
            i = int(item)
            if i not in seen:
                raw.append(i)
                seen.add(i)
        kept: List[int] = []
        rejected: List[TDict[str, Any]] = []
        for i in raw:
            reason = _res_reject_reason(int(i))
            if reason is None:
                kept.append(int(i))
            else:
                rejected.append(
                    {
                        "idx": int(i),
                        "f_GHz": _freq_for(int(i)),
                        "reason": reason,
                        "self_kerr_MHz": float(diag[int(i)]),
                        "max_JJ_participation": _res_max_pm(int(i)),
                    }
                )
        return kept, rejected

    filtered_r_pool, rejected_r_pool = _apply_resonator_like_filter([int(i) for i in r_pool])
    if len(filtered_r_pool) >= 2:
        if len(filtered_r_pool) < len(r_pool):
            warnings.append(
                f"Filtered resonator candidates by |self-Kerr|<={r_kerr_max_val:g}MHz and max JJ participation<={r_pm_max_val:g} "
                f"(kept {len(filtered_r_pool)}/{len(r_pool)})"
            )
        r_pool = list(filtered_r_pool)
    else:
        expanded_r_pool, rejected_expanded = _apply_resonator_like_filter([int(i) for i in remaining])
        if len(expanded_r_pool) >= 2:
            warnings.append(
                f"Resonator-like filter left {len(filtered_r_pool)}/{len(r_pool)} window candidates; expanded to all remaining modes "
                f"and kept {len(expanded_r_pool)} resonator-like candidates"
            )
            r_pool = list(expanded_r_pool)
            rejected_r_pool = rejected_expanded
        else:
            warnings.append(
                f"Resonator-like filter found only {len(expanded_r_pool)} usable candidate(s); keeping unfiltered resonator pool"
            )

    r_info["candidates"] = [int(i) for i in r_pool]
    r_info["resonator_like_filter"] = {
        "self_kerr_max_MHz": float(r_kerr_max_val),
        "max_JJ_participation": float(r_pm_max_val),
        "pm_cols": list(r_pm_cols),
        "rejected": rejected_r_pool,
    }

    # Choose a distinct (r1, r2) pair that best matches (q1,q2).
    best: Optional[Tuple[int, int, float]] = None
    for r1 in r_pool:
        for r2 in r_pool:
            if int(r2) == int(r1):
                continue
            s = float(score_res(idx_q1, int(r1)) + score_res(idx_q2, int(r2)))
            if best is None or s > float(best[2]):
                best = (int(r1), int(r2), float(s))

    if best is None:
        raise ValueError("Failed to pick resonator modes (no candidates)")

    r1, r2 = int(best[0]), int(best[1])
    invalid_reasons: List[str] = []

    def _require_window(label: str, i: int, lo: Optional[float], hi: Optional[float]) -> None:
        f = _freq_for(int(i))
        if f is None or lo is None and hi is None:
            return
        if lo is not None and float(f) < float(lo):
            invalid_reasons.append(f"{label} mode {int(i)} frequency {f:.6g} GHz is below [{lo},{hi}] GHz")
        if hi is not None and float(f) > float(hi):
            invalid_reasons.append(f"{label} mode {int(i)} frequency {f:.6g} GHz is above [{lo},{hi}] GHz")

    if int(idx_q1) == int(idx_q2):
        invalid_reasons.append("Q1 and Q2 selected the same eigenmode")
    if int(r1) == int(r2):
        invalid_reasons.append("R1 and R2 selected the same eigenmode")
    if int(idx_q1) in {int(r1), int(r2)} or int(idx_q2) in {int(r1), int(r2)}:
        invalid_reasons.append("Selected qubit and resonator eigenmodes are not distinct")

    _require_window("Q1", int(idx_q1), q_min, q_max)
    _require_window("Q2", int(idx_q2), q_min, q_max)
    _require_window("R1", int(r1), r_min, r_max)
    _require_window("R2", int(r2), r_min, r_max)

    pm_q1 = (pm_debug.get("pm_q1") or {}) if isinstance(pm_debug, dict) else {}
    pm_q2 = (pm_debug.get("pm_q2") or {}) if isinstance(pm_debug, dict) else {}
    p1_final = None
    p2_final = None
    try:
        p1_final = float(pm_q1.get("p")) if pm_q1.get("p") is not None else None
    except Exception:
        p1_final = None
    try:
        p2_final = float(pm_q2.get("p")) if pm_q2.get("p") is not None else None
    except Exception:
        p2_final = None

    if pick_use_pm and (p1_final is None or p2_final is None):
        try:
            pm_final = _extract_pm_normed(res or {})
        except Exception:
            pm_final = None
        pm_cols = list((pm_debug or {}).get("pm_cols") or [])
        if pm_final is not None and len(pm_final.columns) >= 2:
            if len(pm_cols) < 2:
                c1_final = _find_pm_column(pm_final, os.environ.get("METAL_2Q_PICK_JJ1", "jj1"))
                c2_final = _find_pm_column(pm_final, os.environ.get("METAL_2Q_PICK_JJ2", "jj2"))
                if c1_final is not None and c2_final is not None and c1_final != c2_final:
                    pm_cols = [str(c1_final), str(c2_final)]
                else:
                    pm_cols = [str(c) for c in list(pm_final.columns)[:2]]
            if len(pm_cols) >= 2:
                if p1_final is None:
                    p1_final = _pm_cell_value(pm_final, int(idx_q1), str(pm_cols[0]))
                if p2_final is None:
                    p2_final = _pm_cell_value(pm_final, int(idx_q2), str(pm_cols[1]))
                pm_debug.setdefault("pm_cols", [str(pm_cols[0]), str(pm_cols[1])])
                pm_debug["pm_q1_final"] = {"idx": int(idx_q1), "p": p1_final, "col": str(pm_cols[0])}
                pm_debug["pm_q2_final"] = {"idx": int(idx_q2), "p": p2_final, "col": str(pm_cols[1])}

    if pick_use_pm:
        if p1_final is None:
            invalid_reasons.append("Missing Q1 junction participation for selected mode")
        elif float(p1_final) < float(valid_pm_min):
            invalid_reasons.append(
                f"Q1 junction participation {float(p1_final):.3g} is below valid threshold {float(valid_pm_min):.3g}"
            )
        if p2_final is None:
            invalid_reasons.append("Missing Q2 junction participation for selected mode")
        elif float(p2_final) < float(valid_pm_min):
            invalid_reasons.append(
                f"Q2 junction participation {float(p2_final):.3g} is below valid threshold {float(valid_pm_min):.3g}"
            )

    if invalid_reasons:
        warnings.append("Invalid mode selection: " + "; ".join(invalid_reasons))

    pm_debug.setdefault("pm_cols", list(pm_debug.get("pm_cols") or []))
    pm_debug.setdefault("pm_q1_final", pm_debug.get("pm_q1_final"))
    pm_debug.setdefault("pm_q2_final", pm_debug.get("pm_q2_final"))

    debug: TDict[str, Any] = {
        "n_modes": int(n),
        "q_window_GHz": [q_min, q_max],
        "r_window_GHz": [r_min, r_max],
        "q_pick": q_info,
        "r_pick": r_info,
        "picked_by_pm": bool(picked_by_pm),
        "pm": pm_debug,
        "kerr_floor_MHz": float(kerr_floor) if kerr_floor is not None else None,
        "q_kerr_min_MHz": float(q_kerr_min_val),
        "focus_require_in_window": bool(focus_require_in_window) if focus_pair in {1, 2} else None,
        "n_qubit_like": int(len(q_like)),
        "selected": {"q1": int(idx_q1), "q2": int(idx_q2), "r1": int(r1), "r2": int(r2)},
        "selected_freqs_GHz": {
            "q1": _freq_for(int(idx_q1)),
            "q2": _freq_for(int(idx_q2)),
            "r1": _freq_for(int(r1)),
            "r2": _freq_for(int(r2)),
        },
        "valid_pm_min": float(valid_pm_min),
        "valid": not bool(invalid_reasons),
        "invalid_reasons": list(invalid_reasons),
        "warnings": list(warnings),
    }

    return idx_q1, idx_q2, int(r1), int(r2), debug


def _calc_g_from_chi(
    *,
    chi_MHz: float,
    Delta_GHz: float,
    alpha_MHz: float,
) -> Optional[float]:
    """Approximate g (in GHz) from dispersive shift chi (MHz)."""
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


def _qi_for_mode(res: TDict[str, Any], idx_mode: int, *, warnings: List[str]) -> Optional[float]:
    Qi_val = None

    p = _extract_participation_value(res, idx_mode, kind="dielectrics_bulk", name="main")
    Qi_from_p = _qi_from_p_tandelta(p, base.TAN_DELTA_MAIN)
    if Qi_from_p is not None:
        Qi_val = float(Qi_from_p)
        p_val: Optional[float] = None
        try:
            p_val = float(p) if p is not None else None
        except Exception:
            p_val = None
        p_str = f"{p_val:.3g}" if p_val is not None else "None"
        warnings.append(
            f"Qi from p*tan_delta: p={p_str}, tan_delta={float(base.TAN_DELTA_MAIN):.3g}, Qi={Qi_val:.3g}"
        )

    if Qi_val is None:
        Qi_raw = _extract_qdielectric_main(res, idx_mode)
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


def _env_int(name: str, default: int, *, lo: int, hi: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return int(default)
    try:
        v = int(float(raw))
    except Exception:
        return int(default)
    return int(max(int(lo), min(int(hi), int(v))))


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
    return float(max(float(lo), min(float(hi), float(v))))


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return bool(default)
    if raw in {"1", "true", "yes", "y", "on"}:
        return True
    if raw in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def _ansys_project_target(root: Path, default_name: str) -> Path:
    workdir = base._ansys_short_workdir(root)
    if not _env_bool("METAL_2Q_ANSYS_REUSE_PROJECT_FILE", True):
        return workdir / default_name

    raw = os.environ.get("METAL_2Q_ANSYS_PROJECT_FILE", "").strip().strip('"').strip("'")
    if not raw:
        raw = DEFAULT_2Q_ANSYS_PROJECT_FILE
    target = Path(raw)
    if target.suffix.lower() != ".aedt":
        target = target.with_suffix(".aedt")
    if not target.is_absolute():
        target = workdir / target
    return target


def _configure_renderer_project_file(renderer: Any, target_aedt: Path) -> None:
    """Bind Qiskit Metal's Ansys renderer to a fixed AEDT project before start()."""
    try:
        target_aedt.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    project_path = str(target_aedt.parent)
    project_name = target_aedt.stem
    try:
        renderer.options["project_path"] = project_path
        renderer.options["project_name"] = project_name
    except Exception:
        pass
    try:
        renderer._options["project_path"] = project_path
        renderer._options["project_name"] = project_name
    except Exception:
        pass


def _save_renderer_active_project_as(renderer: Any, target_aedt: Path) -> bool:
    target_aedt.parent.mkdir(parents=True, exist_ok=True)
    errors: List[str] = []

    try:
        pinfo = getattr(renderer, "pinfo", None)
        project = getattr(pinfo, "project", None) if pinfo else None
        if project is not None and hasattr(project, "save"):
            project.save(str(target_aedt))
            return target_aedt.exists()
    except Exception as e:
        errors.append(f"pinfo.project.save: {e}")

    try:
        if hasattr(renderer, "rdesktop") and renderer.rdesktop:
            project = renderer.rdesktop.get_active_project()
            if project:
                project.SaveAs(str(target_aedt), True)
                return target_aedt.exists()
    except Exception as e:
        errors.append(f"rdesktop.active_project.SaveAs: {e}")

    try:
        import pyEPR as epr

        app = epr.ansys.HfssApp()
        desktop = app.get_app_desktop()
        project = desktop.get_active_project()
        if project:
            project.save(str(target_aedt))
            return target_aedt.exists()
    except Exception as e:
        errors.append(f"pyEPR.active_project.save: {e}")

    if errors:
        print(f"[ANSYS] Could not save fixed project {target_aedt}: {'; '.join(errors)}", flush=True)
    return False


def _fixed_project_file_looks_usable(target_aedt: Path) -> bool:
    try:
        return bool(target_aedt.is_file() and target_aedt.stat().st_size > 100_000)
    except Exception:
        return False


def _clear_renderer_project_file(renderer: Any) -> None:
    try:
        renderer.options["project_path"] = None
        renderer.options["project_name"] = None
    except Exception:
        pass
    try:
        renderer._options["project_path"] = None
        renderer._options["project_name"] = None
    except Exception:
        pass


def _connect_renderer_to_open_project(renderer: Any, target_aedt: Path) -> bool:
    project_name = target_aedt.stem
    try:
        import pyEPR as epr

        app = epr.ansys.HfssApp()
        desktop = app.get_app_desktop()
        names = list(desktop.get_project_names())
        if project_name not in names:
            return False
        desktop.set_active_project(project_name)
        if hasattr(renderer, "rapp"):
            renderer.rapp = app
        if hasattr(renderer, "rdesktop"):
            renderer.rdesktop = desktop
        if hasattr(renderer, "connect_ansys"):
            renderer.connect_ansys(project_name=project_name)
        return True
    except Exception as e:
        print(f"[ANSYS] Could not connect to open {project_name}: {e}", flush=True)
        return False


def _refresh_renderer_connection(renderer: Any, target_aedt: Path) -> None:
    try:
        if hasattr(renderer, "connect_ansys"):
            renderer.connect_ansys(project_name=target_aedt.stem)
    except Exception:
        pass


def _start_renderer_with_project_file(renderer: Any, target_aedt: Path) -> None:
    """Start renderer on one fixed project file; create it once if it is missing."""
    if _connect_renderer_to_open_project(renderer, target_aedt):
        _configure_renderer_project_file(renderer, target_aedt)
        return

    _configure_renderer_project_file(renderer, target_aedt)
    try:
        renderer.start()
        _refresh_renderer_connection(renderer, target_aedt)
        return
    except Exception as e:
        print(f"[ANSYS] Project2 direct start failed, retrying via active project: {e}", flush=True)

    if _fixed_project_file_looks_usable(target_aedt):
        _configure_renderer_project_file(renderer, target_aedt)
        try:
            renderer.start()
            _refresh_renderer_connection(renderer, target_aedt)
            return
        except Exception as e:
            print(f"[ANSYS] Fixed project open failed, recreating {target_aedt}: {e}", flush=True)

    _clear_renderer_project_file(renderer)

    renderer.start()
    time.sleep(0.8)
    _save_renderer_active_project_as(renderer, target_aedt)
    _configure_renderer_project_file(renderer, target_aedt)
    _refresh_renderer_connection(renderer, target_aedt)


def _shared_layout_to_dict(shared: Any) -> TDict[str, Any]:
    if hasattr(shared, "__dataclass_fields__"):
        return {k: getattr(shared, k) for k in shared.__dataclass_fields__.keys()}
    if hasattr(shared, "__dict__"):
        return dict(shared.__dict__)
    return {"value": shared}


def _layout2qparams_to_dict(p_local: Layout2QParams) -> TDict[str, Any]:
    out: TDict[str, Any] = {}
    for field_name in Layout2QParams.__dataclass_fields__.keys():
        value = getattr(p_local, field_name)
        if field_name == "shared":
            out[field_name] = _shared_layout_to_dict(value)
        else:
            out[field_name] = value
    return out


def _qdielectric_for_mode(res: TDict[str, Any], idx_mode: int) -> Optional[float]:
    p = _extract_participation_value(res, idx_mode, kind="dielectrics_bulk", name="main")
    q_from_p = _qi_from_p_tandelta(p, base.TAN_DELTA_MAIN)
    if q_from_p is not None:
        return float(q_from_p)

    q_raw = _extract_qdielectric_main(res, idx_mode)
    if q_raw is not None and np.isfinite(q_raw) and q_raw > 0:
        return float(q_raw)
    return None


def _configure_two_junctions_best_effort(eig, *, lj_vars: Tuple[str, str], cj_vars: Tuple[str, str]) -> None:
    """Try to ensure both junctions are wired to Lj1/Cj1 and Lj2/Cj2."""
    try:
        pinfo = eig.sim.renderer.pinfo
    except Exception:
        return

    try:
        juncs = getattr(pinfo, "junctions", {})
        if not isinstance(juncs, dict):
            juncs = {}
    except Exception:
        juncs = {}

    def _set_vars_only(jname: str, *, Lj: str, Cj: str) -> None:
        try:
            jinfo = dict(pinfo.junctions[jname])
        except Exception:
            jinfo = {}
        jinfo["Lj_variable"] = Lj
        jinfo["Cj_variable"] = Cj
        pinfo.junctions[jname] = jinfo
        try:
            eig.setup.junctions[jname] = jinfo
        except Exception:
            pass

    # Prefer Qiskit Metal helper APIs when available.
    try:
        if hasattr(eig, "del_junction") and hasattr(eig, "add_junction"):
            eig.del_junction()

            def _try_add(jj_name: str, *, Lj: str, Cj: str, qname: str) -> bool:
                # Pattern A: tutorial naming (JJ_rect_Lj_Q1_rect_jj)
                patterns = [
                    (f"JJ_rect_Lj_{qname}_rect_jj", f"JJ_Lj_{qname}_rect_jj_"),
                    # Pattern B: older/local variant that includes variable name token
                    (f"JJ_rect_{Lj}_{qname}_rect_jj", f"JJ_{Lj}_{qname}_rect_jj_"),
                ]
                for rect, line in patterns:
                    try:
                        eig.add_junction(jj_name, Lj, Cj, rect=rect, line=line)
                        # Best-effort validate if available
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
            return
    except Exception:
        # Fall back to direct pinfo mutation.
        pass

    if juncs:
        # Prefer name matching; fallback to key order.
        keys = list(juncs.keys())
        q1_keys = [k for k in keys if "q1" in str(k).lower()]
        q2_keys = [k for k in keys if "q2" in str(k).lower()]
        used = set()

        if q1_keys:
            _set_vars_only(q1_keys[0], Lj=lj_vars[0], Cj=cj_vars[0])
            used.add(q1_keys[0])
        if q2_keys:
            _set_vars_only(q2_keys[0], Lj=lj_vars[1], Cj=cj_vars[1])
            used.add(q2_keys[0])

        remaining = [k for k in keys if k not in used]
        if len(used) < 2 and remaining:
            if len(used) == 0:
                _set_vars_only(remaining[0], Lj=lj_vars[0], Cj=cj_vars[0])
                used.add(remaining[0])
                remaining = remaining[1:]
            if len(used) == 1 and remaining:
                _set_vars_only(remaining[0], Lj=lj_vars[1], Cj=cj_vars[1])
        return

    # No junctions: create best-effort names.
    # No junctions detected: synthesize with a validation-based fallback.
    def _mk(qname: str, *, Lj: str, Cj: str, rect: str, line: str) -> TDict[str, str]:
        return {"Lj_variable": Lj, "Cj_variable": Cj, "rect": rect, "line": line}

    candidates = [
        (
            _mk("Q1", Lj=lj_vars[0], Cj=cj_vars[0], rect="JJ_rect_Lj_Q1_rect_jj", line="JJ_Lj_Q1_rect_jj_"),
            _mk("Q2", Lj=lj_vars[1], Cj=cj_vars[1], rect="JJ_rect_Lj_Q2_rect_jj", line="JJ_Lj_Q2_rect_jj_"),
        ),
        (
            _mk(
                "Q1",
                Lj=lj_vars[0],
                Cj=cj_vars[0],
                rect=f"JJ_rect_{lj_vars[0]}_Q1_rect_jj",
                line=f"JJ_{lj_vars[0]}_Q1_rect_jj_",
            ),
            _mk(
                "Q2",
                Lj=lj_vars[1],
                Cj=cj_vars[1],
                rect=f"JJ_rect_{lj_vars[1]}_Q2_rect_jj",
                line=f"JJ_{lj_vars[1]}_Q2_rect_jj_",
            ),
        ),
    ]

    for j1, j2 in candidates:
        try:
            pinfo.junctions.clear()
        except Exception:
            pass
        try:
            pinfo.junctions["jj1"] = j1
            pinfo.junctions["jj2"] = j2
            pinfo.validate_junction_info()
            try:
                eig.setup.junctions["jj1"] = pinfo.junctions["jj1"]
                eig.setup.junctions["jj2"] = pinfo.junctions["jj2"]
            except Exception:
                pass
            return
        except Exception:
            continue



# -------------------------
# analysis
# -------------------------
def run_analysis_pipeline_2q(
    design,
    sample_id: str,
    *,
    root_path: str,
    do_hfss: bool,
    do_q3d: bool,
    lj1_nh: float,
    cj1_fF_user: float,
    lj2_nh: float,
    cj2_fF_user: float,
    q1_effective_cin_fF: float = 0.0,
    q2_effective_cin_fF: float = 0.0,
    q1_target_kext_hz: float = 0.0,
    q2_target_kext_hz: float = 0.0,
    existing_payload: Optional[TDict[str, Any]] = None,
    save_json: bool = True,
    layout_params: Optional[Layout2QParams] = None,
) -> TDict[str, Any]:
    root = Path(root_path)
    gds_subdir = os.environ.get("METAL_2Q_GDS_SUBDIR", "gds").strip().strip("/\\") or "gds"
    json_subdir = os.environ.get("METAL_2Q_JSON_SUBDIR", "json").strip().strip("/\\") or "json"
    gds_dir = root / gds_subdir
    json_dir_out = root / json_subdir
    gds_dir.mkdir(parents=True, exist_ok=True)
    json_dir_out.mkdir(parents=True, exist_ok=True)

    payload = copy.deepcopy(existing_payload) if existing_payload is not None else build_chip_summary_multiq(sample_id=sample_id)
    payload["status"] = "running"

    payload["meta"]["filename_base"] = sample_id
    sim_tag = base.short_hash_tag(sample_id, n=8)
    payload["meta"]["ansys_sim_tag"] = sim_tag
    if layout_params is not None:
        payload["meta"]["layout"] = base.to_jsonable(_layout2qparams_to_dict(layout_params))
    payload["meta"]["tan_delta_main"] = float(base.TAN_DELTA_MAIN)
    payload["meta"]["junctions"] = {
        "Lj1_nH": float(lj1_nh),
        "Cj1_fF": float(cj1_fF_user),
        "Lj2_nH": float(lj2_nh),
        "Cj2_fF": float(cj2_fF_user),
    }
    payload["meta"]["external_coupling"] = {
        "Q1_EFFECTIVE_CIN_FF": float(q1_effective_cin_fF),
        "Q2_EFFECTIVE_CIN_FF": float(q2_effective_cin_fF),
        "Q1_TARGET_KEXT_HZ": float(q1_target_kext_hz),
        "Q2_TARGET_KEXT_HZ": float(q2_target_kext_hz),
    }
    payload["meta"]["positions_mm"] = {
        "Q1": {"x": _try_get_component_option_mm(design, "Q1", "pos_x"), "y": _try_get_component_option_mm(design, "Q1", "pos_y")},
        "Q2": {"x": _try_get_component_option_mm(design, "Q2", "pos_x"), "y": _try_get_component_option_mm(design, "Q2", "pos_y")},
    }

    # 1) GDS
    try:
        gds_path = gds_dir / f"{sample_id}.gds"
        abs_gds_path = str(gds_path.resolve())
        info = base.export_gds_robust(design, abs_gds_path)
        payload["meta"]["gds_path"] = abs_gds_path
        payload["meta"]["gds_subdir"] = gds_subdir
        payload["meta"]["gds_export_mode"] = info["mode"]
        payload["meta"]["gds_error"] = info.get("error") or ""
    except Exception as e:
        payload["meta"]["gds_error"] = str(e)

    # 2) HFSS/EPR
    # Variables: per-qubit junction variables
    lj_vars = ("Lj1", "Lj2")
    cj_vars = ("Cj1", "Cj2")
    hfss_ok_final = not bool(do_hfss)

    if do_hfss:
        hfss_watchdog_s = _env_float("METAL_2Q_HFSS_WATCHDOG_S", 0.0, lo=0.0, hi=86400.0)
        for attempt in [1, 2]:
            eig = None
            hfss = None
            hfss_ok = False
            try:
                if attempt > 1 and _env_bool("METAL_2Q_ANSYS_RESET_ON_RETRY", True):
                    base._ansys_best_effort_reset()
                    time.sleep(1.0)
                eig = base.EPRanalysis(design, "hfss")
                hfss = eig.sim.renderer

                try:
                    eig.sim.setup.reuse_selected_design = False
                    eig.sim.setup.reuse_setup = False
                except Exception:
                    pass

                if hfss is None:
                    raise RuntimeError("HFSS renderer is not available (eig.sim.renderer is None).")

                target = _ansys_project_target(root, f"EIG2Q_{sim_tag}_a{attempt}.aedt")
                payload["meta"]["ansys_hfss_project_file"] = str(target.resolve())

                print(f"[HFSS] attempt={attempt} Starting HFSS...", flush=True)
                _start_renderer_with_project_file(hfss, target)
                time.sleep(0.8)
                _save_renderer_active_project_as(hfss, target)
                try:
                    payload["meta"]["ansys_hfss_project_name"] = str(getattr(hfss.pinfo, "project_name", ""))
                    payload["meta"]["ansys_hfss_project_path"] = str(getattr(hfss.pinfo, "project_path", ""))
                except Exception:
                    pass

                eig.sim.setup.vars = base.Dict(
                    Lj1=f"{float(lj1_nh)} nH",
                    Cj1=f"{float(cj1_fF_user)} fF",
                    Lj2=f"{float(lj2_nh)} nH",
                    Cj2=f"{float(cj2_fF_user)} fF",
                )

                hfss_mesh_options = {
                    "max_mesh_length_jj": os.environ.get("METAL_2Q_HFSS_MAX_MESH_LENGTH_JJ", "7um"),
                    "max_mesh_length_port": os.environ.get("METAL_2Q_HFSS_MAX_MESH_LENGTH_PORT", "7um"),
                }
                try:
                    for _mesh_key, _mesh_val in hfss_mesh_options.items():
                        hfss.options[_mesh_key] = _mesh_val
                except Exception as _mesh_e:
                    payload["meta"]["hfss_mesh_options_error"] = str(_mesh_e)
                payload["meta"]["hfss_mesh_options"] = base.to_jsonable(hfss_mesh_options)

                hfss_setup_updates = {
                    "n_modes": _env_int("METAL_2Q_HFSS_N_MODES", 10, lo=4, hi=20),
                    "max_passes": _env_int("METAL_2Q_HFSS_MAX_PASSES", 10, lo=1, hi=25),
                    "min_passes": _env_int("METAL_2Q_HFSS_MIN_PASSES", 1, lo=1, hi=20),
                    "min_converged": _env_int("METAL_2Q_HFSS_MIN_CONVERGED", 1, lo=1, hi=5),
                    "min_freq_ghz": _env_float("METAL_2Q_HFSS_MIN_FREQ_GHZ", 1.0, lo=0.1, hi=20.0),
                    "max_delta_f": _env_float("METAL_2Q_HFSS_MAX_DELTA_F_GHZ", 0.1, lo=0.001, hi=1.0),
                    "pct_refinement": _env_int("METAL_2Q_HFSS_PCT_REFINEMENT", 30, lo=1, hi=100),
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
                eig.sim.setup.cos_trunc = None
                eig.sim.setup.fock_trunc = None

                all_components = list(design.components.keys())
                components = _components_override("METAL_2Q_HFSS_COMPONENTS") or all_components
                print(f"[HFSS] Running simulation with components: {components}", flush=True)
                with _ansys_watchdog(hfss_watchdog_s, label=f"HFSS2Q_{sim_tag}_a{attempt}"):
                    eig.sim.run(
                        name=f"Eig2Q_{sim_tag}_a{attempt}",
                        components=components,
                        open_terminations=[],
                        box_plus_buffer=True,
                    )

                _configure_two_junctions_best_effort(eig, lj_vars=lj_vars, cj_vars=cj_vars)

                eig.setup.dissipatives = {"dielectrics_bulk": ["main"]}
                try:
                    if hasattr(eig.setup, "dissipative"):
                        eig.setup.dissipative = {"dielectrics_bulk": {"main": base.TAN_DELTA_MAIN}}
                except Exception:
                    pass

                try:
                    import importlib

                    _cda = importlib.import_module("pyEPR.core_distributed_analysis")

                    if not getattr(_cda, "_metal_safe_hfss_report_f_convergence", False):
                        _orig = _cda.DistributedAnalysis.hfss_report_f_convergence

                        def _safe_hfss_report_f_convergence(self, variation: str = "0", save_csv: bool = True):
                            try:
                                return _orig(self, variation=variation, save_csv=save_csv)
                            except Exception:
                                return None

                        setattr(
                            _cda.DistributedAnalysis,
                            "hfss_report_f_convergence",
                            _safe_hfss_report_f_convergence,
                        )
                        setattr(_cda, "_metal_safe_hfss_report_f_convergence", True)
                except Exception:
                    pass

                epr_error: Optional[Exception] = None
                try:
                    eig.clear_data()
                    eig.get_stored_energy(no_junctions=False)
                    eig.run_analysis()
                    try:
                        import importlib

                        _epr = importlib.import_module("pyEPR")

                        renderer = eig.sim.renderer
                        if renderer is None:
                            raise RuntimeError("HFSS renderer is not available")

                        da = getattr(renderer, "epr_distributed_analysis", None)
                        data_filename = getattr(da, "data_filename", None) if da is not None else None
                        if not data_filename:
                            raise RuntimeError("Missing epr_distributed_analysis.data_filename")

                        qa = _epr.QuantumAnalysis(data_filename)
                        setattr(renderer, "epr_quantum_analysis", qa)
                        qa.analyze_all_variations(cos_trunc=None, fock_trunc=None, print_result=False)
                    except Exception as _e:
                        epr_error = _e
                        payload["meta"]["epr_error"] = str(_e)
                except Exception as e:
                    epr_error = e
                    payload["meta"]["epr_error"] = str(e)

                freqs = eig.get_frequencies().iloc[:, 0].values
                payload["resonators"]["readout1"]["modes_f_GHz"] = base.to_jsonable(freqs)
                payload["resonators"]["readout2"]["modes_f_GHz"] = base.to_jsonable(freqs)

                qa = getattr(eig.sim.renderer, "epr_quantum_analysis", None)
                if qa is None or not getattr(qa, "results", None):
                    if epr_error is not None:
                        raise epr_error
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

                try:
                    idx_q1, idx_q2, idx_r1, idx_r2, pick_debug = _pick_two_qubits_and_two_resonators(
                        chi_matrix, freqs_GHz=freqs, res=res
                    )
                except Exception as pick_err:
                    payload["meta"]["mode_picker_error"] = str(pick_err)
                    raise RuntimeError(f"Invalid 2Q mode selection: {pick_err}") from pick_err
                payload["meta"]["mode_picker"] = base.to_jsonable(pick_debug)
                if not bool((pick_debug or {}).get("valid", True)):
                    msg = "; ".join(str(x) for x in (pick_debug or {}).get("invalid_reasons", []))
                    if not msg:
                        msg = "mode_picker.valid is false"
                    payload["meta"]["mode_picker_error"] = msg
                    if not _env_bool("METAL_2Q_ALLOW_INVALID_MODE_RESULT", False):
                        raise RuntimeError("Invalid 2Q mode selection: " + msg)

                selected = (pick_debug or {}).get("selected", {}) if isinstance(pick_debug, dict) else {}
                sel_values = [selected.get("q1"), selected.get("q2"), selected.get("r1"), selected.get("r2")]
                if len([x for x in sel_values if x is not None]) == 4:
                    if len(set(int(x) for x in sel_values)) != 4:
                        msg = f"duplicated selected modes {selected}"
                        payload["meta"]["mode_picker_error"] = msg
                        raise RuntimeError(f"Invalid 2Q mode selection: {msg}")

                conv_f = _safe_get_convergence_f(eig)
                payload["meta"]["hfss_convergence_f"] = base.to_jsonable(conv_f)
                conv_threshold = _env_float(
                    "METAL_2Q_HFSS_CONV_MAX_DELTA_GHZ",
                    0.02,
                    lo=0.001,
                    hi=1.0,
                )
                conv_debug: TDict[str, Any] = {}
                for label, idx in {
                    "Q1": idx_q1,
                    "Q2": idx_q2,
                    "R1": idx_r1,
                    "R2": idx_r2,
                }.items():
                    d = _last_pass_delta_ghz(conv_f, int(idx))
                    conv_debug[label] = {
                        "mode_index": int(idx),
                        "last_pass_delta_GHz": d,
                        "threshold_GHz": conv_threshold,
                    }
                payload["meta"]["hfss_last_pass_delta_GHz"] = base.to_jsonable(conv_debug)

                strict_conv_raw = os.environ.get("METAL_2Q_HFSS_STRICT_CONVERGENCE", "1").strip().lower()
                strict_conv = strict_conv_raw not in {"0", "false", "no", "off"}
                if strict_conv:
                    bad = []
                    for label, info in conv_debug.items():
                        d = info.get("last_pass_delta_GHz")
                        if d is None:
                            bad.append(
                                f"{label}: missing convergence_f for selected mode {info.get('mode_index')}"
                            )
                        elif float(d) > float(conv_threshold):
                            bad.append(
                                f"{label}: last-pass delta {float(d):.6g} GHz > {float(conv_threshold):.6g} GHz"
                            )
                    if bad:
                        payload["meta"]["hfss_convergence_error"] = "; ".join(bad)
                        raise RuntimeError("HFSS eigenmode not converged: " + "; ".join(bad))
                f_q1 = float(freqs[idx_q1])
                f_q2 = float(freqs[idx_q2])
                f_r1 = float(freqs[idx_r1])
                f_r2 = float(freqs[idx_r2])

                # self-kerr convention from 1Q script
                alpha1_MHz = -abs(float(chi_matrix[idx_q1, idx_q1]))
                alpha2_MHz = -abs(float(chi_matrix[idx_q2, idx_q2]))
                K1_MHz = float(chi_matrix[idx_r1, idx_r1])
                K2_MHz = float(chi_matrix[idx_r2, idx_r2])

                chi1_MHz = float(chi_matrix[idx_q1, idx_r1])
                chi2_MHz = float(chi_matrix[idx_q2, idx_r2])
                chi12_MHz = float(chi_matrix[idx_q1, idx_q2])

                Delta1_GHz = f_q1 - f_r1
                Delta2_GHz = f_q2 - f_r2

                g1_GHz = _calc_g_from_chi(chi_MHz=chi1_MHz, Delta_GHz=Delta1_GHz, alpha_MHz=alpha1_MHz)
                g2_GHz = _calc_g_from_chi(chi_MHz=chi2_MHz, Delta_GHz=Delta2_GHz, alpha_MHz=alpha2_MHz)

                ro1 = payload["resonators"]["readout1"]
                ro2 = payload["resonators"]["readout2"]
                q1 = payload["qubits"]["Q1"]
                q2 = payload["qubits"]["Q2"]

                pick_warns = (pick_debug or {}).get("warnings") or []
                if pick_warns:
                    ro1["warnings"].extend([str(w) for w in pick_warns])
                    ro2["warnings"].extend([str(w) for w in pick_warns])

                ro1["f_GHz"] = f_r1
                ro1["K_MHz"] = K1_MHz
                ro1["picked_qubit_mode_index"] = idx_q1
                ro1["picked_res_mode_index"] = idx_r1

                ro2["f_GHz"] = f_r2
                ro2["K_MHz"] = K2_MHz
                ro2["picked_qubit_mode_index"] = idx_q2
                ro2["picked_res_mode_index"] = idx_r2

                q1["f01_epr_GHz"] = f_q1
                q1["alpha_epr_MHz"] = alpha1_MHz
                q1["chi_MHz"] = chi1_MHz
                q1["dispersive"]["chi_GHz"] = chi1_MHz / 1e3
                q1["dispersive"]["Delta_GHz"] = Delta1_GHz
                q1["dispersive"]["g_GHz"] = g1_GHz

                q2["f01_epr_GHz"] = f_q2
                q2["alpha_epr_MHz"] = alpha2_MHz
                q2["chi_MHz"] = chi2_MHz
                q2["dispersive"]["chi_GHz"] = chi2_MHz / 1e3
                q2["dispersive"]["Delta_GHz"] = Delta2_GHz
                q2["dispersive"]["g_GHz"] = g2_GHz

                payload["chip"]["chi12_MHz"] = chi12_MHz

                # Qi and kappa_i per resonator mode
                ro1_w = ro1.get("warnings") or []
                ro2_w = ro2.get("warnings") or []
                Qi1 = _qi_for_mode(res, idx_r1, warnings=ro1_w)
                Qi2 = _qi_for_mode(res, idx_r2, warnings=ro2_w)
                ro1["Qi"] = Qi1
                ro2["Qi"] = Qi2
                ro1["kappa_i_over_2pi_Hz"] = (f_r1 * 1e9) / float(Qi1) if Qi1 else None
                ro2["kappa_i_over_2pi_Hz"] = (f_r2 * 1e9) / float(Qi2) if Qi2 else None

                # qubit dielectric Q
                q1Q = _qdielectric_for_mode(res, idx_q1)
                q2Q = _qdielectric_for_mode(res, idx_q2)
                if q1Q is not None:
                    q1["Q_dielectric_main"] = float(q1Q)
                if q2Q is not None:
                    q2["Q_dielectric_main"] = float(q2Q)

                hfss_ok = True
                hfss_ok_final = True
                for key in ("hfss_error", "hfss_traceback_tail", "mode_picker_error", "epr_error"):
                    payload["meta"].pop(key, None)
                break

            except Exception as e:
                payload["meta"]["hfss_error"] = str(e)
                try:
                    payload["meta"]["hfss_traceback_tail"] = traceback.format_exc().splitlines()[-30:]
                except Exception:
                    pass
                print(f"[HFSS] attempt={attempt} Error: {e}", flush=True)
                _print_ansys_messages(hfss, "HFSS", level=2)
                if _has_deterministic_geometry_error(hfss):
                    payload["meta"]["hfss_error"] = "deterministic_geometry_error"
                    break
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
    Ceff1_fF = None
    Ceff2_fF = None
    q3d_ok_final = not bool(do_q3d)
    if not do_q3d:
        try:
            Ceff1_fF = float(payload.get("qubits", {}).get("Q1", {}).get("C_eff_fF"))
        except Exception:
            Ceff1_fF = None
        try:
            Ceff2_fF = float(payload.get("qubits", {}).get("Q2", {}).get("C_eff_fF"))
        except Exception:
            Ceff2_fF = None
    cj1_fF = float(cj1_fF_user)
    cj2_fF = float(cj2_fF_user)

    if do_q3d:
        q3d_watchdog_s = _env_float("METAL_2Q_Q3D_WATCHDOG_S", 0.0, lo=0.0, hi=86400.0)
        for attempt in [1, 2]:
            lom = None
            lom_sim = None
            q3d = None
            q3d_ok = False
            try:
                if attempt > 1 and _env_bool("METAL_2Q_ANSYS_RESET_ON_RETRY", True):
                    base._ansys_best_effort_reset()
                    time.sleep(1.0)
                lom = base.LOManalysis(design, "q3d")
                lom_sim = lom.sim
                q3d = lom_sim.renderer

                try:
                    lom_sim.setup.reuse_selected_design = False
                    lom_sim.setup.reuse_setup = False
                except Exception:
                    pass

                if q3d is None:
                    raise RuntimeError("Q3D renderer is not available (lom.sim.renderer is None).")

                target = _ansys_project_target(root, f"Q3D2Q_{sim_tag}_a{attempt}.aedt")
                payload["meta"]["ansys_q3d_project_file"] = str(target.resolve())

                print(f"[Q3D] attempt={attempt} Starting Q3D...", flush=True)
                _start_renderer_with_project_file(q3d, target)
                time.sleep(0.8)
                _save_renderer_active_project_as(q3d, target)
                try:
                    payload["meta"]["ansys_q3d_project_name"] = str(getattr(q3d.pinfo, "project_name", ""))
                    payload["meta"]["ansys_q3d_project_path"] = str(getattr(q3d.pinfo, "project_path", ""))
                except Exception:
                    pass

                try:
                    setattr(lom_sim, "renderer_initialized", True)
                except Exception:
                    pass

                lom_sim.setup.freq_ghz = 5.0
                lom_sim.setup.max_passes = 4
                components = _components_override("METAL_2Q_Q3D_COMPONENTS") or list(design.components.keys())
                print(f"[Q3D] Running simulation...", flush=True)
                with _ansys_watchdog(q3d_watchdog_s, label=f"Q3D2Q_{sim_tag}_a{attempt}"):
                    lom_sim.run(name=f"Q3D2Q_{sim_tag}_a{attempt}", components=components, box_plus_buffer=False)

                Cmat = lom_sim.capacitance_matrix

                payload["q3d"].setdefault("runs", [])
                payload["q3d"]["runs"].append(
                    {
                        "name": f"Q3D2Q_{sim_tag}_a{attempt}",
                        "purpose": "full_chip",
                        "components": list(components),
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

                Ceff1_fF = _ceff_for_qubit("Q1", cj1_fF)
                Ceff2_fF = _ceff_for_qubit("Q2", cj2_fF)

                payload["q3d"]["internal"]["capacitances_fF"] = {
                    "units": "fF",
                    "nodes": list(Cmat.index),
                    "capacitance_matrix_fF": base.to_jsonable(Cmat),
                }

                def _annotate_cin_full_chip(cin: TDict[str, Any], *, tee_name: str) -> TDict[str, Any]:
                    a = f"cap_body_0_{tee_name}"
                    b = f"cap_body_1_{tee_name}"
                    missing: List[str] = []
                    if a not in Cmat.index:
                        missing.append(a)
                    if b not in Cmat.index:
                        missing.append(b)

                    cin["source_run"] = f"Q3D2Q_{sim_tag}_a{attempt}"
                    if not missing:
                        cin["method"] = "q3d_full_chip"
                    elif a in Cmat.index:
                        cin["method"] = "q3d_full_chip_fallback"
                        cin["full_chip_missing_nodes"] = missing
                    else:
                        cin["method"] = "q3d_full_chip_missing"
                        cin["full_chip_missing_nodes"] = missing
                    return cin

                def _cin_components_for_suffix(suffix: str) -> List[str]:
                    out: List[str] = []
                    for n in (
                        f"RO_TEE{suffix}",
                        f"RO_RES{suffix}",
                        f"RO_FEED{suffix}",
                        f"LP_RO{suffix}",
                        f"RO_FILTER_TEE{suffix}",
                        f"RO_FILTER_LINK{suffix}",
                        f"RO_PSTUB{suffix}",
                        f"RO_PSTUB_GND{suffix}",
                    ):
                        if n in design.components:
                            out.append(n)
                    return out

                def _refine_cin_if_missing_plate(
                    cin: TDict[str, Any],
                    *,
                    tee_name: str,
                    suffix: str,
                ) -> TDict[str, Any]:
                    # If cap_body_1_{tee} is missing from the main C-matrix, the exported terminals
                    # can get merged/aliased across multiple readouts. A small re-run restricted to
                    # the readout chain reliably restores a per-tee cap_body_1 node.
                    a = f"cap_body_0_{tee_name}"
                    b = f"cap_body_1_{tee_name}"
                    force_chain = os.environ.get("METAL_2Q_Q3D_FORCE_CHAIN_CIN", "").strip().lower() in {
                        "1",
                        "true",
                        "t",
                        "yes",
                        "y",
                        "on",
                    }
                    if a in Cmat.index and b in Cmat.index and not force_chain:
                        return cin

                    missing: List[str] = []
                    if a not in Cmat.index:
                        missing.append(a)
                    if b not in Cmat.index:
                        missing.append(b)

                    pair = str(cin.get("pair") or "")
                    res_node = str(cin.get("res_node") or "")
                    feed_node = str(cin.get("feed_node") or "")
                    other_tee = "RO_TEE2" if str(tee_name).endswith("1") else "RO_TEE1"

                    suspicious = bool(missing)
                    if other_tee in pair or other_tee in res_node or other_tee in feed_node:
                        suspicious = True
                    try:
                        if float(cin.get("Cin_fF") or 0.0) <= 0.0:
                            suspicious = True
                    except Exception:
                        suspicious = True
                    if not suspicious and not force_chain:
                        return cin

                    chain = _cin_components_for_suffix(suffix)
                    if not chain:
                        return cin

                    override_components = _components_override("METAL_2Q_Q3D_COMPONENTS")
                    if override_components is not None:
                        if not any(c in override_components for c in chain):
                            return cin
                        chain = [c for c in chain if c in override_components]
                        if not chain:
                            return cin

                    try:
                        if lom_sim is None:
                            return cin
                        run_name = f"Q3D2Q_{sim_tag}_a{attempt}_cin{suffix}"
                        print(
                            f"[Q3D] Refining Cin for {tee_name}: missing={missing}, components={chain}",
                            flush=True,
                        )
                        with _ansys_watchdog(q3d_watchdog_s, label=run_name):
                            lom_sim.run(name=run_name, components=chain, box_plus_buffer=False)
                        Cmat_local = lom_sim.capacitance_matrix
                    except Exception:
                        return cin

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
                        cin_local = base.pick_cin_prefer_tee_bodies(Cmat_local, tee_name=tee_name)
                        if a not in Cmat.index or b not in Cmat.index:
                            cin_local["method"] = "q3d_chain_refine"
                        else:
                            cin_local["method"] = "q3d_chain_only"
                        cin_local["source_run"] = run_name
                        cin_local["full_chip_missing_nodes"] = list(missing)
                        return cin_local

                    return cin

                # Cin per tee (with targeted refinement when one plate is missing)
                cin1 = _annotate_cin_full_chip(base.pick_cin_prefer_tee_bodies(Cmat, tee_name="RO_TEE1"), tee_name="RO_TEE1")
                cin2 = _annotate_cin_full_chip(base.pick_cin_prefer_tee_bodies(Cmat, tee_name="RO_TEE2"), tee_name="RO_TEE2")
                cin1 = _refine_cin_if_missing_plate(cin1, tee_name="RO_TEE1", suffix="1")
                cin2 = _refine_cin_if_missing_plate(cin2, tee_name="RO_TEE2", suffix="2")
                payload["q3d"]["external"] = {
                    "units": "fF",
                    "readout1": cin1,
                    "readout2": cin2,
                }

                ro1 = payload["resonators"]["readout1"]
                ro2 = payload["resonators"]["readout2"]
                ro1["external"].update(cin1)
                ro2["external"].update(cin2)

                if cin1.get("method") == "q3d_chain_refine":
                    ro1["warnings"].append(
                        "Cin refined via chain-only Q3D run="
                        f"{cin1.get('source_run')} missing_full_chip_nodes={cin1.get('full_chip_missing_nodes')}"
                    )
                if cin2.get("method") == "q3d_chain_refine":
                    ro2["warnings"].append(
                        "Cin refined via chain-only Q3D run="
                        f"{cin2.get('source_run')} missing_full_chip_nodes={cin2.get('full_chip_missing_nodes')}"
                    )

                # kappa_e estimates (needs fr)
                kappa_in_fn: Optional[Callable[[float, float, float], float]] = None
                try:
                    import importlib

                    _kc = importlib.import_module("qiskit_metal.analyses.em.kappa_calculation")
                    _fn = getattr(_kc, "kappa_in", None)
                    if callable(_fn):
                        kappa_in_fn = cast(Callable[[float, float, float], float], _fn)
                except Exception:
                    kappa_in_fn = None

                if kappa_in_fn is not None and cin1.get("Cin_fF") and ro1.get("f_GHz"):
                    fr_hz = float(ro1["f_GHz"]) * 1e9
                    Cin_F = float(cin1["Cin_fF"]) * 1e-15
                    kappa_e_hz = float(kappa_in_fn(fr_hz, Cin_F, fr_hz))
                    ro1["external"]["kappa_Hz"] = kappa_e_hz
                    ro1["external"]["kappa_over_2pi_Hz"] = kappa_e_hz / (2.0 * math.pi) if kappa_e_hz > 0 else None
                    ro1["external"]["Qe"] = fr_hz / kappa_e_hz if kappa_e_hz > 0 else None

                if kappa_in_fn is not None and cin2.get("Cin_fF") and ro2.get("f_GHz"):
                    fr_hz = float(ro2["f_GHz"]) * 1e9
                    Cin_F = float(cin2["Cin_fF"]) * 1e-15
                    kappa_e_hz = float(kappa_in_fn(fr_hz, Cin_F, fr_hz))
                    ro2["external"]["kappa_Hz"] = kappa_e_hz
                    ro2["external"]["kappa_over_2pi_Hz"] = kappa_e_hz / (2.0 * math.pi) if kappa_e_hz > 0 else None
                    ro2["external"]["Qe"] = fr_hz / kappa_e_hz if kappa_e_hz > 0 else None

                pf1 = _purcell_filter_meta_for_readout(layout_params, 1)
                pf2 = _purcell_filter_meta_for_readout(layout_params, 2)
                if pf1:
                    ro1["purcell_filter"] = pf1
                if pf2:
                    ro2["purcell_filter"] = pf2

                q3d_ok = True
                q3d_ok_final = True
                payload["meta"].pop("q3d_error", None)
                break

            except Exception as e:
                payload["meta"]["q3d_error"] = str(e)
                print(f"[Q3D] attempt={attempt} Error: {e}", flush=True)
                _print_ansys_messages(q3d, "Q3D", level=2)
                if _has_deterministic_geometry_error(q3d):
                    payload["meta"]["q3d_error"] = "deterministic_geometry_error"
                    break
                if attempt == 1:
                    try:
                        if lom_sim is not None:
                            lom_sim.close()
                        elif lom is not None:
                            lom.sim.close()
                    except Exception:
                        pass
                    base._ansys_best_effort_reset()
                    time.sleep(2.0)
                    gc.collect()
                    continue
            finally:
                try:
                    if lom_sim is not None:
                        lom_sim.close()
                    elif lom is not None:
                        lom.sim.close()
                except Exception:
                    pass

    payload["qubits"]["Q1"]["Lj_H"] = float(lj1_nh) * 1e-9
    payload["qubits"]["Q1"]["Cj_fF"] = float(cj1_fF)
    payload["qubits"]["Q1"]["C_eff_fF"] = Ceff1_fF
    payload["qubits"]["Q2"]["Lj_H"] = float(lj2_nh) * 1e-9
    payload["qubits"]["Q2"]["Cj_fF"] = float(cj2_fF)
    payload["qubits"]["Q2"]["C_eff_fF"] = Ceff2_fF

    _apply_external_coupling_overrides(
        payload,
        q1_effective_cin_fF=float(q1_effective_cin_fF),
        q2_effective_cin_fF=float(q2_effective_cin_fF),
        q1_target_kext_hz=float(q1_target_kext_hz),
        q2_target_kext_hz=float(q2_target_kext_hz),
    )

    payload = postprocess_2q(payload)
    hard_failures: List[str] = []
    if do_hfss and not hfss_ok_final:
        hard_failures.append(str(payload.get("meta", {}).get("hfss_error") or "HFSS/EPR did not complete"))
    if do_q3d and not q3d_ok_final:
        hard_failures.append(str(payload.get("meta", {}).get("q3d_error") or "Q3D did not complete"))
    if hard_failures:
        msg = "; ".join(hard_failures)
        payload["meta"]["analysis_error"] = msg
        payload["status"] = "failed"
        if save_json:
            json_dir = json_dir_out
            json_dir.mkdir(parents=True, exist_ok=True)
            json_path = json_dir / f"{sample_id}.json"
            try:
                if len(str(json_path.resolve())) >= 240:
                    short_id = base.short_tag(sample_id, max_len=140)
                    json_path = json_dir / f"{short_id}.json"
            except Exception:
                pass
            meta_val = payload.get("meta")
            if isinstance(meta_val, dict):
                meta_val["json_filename"] = json_path.name
                meta_val["json_subdir"] = json_subdir
                if json_path.stem != sample_id:
                    meta_val["json_filename_original"] = f"{sample_id}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(base.to_jsonable(payload), f, indent=2, ensure_ascii=False)
        raise RuntimeError("2Q analysis failed: " + msg)
    mode_invalid_reasons = _mode_picker_invalid_reasons(payload)
    if mode_invalid_reasons and not _env_bool("METAL_2Q_ALLOW_INVALID_MODE_RESULT", False):
        msg = "; ".join(str(x) for x in mode_invalid_reasons)
        payload["meta"]["mode_picker_error"] = msg
        payload["status"] = "failed"
        if save_json:
            json_dir = json_dir_out
            json_dir.mkdir(parents=True, exist_ok=True)
            json_path = json_dir / f"{sample_id}.json"
            try:
                if len(str(json_path.resolve())) >= 240:
                    short_id = base.short_tag(sample_id, max_len=140)
                    json_path = json_dir / f"{short_id}.json"
            except Exception:
                pass
            meta_val = payload.get("meta")
            if isinstance(meta_val, dict):
                meta_val["json_filename"] = json_path.name
                meta_val["json_subdir"] = json_subdir
                if json_path.stem != sample_id:
                    meta_val["json_filename_original"] = f"{sample_id}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(base.to_jsonable(payload), f, indent=2, ensure_ascii=False)
        raise RuntimeError("Invalid 2Q mode selection: " + msg)
    payload["status"] = "completed"

    if save_json:
        json_dir = json_dir_out
        json_dir.mkdir(parents=True, exist_ok=True)

        json_path = json_dir / f"{sample_id}.json"
        try:
            if len(str(json_path.resolve())) >= 240:
                short_id = base.short_tag(sample_id, max_len=140)
                json_path = json_dir / f"{short_id}.json"
        except Exception:
            pass

        meta_val = payload.get("meta")
        if isinstance(meta_val, dict):
            meta_val["json_filename"] = json_path.name
            meta_val["json_subdir"] = json_subdir
            if json_path.stem != sample_id:
                meta_val["json_filename_original"] = f"{sample_id}.json"

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(base.to_jsonable(payload), f, indent=2, ensure_ascii=False)

    return payload


# -------------------------
# main
# -------------------------
# Recommended stable 2Q baseline:
#
# export METAL_2Q_AUTOTUNE_TRIES=1
# export METAL_2Q_OPT_STAGE=baseline
#
# export METAL_2Q_LAYOUT_STYLE=official_singlepad
# export METAL_2Q_Q1_X_MM=-1.5
# export METAL_2Q_Q1_Y_MM=0.0
# export METAL_2Q_Q2_X_MM=1.5
# export METAL_2Q_Q2_Y_MM=0.0
#
# export METAL_2Q_HFSS_N_MODES=10
# export METAL_2Q_HFSS_MAX_PASSES=10
# export METAL_2Q_HFSS_MIN_PASSES=1
# export METAL_2Q_HFSS_MIN_CONVERGED=1
# export METAL_2Q_HFSS_MIN_FREQ_GHZ=1.0
# export METAL_2Q_HFSS_MAX_DELTA_F_GHZ=0.1
# export METAL_2Q_HFSS_CONV_MAX_DELTA_GHZ=0.02
# export METAL_2Q_HFSS_STRICT_CONVERGENCE=1
#
# export METAL_2Q_PICK_USE_PM=1
# export METAL_2Q_PICK_STRICT_PM=1
# export METAL_2Q_PICK_VALID_PM_MIN=0.25
# export METAL_2Q_PICK_STRICT_WINDOWS=1
#
# export METAL_2Q_PICK_Q_MIN_GHZ=4.0
# export METAL_2Q_PICK_Q_MAX_GHZ=5.3
# export METAL_2Q_PICK_R_MIN_GHZ=6.0
# export METAL_2Q_PICK_R_MAX_GHZ=7.8
#
if __name__ == "__main__":
    DEFAULT_SAMPLE_PREFIX = "sqchip_2q_default_qxsep0p0125_ro1stub14p33_ro2stub14p95_lj2_10p5_modes8_pass10_q3d"
    DEFAULT_Q1_X_MM = -0.00625
    DEFAULT_Q1_Y_MM = 1.5
    DEFAULT_Q2_X_MM = 0.00625
    DEFAULT_Q2_Y_MM = -1.5
    DEFAULT_RO1_L_MM = 16.90
    DEFAULT_RO2_L_MM = 15.00
    DEFAULT_RO1_TEE_CAP_GAP_UM = 80.0
    DEFAULT_RO2_TEE_CAP_GAP_UM = 100.0
    DEFAULT_RO1_TEE_CAP_WIDTH_UM = 0.28
    DEFAULT_RO2_TEE_CAP_WIDTH_UM = 0.22
    DEFAULT_RO1_TEE_CAP_DISTANCE_UM = 170.0
    DEFAULT_RO2_TEE_CAP_DISTANCE_UM = 170.0
    DEFAULT_RO1_TEE_FINGER_LENGTH_UM = 10
    DEFAULT_RO2_TEE_FINGER_LENGTH_UM = 10
    DEFAULT_RO1_TEE_FINGER_COUNT = 1
    DEFAULT_RO2_TEE_FINGER_COUNT = 1
    DEFAULT_FEED_L_MM = 3.0
    DEFAULT_FEED_SPACING_UM = 1100.0
    DEFAULT_RO_SPACING_UM = 260.0
    DEFAULT_FEED_LEAD_START_UM = 50.0
    DEFAULT_FEED_LEAD_END_UM = 50.0
    DEFAULT_SWAP_TEE_PORTS = True
    DEFAULT_RO1_DX_MM = 3.20
    DEFAULT_RO1_DY_MM = -0.35
    DEFAULT_RO2_DX_MM = 3.20
    DEFAULT_RO2_DY_MM = 0.35
    DEFAULT_RO1_PURCELL_STUB_LENGTH_MM = 14.33
    DEFAULT_RO2_PURCELL_STUB_LENGTH_MM = 14.95
    DEFAULT_RO1_PURCELL_STUB_OFFSET_MM = 2.8
    DEFAULT_RO2_PURCELL_STUB_OFFSET_MM = 2.8
    DEFAULT_RO1_PURCELL_STUB_WIDTH_UM = 10.0
    DEFAULT_RO2_PURCELL_STUB_WIDTH_UM = 10.0
    DEFAULT_RO1_PURCELL_STUB_GAP_UM = 6.0
    DEFAULT_RO2_PURCELL_STUB_GAP_UM = 6.0
    DEFAULT_LJ1_NH = 10.0
    DEFAULT_CJ1_FF = 0.0
    DEFAULT_LJ2_NH = 10.5
    DEFAULT_CJ2_FF = 0.0
    DEFAULT_Q1_POCKET_HEIGHT_UM = 650.0
    DEFAULT_Q2_POCKET_HEIGHT_UM = 650.0
    DEFAULT_Q1_RO_PAD_W_UM = 420.0
    DEFAULT_Q1_RO_PAD_H_UM = 420.0
    DEFAULT_Q1_RO_PAD_GAP_UM = 4.0
    DEFAULT_Q2_RO_PAD_W_UM = 400.0
    DEFAULT_Q2_RO_PAD_H_UM = 400.0
    DEFAULT_Q2_RO_PAD_GAP_UM = 4.0

    os.environ.setdefault("METAL_2Q_SAMPLE_ID", f"{DEFAULT_SAMPLE_PREFIX}_{time.strftime('%Y%m%d_%H%M%S')}")
    os.environ.setdefault("METAL_2Q_AUTOTUNE_TRIES", "1")
    os.environ.setdefault("METAL_2Q_OPT_STAGE", "baseline")
    os.environ.setdefault("METAL_2Q_BATCH_POSITION_SWEEP", "1")
    os.environ.setdefault("METAL_2Q_BATCH_SIZE", "2")
    os.environ.setdefault("METAL_2Q_BATCH_TOTAL", "60")
    os.environ.setdefault("METAL_2Q_JSON_SUBDIR", "json_tmp")
    os.environ.setdefault("METAL_2Q_GDS_SUBDIR", "gds_tmp")
    os.environ.setdefault("METAL_2Q_DO_HFSS", "1")
    os.environ.setdefault("METAL_2Q_DO_Q3D", "1")
    os.environ.setdefault("METAL_2Q_CHIP_SIZE_X_MM", "12.0")
    os.environ.setdefault("METAL_2Q_CHIP_SIZE_Y_MM", "12.0")
    os.environ.setdefault("METAL_2Q_PICK_Q_MIN_GHZ", "3.0")
    os.environ.setdefault("METAL_2Q_PICK_Q_MAX_GHZ", "6.2")
    os.environ.setdefault("METAL_2Q_PICK_R_MIN_GHZ", "5.0")
    os.environ.setdefault("METAL_2Q_PICK_R_MAX_GHZ", "8.0")
    os.environ.setdefault("METAL_2Q_PICK_USE_PM", "1")
    os.environ.setdefault("METAL_2Q_FOCUS_PAIR", "2")
    os.environ.setdefault("METAL_2Q_PICK_FOCUS_REQUIRE_IN_WINDOW", "0")
    os.environ.setdefault("METAL_2Q_PICK_STRICT_PM", "0")
    os.environ.setdefault("METAL_2Q_PICK_STRICT_WINDOWS", "1")
    os.environ.setdefault("METAL_2Q_PICK_VALID_PM_MIN", "0.30")
    os.environ.setdefault("METAL_2Q_PICK_Q_KERR_MIN_MHZ", "0.01")
    os.environ.setdefault("METAL_2Q_PICK_Q_MAX_DIST_GHZ", "3.5")
    os.environ.setdefault("METAL_2Q_PICK_R_PM_MAX", "0.12")
    os.environ.setdefault("METAL_2Q_PICK_R_KERR_MAX_MHZ", "1.0")
    os.environ.setdefault("METAL_2Q_HFSS_N_MODES", "8")
    os.environ.setdefault("METAL_2Q_HFSS_MAX_PASSES", "10")
    os.environ.setdefault("METAL_2Q_HFSS_MIN_PASSES", "10")
    os.environ.setdefault("METAL_2Q_HFSS_MIN_CONVERGED", "2")
    os.environ.setdefault("METAL_2Q_HFSS_MIN_FREQ_GHZ", "3.5")
    os.environ.setdefault("METAL_2Q_HFSS_MAX_DELTA_F_GHZ", "0.2")
    os.environ.setdefault("METAL_2Q_HFSS_PCT_REFINEMENT", "30")
    os.environ.setdefault("METAL_2Q_HFSS_MAX_MESH_LENGTH_JJ", "7um")
    os.environ.setdefault("METAL_2Q_HFSS_MAX_MESH_LENGTH_PORT", "7um")
    os.environ.setdefault("METAL_2Q_HFSS_STRICT_CONVERGENCE", "0")
    os.environ.setdefault("METAL_2Q_ALLOW_INVALID_MODE_RESULT", "1")
    os.environ.setdefault("METAL_2Q_ANSYS_RESET_ON_RETRY", "1")
    os.environ.setdefault("METAL_2Q_ANSYS_REUSE_PROJECT_FILE", "1")
    os.environ.setdefault("METAL_2Q_ANSYS_PROJECT_FILE", DEFAULT_2Q_ANSYS_PROJECT_FILE)
    os.environ.setdefault("METAL_2Q_Q3D_FORCE_CHAIN_CIN", "1")
    os.environ.setdefault("METAL_2Q_Q3D_CONNECTOR_PAD_CUTOUTS", "1")
    os.environ.setdefault("METAL_2Q_USE_QI_FALLBACK", "1")
    os.environ.setdefault("METAL_2Q_QI_FALLBACK", "4000000")
    os.environ.setdefault("METAL_2Q_Q1_BUS_PAD_GAP_UM", "3.0")
    os.environ.setdefault("METAL_2Q_Q2_BUS_PAD_GAP_UM", "3.0")
    os.environ.setdefault("METAL_2Q_Q1_BUS_PAD_W_UM", "270.0")
    os.environ.setdefault("METAL_2Q_Q2_BUS_PAD_W_UM", "270.0")
    os.environ.setdefault("METAL_2Q_Q1_BUS_PAD_H_UM", "270.0")
    os.environ.setdefault("METAL_2Q_Q2_BUS_PAD_H_UM", "270.0")
    os.environ.setdefault("METAL_2Q_BUS_TOTAL_LENGTH_MM", "6.0")
    os.environ.setdefault("METAL_2Q_LP1_DX_MM", "1.2")
    os.environ.setdefault("METAL_2Q_LP2_DX_MM", "1.2")
    os.environ.setdefault("METAL_2Q_TARGET_T1_US", "50")
    os.environ.setdefault("METAL_2Q_CHI_T1_MIN_US", "50")
    os.environ.setdefault("Q1_TARGET_KEXT_HZ", "0")
    os.environ.setdefault("Q2_TARGET_KEXT_HZ", "0")

    def _env_float_fallback(primary: str, fallback: str, default: float, *, lo: float, hi: float) -> float:
        if os.environ.get(primary, "").strip():
            return _env_float(primary, default, lo=lo, hi=hi)
        return _env_float(fallback, default, lo=lo, hi=hi)

    def _env_int_fallback(primary: str, fallback: str, default: int, *, lo: int, hi: int) -> int:
        if os.environ.get(primary, "").strip():
            return _env_int(primary, default, lo=lo, hi=hi)
        return _env_int(fallback, default, lo=lo, hi=hi)

    def _env_bool_fallback(primary: str, fallback: str, default: bool = False) -> bool:
        if os.environ.get(primary, "").strip():
            return _env_bool(primary, default)
        return _env_bool(fallback, default)

    dataset_root = Path(os.environ.get("METAL_2Q_OUT_DIR", "./data/sqchip_em_2q"))
    dataset_root.mkdir(parents=True, exist_ok=True)
    (dataset_root / "json_tmp").mkdir(parents=True, exist_ok=True)
    (dataset_root / "gds_tmp").mkdir(parents=True, exist_ok=True)

    # Keep the standalone script defaults aligned with scripts/strict_2q_optimize.py.
    Q1_X_MM = _env_float_fallback("Q1_X_MM", "METAL_2Q_Q1_X_MM", DEFAULT_Q1_X_MM, lo=-10.0, hi=10.0)
    Q1_Y_MM = _env_float_fallback("Q1_Y_MM", "METAL_2Q_Q1_Y_MM", DEFAULT_Q1_Y_MM, lo=-10.0, hi=10.0)
    Q2_X_MM = _env_float_fallback("Q2_X_MM", "METAL_2Q_Q2_X_MM", DEFAULT_Q2_X_MM, lo=-10.0, hi=10.0)
    Q2_Y_MM = _env_float_fallback("Q2_Y_MM", "METAL_2Q_Q2_Y_MM", DEFAULT_Q2_Y_MM, lo=-10.0, hi=10.0)

    LJ1_NH = _env_float_fallback("LJ1_NH", "METAL_2Q_LJ1_NH", DEFAULT_LJ1_NH, lo=0.1, hi=100.0)
    CJ1_FF = _env_float_fallback("CJ1_FF", "METAL_2Q_CJ1_FF", DEFAULT_CJ1_FF, lo=0.0, hi=100.0)
    LJ2_NH = _env_float_fallback("LJ2_NH", "METAL_2Q_LJ2_NH", DEFAULT_LJ2_NH, lo=0.1, hi=100.0)
    CJ2_FF = _env_float_fallback("CJ2_FF", "METAL_2Q_CJ2_FF", DEFAULT_CJ2_FF, lo=0.0, hi=100.0)

    CHIP_SIZE_X_MM = _env_float_fallback("CHIP_SIZE_X_MM", "METAL_2Q_CHIP_SIZE_X_MM", 12.0, lo=1.0, hi=100.0)
    CHIP_SIZE_Y_MM = _env_float_fallback("CHIP_SIZE_Y_MM", "METAL_2Q_CHIP_SIZE_Y_MM", 12.0, lo=1.0, hi=100.0)
    CPW_WIDTH_UM = _env_float_fallback("CPW_WIDTH_UM", "METAL_2Q_CPW_WIDTH_UM", 10.0, lo=0.1, hi=500.0)
    CPW_GAP_UM = _env_float_fallback("CPW_GAP_UM", "METAL_2Q_CPW_GAP_UM", 6.0, lo=0.1, hi=500.0)
    FEED_L_MM = _env_float_fallback("FEED_L_MM", "METAL_2Q_FEED_L_MM", DEFAULT_FEED_L_MM, lo=0.1, hi=50.0)
    FEED_FILLET_UM = _env_float_fallback("FEED_FILLET_UM", "METAL_2Q_FEED_FILLET_UM", 25.0, lo=0.0, hi=1000.0)
    RO_FILLET_UM = _env_float_fallback("RO_FILLET_UM", "METAL_2Q_RO_FILLET_UM", 25.0, lo=0.0, hi=1000.0)
    FEED_SPACING_UM = _env_float_fallback("FEED_SPACING_UM", "METAL_2Q_FEED_SPACING_UM", DEFAULT_FEED_SPACING_UM, lo=1.0, hi=5000.0)
    RO_SPACING_UM = _env_float_fallback("RO_SPACING_UM", "METAL_2Q_RO_SPACING_UM", DEFAULT_RO_SPACING_UM, lo=1.0, hi=5000.0)
    FEED_LEAD_START_UM = _env_float_fallback("FEED_LEAD_START_UM", "METAL_2Q_FEED_LEAD_START_UM", DEFAULT_FEED_LEAD_START_UM, lo=0.0, hi=5000.0)
    FEED_LEAD_END_UM = _env_float_fallback("FEED_LEAD_END_UM", "METAL_2Q_FEED_LEAD_END_UM", DEFAULT_FEED_LEAD_END_UM, lo=0.0, hi=5000.0)
    RO_LEAD_START_UM = _env_float_fallback("RO_LEAD_START_UM", "METAL_2Q_RO_LEAD_START_UM", 250.0, lo=0.0, hi=5000.0)
    RO_LEAD_END_UM = _env_float_fallback("RO_LEAD_END_UM", "METAL_2Q_RO_LEAD_END_UM", 250.0, lo=0.0, hi=5000.0)
    PRIME_WIDTH_UM = _env_float_fallback("PRIME_WIDTH_UM", "METAL_2Q_PRIME_WIDTH_UM", 14.0, lo=0.1, hi=500.0)
    PRIME_GAP_UM = _env_float_fallback("PRIME_GAP_UM", "METAL_2Q_PRIME_GAP_UM", 6.0, lo=0.1, hi=500.0)
    SECOND_WIDTH_UM = _env_float_fallback("SECOND_WIDTH_UM", "METAL_2Q_SECOND_WIDTH_UM", 10.0, lo=0.1, hi=500.0)
    SECOND_GAP_UM = _env_float_fallback("SECOND_GAP_UM", "METAL_2Q_SECOND_GAP_UM", 6.0, lo=0.1, hi=500.0)

    RO_L_MM = _env_float_fallback("RO_L_MM", "METAL_2Q_RO_L_MM", 10.2, lo=3.0, hi=20.0)
    Q_RO_PAD_GAP_UM = _env_float_fallback("Q_RO_PAD_GAP_UM", "METAL_2Q_Q_RO_PAD_GAP_UM", 1.0, lo=0.2, hi=30.0)

    TEE_FINGER_LENGTH_UM = _env_int_fallback("TEE_FINGER_LENGTH_UM", "METAL_2Q_TEE_FINGER_LENGTH_UM", 160, lo=10, hi=500)
    TEE_FINGER_COUNT = _env_int_fallback("TEE_FINGER_COUNT", "METAL_2Q_TEE_FINGER_COUNT", 12, lo=1, hi=100)
    TEE_CAP_GAP_UM = _env_float_fallback("TEE_CAP_GAP_UM", "METAL_2Q_TEE_CAP_GAP_UM", 1.2, lo=0.2, hi=200.0)
    TEE_CAP_WIDTH_UM = _env_float_fallback("TEE_CAP_WIDTH_UM", "METAL_2Q_TEE_CAP_WIDTH_UM", 18.0, lo=0.2, hi=500.0)
    TEE_CAP_DISTANCE_UM = _env_float_fallback("TEE_CAP_DISTANCE_UM", "METAL_2Q_TEE_CAP_DISTANCE_UM", 40.0, lo=1.0, hi=500.0)
    PURCELL_STUB_LENGTH_MM = _env_float_fallback("PURCELL_STUB_LENGTH_MM", "METAL_2Q_PURCELL_STUB_LENGTH_MM", 0.0, lo=0.0, hi=50.0)
    PURCELL_STUB_OFFSET_MM = _env_float_fallback("PURCELL_STUB_OFFSET_MM", "METAL_2Q_PURCELL_STUB_OFFSET_MM", 1.2, lo=0.1, hi=20.0)
    PURCELL_STUB_WIDTH_UM = _env_float_fallback("PURCELL_STUB_WIDTH_UM", "METAL_2Q_PURCELL_STUB_WIDTH_UM", 10.0, lo=0.1, hi=500.0)
    PURCELL_STUB_GAP_UM = _env_float_fallback("PURCELL_STUB_GAP_UM", "METAL_2Q_PURCELL_STUB_GAP_UM", 6.0, lo=0.1, hi=500.0)
    TAN_DELTA_MAIN = _env_float_fallback(
        "METAL_2Q_TAN_DELTA_MAIN",
        "METAL_TAN_DELTA_MAIN",
        1e-6,
        lo=1e-7,
        hi=1e-1,
    )
    base.TAN_DELTA_MAIN = float(TAN_DELTA_MAIN)
    base.QI_FALLBACK = _env_float_fallback(
        "METAL_2Q_QI_FALLBACK",
        "METAL_QI_FALLBACK",
        float(base.QI_FALLBACK),
        lo=1.0,
        hi=1.0e12,
    )
    base.USE_QI_FALLBACK = _env_bool_fallback(
        "METAL_2Q_USE_QI_FALLBACK",
        "METAL_USE_QI_FALLBACK",
        False,
    )
    Q1_EFFECTIVE_CIN_FF = _env_float_fallback("Q1_EFFECTIVE_CIN_FF", "METAL_2Q_Q1_EFFECTIVE_CIN_FF", 0.0, lo=0.0, hi=1.0e5)
    Q2_EFFECTIVE_CIN_FF = _env_float_fallback("Q2_EFFECTIVE_CIN_FF", "METAL_2Q_Q2_EFFECTIVE_CIN_FF", 0.0, lo=0.0, hi=1.0e5)
    Q1_TARGET_KEXT_HZ = _env_float_fallback("Q1_TARGET_KEXT_HZ", "METAL_2Q_Q1_TARGET_KEXT_HZ", 0.0, lo=0.0, hi=1.0e10)
    Q2_TARGET_KEXT_HZ = _env_float_fallback("Q2_TARGET_KEXT_HZ", "METAL_2Q_Q2_TARGET_KEXT_HZ", 0.0, lo=0.0, hi=1.0e10)

    separation_mm = math.sqrt((Q2_X_MM - Q1_X_MM) ** 2 + (Q2_Y_MM - Q1_Y_MM) ** 2)
    bus_length_mm = _env_float_fallback(
        "BUS_TOTAL_LENGTH_MM",
        "METAL_2Q_BUS_TOTAL_LENGTH_MM",
        max(5.0, separation_mm + 2.0),
        lo=0.1,
        hi=50.0,
    )
    BUS_WIDTH_UM = _env_float_fallback("BUS_WIDTH_UM", "METAL_2Q_BUS_WIDTH_UM", 10.0, lo=0.1, hi=500.0)
    BUS_GAP_UM = _env_float_fallback("BUS_GAP_UM", "METAL_2Q_BUS_GAP_UM", 6.0, lo=0.1, hi=500.0)
    BUS_LEAD_START_UM = _env_float_fallback("BUS_LEAD_START_UM", "METAL_2Q_BUS_LEAD_START_UM", 250.0, lo=0.0, hi=5000.0)
    BUS_LEAD_END_UM = _env_float_fallback("BUS_LEAD_END_UM", "METAL_2Q_BUS_LEAD_END_UM", 250.0, lo=0.0, hi=5000.0)
    BUS_SPACING_UM = _env_float_fallback("BUS_SPACING_UM", "METAL_2Q_BUS_SPACING_UM", 250.0, lo=1.0, hi=5000.0)

    lp_shared = base.LayoutParams(
        chip_size_x=f"{CHIP_SIZE_X_MM}mm",
        chip_size_y=f"{CHIP_SIZE_Y_MM}mm",
        cpw_width=f"{CPW_WIDTH_UM}um",
        cpw_gap=f"{CPW_GAP_UM}um",
        feed_fillet=f"{FEED_FILLET_UM}um",
        ro_fillet=f"{RO_FILLET_UM}um",
        ro_pad_w="360um",
        ro_pad_h="360um",
        ro_pad_gap=f"{Q_RO_PAD_GAP_UM}um",
        ro_total_length=f"{RO_L_MM}mm",
        prime_width=f"{PRIME_WIDTH_UM}um",
        prime_gap=f"{PRIME_GAP_UM}um",
        second_width=f"{SECOND_WIDTH_UM}um",
        second_gap=f"{SECOND_GAP_UM}um",
        finger_length=f"{TEE_FINGER_LENGTH_UM}um",
        finger_count=str(int(TEE_FINGER_COUNT)),
        cap_gap=f"{TEE_CAP_GAP_UM}um",
        cap_width=f"{TEE_CAP_WIDTH_UM}um",
        cap_distance=f"{TEE_CAP_DISTANCE_UM}um",
        feed_total_length=f"{FEED_L_MM}mm",
        feed_spacing=f"{FEED_SPACING_UM}um",
        feed_lead_start=f"{FEED_LEAD_START_UM}um",
        feed_lead_end=f"{FEED_LEAD_END_UM}um",
        ro_spacing=f"{RO_SPACING_UM}um",
        ro_lead_start=f"{RO_LEAD_START_UM}um",
        ro_lead_end=f"{RO_LEAD_END_UM}um",
    )

    p2 = Layout2QParams(
        shared=lp_shared,
        tee_offset_mm=_env_float_fallback("TEE_OFFSET_MM", "METAL_2Q_TEE_OFFSET_MM", 0.8, lo=0.1, hi=10.0),
        ro1_total_length_mm=_env_float_fallback("RO1_L_MM", "METAL_2Q_RO1_L_MM", DEFAULT_RO1_L_MM, lo=3.0, hi=20.0),
        ro2_total_length_mm=_env_float_fallback("RO2_L_MM", "METAL_2Q_RO2_L_MM", DEFAULT_RO2_L_MM, lo=3.0, hi=20.0),
        ro1_tee_cap_gap_um=_env_float_fallback("RO1_TEE_CAP_GAP_UM", "METAL_2Q_RO1_TEE_CAP_GAP_UM", DEFAULT_RO1_TEE_CAP_GAP_UM, lo=0.2, hi=200.0),
        ro2_tee_cap_gap_um=_env_float_fallback("RO2_TEE_CAP_GAP_UM", "METAL_2Q_RO2_TEE_CAP_GAP_UM", DEFAULT_RO2_TEE_CAP_GAP_UM, lo=0.2, hi=200.0),
        ro1_tee_cap_width_um=_env_float_fallback("RO1_TEE_CAP_WIDTH_UM", "METAL_2Q_RO1_TEE_CAP_WIDTH_UM", DEFAULT_RO1_TEE_CAP_WIDTH_UM, lo=0.2, hi=500.0),
        ro2_tee_cap_width_um=_env_float_fallback("RO2_TEE_CAP_WIDTH_UM", "METAL_2Q_RO2_TEE_CAP_WIDTH_UM", DEFAULT_RO2_TEE_CAP_WIDTH_UM, lo=0.2, hi=500.0),
        ro1_tee_cap_distance_um=_env_float_fallback("RO1_TEE_CAP_DISTANCE_UM", "METAL_2Q_RO1_TEE_CAP_DISTANCE_UM", DEFAULT_RO1_TEE_CAP_DISTANCE_UM, lo=1.0, hi=500.0),
        ro2_tee_cap_distance_um=_env_float_fallback("RO2_TEE_CAP_DISTANCE_UM", "METAL_2Q_RO2_TEE_CAP_DISTANCE_UM", DEFAULT_RO2_TEE_CAP_DISTANCE_UM, lo=1.0, hi=500.0),
        ro1_tee_finger_length_um=_env_int_fallback("RO1_TEE_FINGER_LENGTH_UM", "METAL_2Q_RO1_TEE_FINGER_LENGTH_UM", DEFAULT_RO1_TEE_FINGER_LENGTH_UM, lo=0, hi=500),
        ro2_tee_finger_length_um=_env_int_fallback("RO2_TEE_FINGER_LENGTH_UM", "METAL_2Q_RO2_TEE_FINGER_LENGTH_UM", DEFAULT_RO2_TEE_FINGER_LENGTH_UM, lo=0, hi=500),
        ro1_tee_finger_count=_env_int_fallback("RO1_TEE_FINGER_COUNT", "METAL_2Q_RO1_TEE_FINGER_COUNT", DEFAULT_RO1_TEE_FINGER_COUNT, lo=1, hi=100),
        ro2_tee_finger_count=_env_int_fallback("RO2_TEE_FINGER_COUNT", "METAL_2Q_RO2_TEE_FINGER_COUNT", DEFAULT_RO2_TEE_FINGER_COUNT, lo=1, hi=100),
        ro1_purcell_stub_length_mm=_env_float_fallback("RO1_PURCELL_STUB_LENGTH_MM", "METAL_2Q_RO1_PURCELL_STUB_LENGTH_MM", DEFAULT_RO1_PURCELL_STUB_LENGTH_MM, lo=0.0, hi=50.0),
        ro2_purcell_stub_length_mm=_env_float_fallback("RO2_PURCELL_STUB_LENGTH_MM", "METAL_2Q_RO2_PURCELL_STUB_LENGTH_MM", DEFAULT_RO2_PURCELL_STUB_LENGTH_MM, lo=0.0, hi=50.0),
        ro1_purcell_stub_offset_mm=_env_float_fallback("RO1_PURCELL_STUB_OFFSET_MM", "METAL_2Q_RO1_PURCELL_STUB_OFFSET_MM", DEFAULT_RO1_PURCELL_STUB_OFFSET_MM, lo=0.1, hi=20.0),
        ro2_purcell_stub_offset_mm=_env_float_fallback("RO2_PURCELL_STUB_OFFSET_MM", "METAL_2Q_RO2_PURCELL_STUB_OFFSET_MM", DEFAULT_RO2_PURCELL_STUB_OFFSET_MM, lo=0.1, hi=20.0),
        ro1_purcell_stub_width_um=_env_float_fallback("RO1_PURCELL_STUB_WIDTH_UM", "METAL_2Q_RO1_PURCELL_STUB_WIDTH_UM", DEFAULT_RO1_PURCELL_STUB_WIDTH_UM, lo=0.1, hi=500.0),
        ro2_purcell_stub_width_um=_env_float_fallback("RO2_PURCELL_STUB_WIDTH_UM", "METAL_2Q_RO2_PURCELL_STUB_WIDTH_UM", DEFAULT_RO2_PURCELL_STUB_WIDTH_UM, lo=0.1, hi=500.0),
        ro1_purcell_stub_gap_um=_env_float_fallback("RO1_PURCELL_STUB_GAP_UM", "METAL_2Q_RO1_PURCELL_STUB_GAP_UM", DEFAULT_RO1_PURCELL_STUB_GAP_UM, lo=0.1, hi=500.0),
        ro2_purcell_stub_gap_um=_env_float_fallback("RO2_PURCELL_STUB_GAP_UM", "METAL_2Q_RO2_PURCELL_STUB_GAP_UM", DEFAULT_RO2_PURCELL_STUB_GAP_UM, lo=0.1, hi=500.0),
        ro1_dx_mm=_env_float_fallback("RO1_DX_MM", "METAL_2Q_RO1_DX_MM", DEFAULT_RO1_DX_MM, lo=-10.0, hi=10.0),
        ro1_dy_mm=_env_float_fallback("RO1_DY_MM", "METAL_2Q_RO1_DY_MM", DEFAULT_RO1_DY_MM, lo=-10.0, hi=10.0),
        ro2_dx_mm=_env_float_fallback("RO2_DX_MM", "METAL_2Q_RO2_DX_MM", DEFAULT_RO2_DX_MM, lo=-10.0, hi=10.0),
        ro2_dy_mm=_env_float_fallback("RO2_DY_MM", "METAL_2Q_RO2_DY_MM", DEFAULT_RO2_DY_MM, lo=-10.0, hi=10.0),
        lp1_dx_mm=_env_float_fallback("LP1_DX_MM", "METAL_2Q_LP1_DX_MM", 2.4, lo=0.1, hi=20.0),
        lp2_dx_mm=_env_float_fallback("LP2_DX_MM", "METAL_2Q_LP2_DX_MM", 2.4, lo=0.1, hi=20.0),
        swap_tee_ports=_env_bool_fallback("SWAP_TEE_PORTS", "METAL_2Q_SWAP_TEE_PORTS", DEFAULT_SWAP_TEE_PORTS),
        # 425um / 650um track the official TransmonPocket examples more closely.
        # Older engineering defaults such as 520um / 780um still work via env overrides.
        q1_pad_width_um=_env_float_fallback("Q1_PAD_WIDTH_UM", "METAL_2Q_Q1_PAD_WIDTH_UM", 425.0, lo=100.0, hi=2000.0),
        q2_pad_width_um=_env_float_fallback("Q2_PAD_WIDTH_UM", "METAL_2Q_Q2_PAD_WIDTH_UM", 425.0, lo=100.0, hi=2000.0),
        q1_pocket_height_um=_env_float_fallback("Q1_POCKET_HEIGHT_UM", "METAL_2Q_Q1_POCKET_HEIGHT_UM", DEFAULT_Q1_POCKET_HEIGHT_UM, lo=100.0, hi=3000.0),
        q2_pocket_height_um=_env_float_fallback("Q2_POCKET_HEIGHT_UM", "METAL_2Q_Q2_POCKET_HEIGHT_UM", DEFAULT_Q2_POCKET_HEIGHT_UM, lo=100.0, hi=3000.0),
        q1_ro_pad_w_um=_env_float_fallback("Q1_RO_PAD_W_UM", "METAL_2Q_Q1_RO_PAD_W_UM", DEFAULT_Q1_RO_PAD_W_UM, lo=20.0, hi=2000.0),
        q2_ro_pad_w_um=_env_float_fallback("Q2_RO_PAD_W_UM", "METAL_2Q_Q2_RO_PAD_W_UM", DEFAULT_Q2_RO_PAD_W_UM, lo=20.0, hi=2000.0),
        q1_ro_pad_h_um=_env_float_fallback("Q1_RO_PAD_H_UM", "METAL_2Q_Q1_RO_PAD_H_UM", DEFAULT_Q1_RO_PAD_H_UM, lo=20.0, hi=2000.0),
        q2_ro_pad_h_um=_env_float_fallback("Q2_RO_PAD_H_UM", "METAL_2Q_Q2_RO_PAD_H_UM", DEFAULT_Q2_RO_PAD_H_UM, lo=20.0, hi=2000.0),
        q1_ro_pad_gap_um=_env_float_fallback("Q1_RO_PAD_GAP_UM", "METAL_2Q_Q1_RO_PAD_GAP_UM", DEFAULT_Q1_RO_PAD_GAP_UM, lo=0.2, hi=200.0),
        q2_ro_pad_gap_um=_env_float_fallback("Q2_RO_PAD_GAP_UM", "METAL_2Q_Q2_RO_PAD_GAP_UM", DEFAULT_Q2_RO_PAD_GAP_UM, lo=0.2, hi=200.0),
        q1_bus_pad_w_um=_env_float_fallback("Q1_BUS_PAD_W_UM", "METAL_2Q_Q1_BUS_PAD_W_UM", 270.0, lo=20.0, hi=2000.0),
        q2_bus_pad_w_um=_env_float_fallback("Q2_BUS_PAD_W_UM", "METAL_2Q_Q2_BUS_PAD_W_UM", 270.0, lo=20.0, hi=2000.0),
        q1_bus_pad_h_um=_env_float_fallback("Q1_BUS_PAD_H_UM", "METAL_2Q_Q1_BUS_PAD_H_UM", 270.0, lo=20.0, hi=2000.0),
        q2_bus_pad_h_um=_env_float_fallback("Q2_BUS_PAD_H_UM", "METAL_2Q_Q2_BUS_PAD_H_UM", 270.0, lo=20.0, hi=2000.0),
        q1_bus_pad_gap_um=_env_float_fallback("Q1_BUS_PAD_GAP_UM", "METAL_2Q_Q1_BUS_PAD_GAP_UM", 1.2, lo=0.2, hi=200.0),
        q2_bus_pad_gap_um=_env_float_fallback("Q2_BUS_PAD_GAP_UM", "METAL_2Q_Q2_BUS_PAD_GAP_UM", 1.2, lo=0.2, hi=200.0),
        bus_total_length=f"{bus_length_mm:.2f}mm",
        bus_width=f"{BUS_WIDTH_UM}um",
        bus_gap=f"{BUS_GAP_UM}um",
        bus_lead_start=f"{BUS_LEAD_START_UM}um",
        bus_lead_end=f"{BUS_LEAD_END_UM}um",
        bus_fillet="50um",
        bus_spacing=f"{BUS_SPACING_UM}um",
    )

    autotune_tries = _env_int_fallback("METAL_2Q_AUTOTUNE_TRIES", "AUTOTUNE_TRIES", 1, lo=1, hi=10_000)
    autotune_seed = _env_int_fallback("METAL_2Q_AUTOTUNE_SEED", "AUTOTUNE_SEED", 0, lo=0, hi=2_000_000_000)
    do_hfss = _env_bool_fallback("METAL_2Q_DO_HFSS", "DO_HFSS", True)
    do_q3d = _env_bool_fallback("METAL_2Q_DO_Q3D", "DO_Q3D", True)
    opt_stage = (os.environ.get("METAL_2Q_OPT_STAGE", "baseline").strip().lower() or "baseline")
    stage_plan_map: TDict[str, List[str]] = {
        "baseline": [],
        "fq": ["fq"],
        "fr": ["fq", "fr"],
        "chi": ["fq", "fr", "chi"],
        "t1": ["fq", "fr", "chi", "t1"],
        "pipeline": ["fq", "fr", "chi", "t1"],
        "full": ["fq", "fr", "chi", "t1", "full"],
    }
    if opt_stage not in stage_plan_map:
        raise ValueError(
            "METAL_2Q_OPT_STAGE must be one of: baseline, fq, fr, chi, t1, pipeline, full"
        )
    allow_bus_pad_gap_opt = _env_bool("METAL_2Q_OPT_ALLOW_BUS_PAD_GAP", False)

    def _env_float_optional(name: str, *, lo: float, hi: float) -> Optional[float]:
        raw = os.environ.get(name, "").strip()
        if not raw:
            return None
        return _env_float(name, 0.0, lo=lo, hi=hi)

    def _env_specific_or_shared_float(
        specific_name: str,
        shared_name: str,
        default: float,
        *,
        lo: float,
        hi: float,
    ) -> float:
        specific = _env_float_optional(specific_name, lo=lo, hi=hi)
        if specific is not None:
            return float(specific)
        shared = _env_float_optional(shared_name, lo=lo, hi=hi)
        if shared is not None:
            return float(shared)
        return float(default)

    q_pick_min = _env_float("METAL_2Q_PICK_Q_MIN_GHZ", 4.0, lo=0.1, hi=20.0)
    q_pick_max = _env_float("METAL_2Q_PICK_Q_MAX_GHZ", 6.0, lo=0.1, hi=20.0)
    if q_pick_min > q_pick_max:
        q_pick_min, q_pick_max = q_pick_max, q_pick_min
    r_pick_min = _env_float("METAL_2Q_PICK_R_MIN_GHZ", 5.5, lo=0.1, hi=20.0)
    r_pick_max = _env_float("METAL_2Q_PICK_R_MAX_GHZ", 7.5, lo=0.1, hi=20.0)
    if r_pick_min > r_pick_max:
        r_pick_min, r_pick_max = r_pick_max, r_pick_min

    fq_target_default = 4.5
    fr_target_default = 0.5 * (r_pick_min + r_pick_max)
    chi_target_default = 2.0
    fq_score_min_ghz = _env_float("METAL_2Q_FQ_SCORE_MIN_GHZ", 4.0, lo=0.1, hi=20.0)
    fq_score_max_ghz = _env_float("METAL_2Q_FQ_SCORE_MAX_GHZ", 5.0, lo=0.1, hi=20.0)
    if fq_score_min_ghz > fq_score_max_ghz:
        fq_score_min_ghz, fq_score_max_ghz = fq_score_max_ghz, fq_score_min_ghz
    g_target_mhz = _env_float("METAL_2Q_TARGET_G_MHZ", 100.0, lo=1.0, hi=500.0)
    t1_target_us = _env_float("METAL_2Q_TARGET_T1_US", 100.0, lo=0.1, hi=1.0e7)
    chi_t1_min_us = _env_float("METAL_2Q_CHI_T1_MIN_US", 10.0, lo=0.1, hi=1.0e6)
    transmon_epr_gap_slack_ghz = _env_float("METAL_2Q_TRANSMON_EPR_GAP_SLACK_GHZ", 0.05, lo=0.0, hi=5.0)

    fq1_target = _env_specific_or_shared_float(
        "METAL_2Q_TARGET_FQ1_GHZ",
        "METAL_2Q_TARGET_FQ_GHZ",
        fq_target_default,
        lo=0.1,
        hi=20.0,
    )
    fq2_target = _env_specific_or_shared_float(
        "METAL_2Q_TARGET_FQ2_GHZ",
        "METAL_2Q_TARGET_FQ_GHZ",
        fq_target_default,
        lo=0.1,
        hi=20.0,
    )
    fr1_target = _env_specific_or_shared_float(
        "METAL_2Q_TARGET_FR1_GHZ",
        "METAL_2Q_TARGET_FR_GHZ",
        fr_target_default,
        lo=0.1,
        hi=20.0,
    )
    fr2_target = _env_specific_or_shared_float(
        "METAL_2Q_TARGET_FR2_GHZ",
        "METAL_2Q_TARGET_FR_GHZ",
        fr_target_default,
        lo=0.1,
        hi=20.0,
    )
    chi1_target = _env_specific_or_shared_float(
        "METAL_2Q_TARGET_CHI1_MHZ",
        "METAL_2Q_TARGET_CHI_MHZ",
        chi_target_default,
        lo=0.05,
        hi=50.0,
    )
    chi2_target = _env_specific_or_shared_float(
        "METAL_2Q_TARGET_CHI2_MHZ",
        "METAL_2Q_TARGET_CHI_MHZ",
        chi_target_default,
        lo=0.05,
        hi=50.0,
    )

    alpha_min_mhz = _env_float("METAL_2Q_ALPHA_MIN_MHZ", -150.0, lo=-1000.0, hi=0.0)
    alpha_max_mhz = _env_float("METAL_2Q_ALPHA_MAX_MHZ", -40.0, lo=-1000.0, hi=0.0)
    if alpha_min_mhz > alpha_max_mhz:
        alpha_min_mhz, alpha_max_mhz = alpha_max_mhz, alpha_min_mhz

    kappa_min_hz = _env_float("METAL_2Q_KAPPA_MIN_HZ", 1.0e5, lo=1.0, hi=1.0e12)
    kappa_max_hz = _env_float("METAL_2Q_KAPPA_MAX_HZ", 1.5e7, lo=1.0, hi=1.0e12)
    if kappa_min_hz > kappa_max_hz:
        kappa_min_hz, kappa_max_hz = kappa_max_hz, kappa_min_hz

    q_loaded_min = _env_float("METAL_2Q_QLOADED_MIN", 1.0e3, lo=1.0, hi=1.0e12)
    q_loaded_max = _env_float("METAL_2Q_QLOADED_MAX", 2.0e5, lo=1.0, hi=1.0e12)
    if q_loaded_min > q_loaded_max:
        q_loaded_min, q_loaded_max = q_loaded_max, q_loaded_min

    t1_anchor_fq_tol_ghz = _env_float("METAL_2Q_T1_ANCHOR_FQ_TOL_GHZ", 0.15, lo=0.001, hi=2.0)
    t1_anchor_fr_tol_ghz = _env_float("METAL_2Q_T1_ANCHOR_FR_TOL_GHZ", 0.10, lo=0.001, hi=2.0)

    def _payload_float(payload: TDict[str, Any], path: Tuple[str, ...]) -> Optional[float]:
        cur: Any = payload
        for key in path:
            if not isinstance(cur, dict) or key not in cur:
                return None
            cur = cur[key]
        try:
            value = float(cur)
        except Exception:
            return None
        if not math.isfinite(value):
            return None
        return float(value)

    def _soft_window_pen(x: Optional[float], lo: float, hi: float, *, missing_pen: float = 20.0) -> float:
        if x is None:
            return float(missing_pen)
        lo_f = float(lo)
        hi_f = float(hi)
        if lo_f > hi_f:
            lo_f, hi_f = hi_f, lo_f
        span = max(1e-9, hi_f - lo_f)
        center = 0.5 * (lo_f + hi_f)
        xf = float(x)
        if xf < lo_f:
            return 1.0 + (lo_f - xf) / span
        if xf > hi_f:
            return 1.0 + (xf - hi_f) / span
        return 0.10 * abs(xf - center) / span

    def _target_pen(x: Optional[float], target: float, *, scale: float, missing_pen: float = 20.0) -> float:
        if x is None:
            return float(missing_pen)
        return abs(float(x) - float(target)) / max(float(scale), 1e-9)

    def _anchor_pen(
        x: Optional[float],
        anchor: Optional[float],
        *,
        tol: float,
        missing_pen: float = 5.0,
    ) -> float:
        if anchor is None:
            return 0.0
        if x is None:
            return float(missing_pen)
        return abs(float(x) - float(anchor)) / max(float(tol), 1e-9)

    def _floor_pen(x: Optional[float], floor: float, *, scale: float, missing_pen: float = 20.0) -> float:
        if x is None:
            return float(missing_pen)
        xf = float(x)
        ff = float(floor)
        if xf >= ff:
            return 0.0
        return (ff - xf) / max(float(scale), 1e-9)

    def _log_target_pen(x: Optional[float], target: float, *, missing_pen: float = 20.0) -> float:
        if x is None:
            return float(missing_pen)
        xf = float(x)
        tf = float(target)
        if xf <= 0.0 or tf <= 0.0:
            return float(missing_pen)
        return abs(math.log10(xf / tf))

    def _gap_growth_pen(
        gap: Optional[float],
        anchor_gap: Optional[float],
        *,
        slack: float,
        scale: float = 0.10,
        missing_pen: float = 8.0,
    ) -> float:
        if gap is None:
            return float(missing_pen)
        if anchor_gap is None:
            return 0.0
        limit = float(anchor_gap) + float(slack)
        gf = float(gap)
        if gf <= limit:
            return 0.0
        return (gf - limit) / max(float(scale), 1e-9)

    def _hard_error_summary(payload: TDict[str, Any]) -> List[str]:
        meta = payload.get("meta") or {}
        out: List[str] = []
        for key in ("mode_picker_error", "hfss_error", "q3d_error"):
            raw = str(meta.get(key) or "").strip()
            if raw:
                out.append(f"{key}={raw}")
        return out

    def _score_payload(
        payload: TDict[str, Any],
        *,
        stage: str = "full",
        anchor_payload: Optional[TDict[str, Any]] = None,
    ) -> Tuple[float, TDict[str, Any]]:
        fq1_t = _payload_float(payload, ("qubits", "Q1", "f01_transmon_GHz"))
        fq2_t = _payload_float(payload, ("qubits", "Q2", "f01_transmon_GHz"))
        fq1_epr = _payload_float(payload, ("qubits", "Q1", "f01_epr_GHz"))
        fq2_epr = _payload_float(payload, ("qubits", "Q2", "f01_epr_GHz"))
        alpha1 = _payload_float(payload, ("qubits", "Q1", "alpha_epr_MHz"))
        alpha2 = _payload_float(payload, ("qubits", "Q2", "alpha_epr_MHz"))
        g1_ghz = _payload_float(payload, ("qubits", "Q1", "dispersive", "g_GHz"))
        g2_ghz = _payload_float(payload, ("qubits", "Q2", "dispersive", "g_GHz"))
        g1_mhz = (g1_ghz * 1e3) if g1_ghz is not None else None
        g2_mhz = (g2_ghz * 1e3) if g2_ghz is not None else None
        fr1 = _payload_float(payload, ("resonators", "readout1", "f_GHz"))
        fr2 = _payload_float(payload, ("resonators", "readout2", "f_GHz"))
        chi1 = _payload_float(payload, ("qubits", "Q1", "chi_MHz"))
        chi2 = _payload_float(payload, ("qubits", "Q2", "chi_MHz"))
        kappa1 = _payload_float(payload, ("resonators", "readout1", "kappa_over_2pi_Hz"))
        kappa2 = _payload_float(payload, ("resonators", "readout2", "kappa_over_2pi_Hz"))
        q_loaded1 = _payload_float(payload, ("resonators", "readout1", "Q_loaded"))
        q_loaded2 = _payload_float(payload, ("resonators", "readout2", "Q_loaded"))
        t1_1_us = _payload_float(payload, ("chip", "T1_qubit1_est_us"))
        t1_2_us = _payload_float(payload, ("chip", "T1_qubit2_est_us"))
        gap1 = abs(float(fq1_t) - float(fq1_epr)) if fq1_t is not None and fq1_epr is not None else None
        gap2 = abs(float(fq2_t) - float(fq2_epr)) if fq2_t is not None and fq2_epr is not None else None
        anchor_gap1 = None
        anchor_gap2 = None
        if anchor_payload is not None:
            anchor_fq1_t = _payload_float(anchor_payload, ("qubits", "Q1", "f01_transmon_GHz"))
            anchor_fq2_t = _payload_float(anchor_payload, ("qubits", "Q2", "f01_transmon_GHz"))
            anchor_fq1_epr = _payload_float(anchor_payload, ("qubits", "Q1", "f01_epr_GHz"))
            anchor_fq2_epr = _payload_float(anchor_payload, ("qubits", "Q2", "f01_epr_GHz"))
            if anchor_fq1_t is not None and anchor_fq1_epr is not None:
                anchor_gap1 = abs(float(anchor_fq1_t) - float(anchor_fq1_epr))
            if anchor_fq2_t is not None and anchor_fq2_epr is not None:
                anchor_gap2 = abs(float(anchor_fq2_t) - float(anchor_fq2_epr))

        hard_errors = _hard_error_summary(payload)
        pen = 0.0
        if hard_errors:
            pen += 1.0e6

        if stage == "fq":
            pen += _soft_window_pen(fq1_epr, fq_score_min_ghz, fq_score_max_ghz)
            pen += _soft_window_pen(fq2_epr, fq_score_min_ghz, fq_score_max_ghz)
            pen += 0.75 * _soft_window_pen(alpha1, alpha_min_mhz, alpha_max_mhz)
            pen += 0.75 * _soft_window_pen(alpha2, alpha_min_mhz, alpha_max_mhz)
            pen += 0.50 * _gap_growth_pen(
                gap1,
                anchor_gap1,
                slack=transmon_epr_gap_slack_ghz,
            )
            pen += 0.50 * _gap_growth_pen(
                gap2,
                anchor_gap2,
                slack=transmon_epr_gap_slack_ghz,
            )
        elif stage == "fr":
            pen += _target_pen(fr1, fr1_target, scale=0.02)
            pen += _target_pen(fr2, fr2_target, scale=0.02)
        elif stage == "chi":
            pen += _target_pen(abs(chi1) if chi1 is not None else None, chi1_target, scale=0.20)
            pen += _target_pen(abs(chi2) if chi2 is not None else None, chi2_target, scale=0.20)
            pen += 0.45 * _floor_pen(g1_mhz, g_target_mhz, scale=15.0, missing_pen=12.0)
            pen += 0.45 * _floor_pen(g2_mhz, g_target_mhz, scale=15.0, missing_pen=12.0)
            pen += 0.35 * _floor_pen(t1_1_us, chi_t1_min_us, scale=chi_t1_min_us, missing_pen=12.0)
            pen += 0.35 * _floor_pen(t1_2_us, chi_t1_min_us, scale=chi_t1_min_us, missing_pen=12.0)
            pen += 0.40 * _soft_window_pen(alpha1, alpha_min_mhz, alpha_max_mhz)
            pen += 0.40 * _soft_window_pen(alpha2, alpha_min_mhz, alpha_max_mhz)
            pen += 0.60 * _soft_window_pen(kappa1, kappa_min_hz, kappa_max_hz, missing_pen=8.0)
            pen += 0.60 * _soft_window_pen(kappa2, kappa_min_hz, kappa_max_hz, missing_pen=8.0)
            pen += 0.25 * _soft_window_pen(q_loaded1, q_loaded_min, q_loaded_max, missing_pen=6.0)
            pen += 0.25 * _soft_window_pen(q_loaded2, q_loaded_min, q_loaded_max, missing_pen=6.0)
            pen += 0.50 * _gap_growth_pen(
                gap1,
                anchor_gap1,
                slack=transmon_epr_gap_slack_ghz,
            )
            pen += 0.50 * _gap_growth_pen(
                gap2,
                anchor_gap2,
                slack=transmon_epr_gap_slack_ghz,
            )
        elif stage == "t1":
            pen += 4.0 * _floor_pen(t1_1_us, t1_target_us, scale=t1_target_us, missing_pen=50.0)
            pen += 4.0 * _floor_pen(t1_2_us, t1_target_us, scale=t1_target_us, missing_pen=50.0)
            pen += _soft_window_pen(fq1_epr, q_pick_min, q_pick_max, missing_pen=50.0)
            pen += _soft_window_pen(fq2_epr, q_pick_min, q_pick_max, missing_pen=50.0)
            pen += _soft_window_pen(fr1, r_pick_min, r_pick_max, missing_pen=50.0)
            pen += _soft_window_pen(fr2, r_pick_min, r_pick_max, missing_pen=50.0)
            if fq1_epr is None or fr1 is None or float(fq1_epr) >= float(fr1):
                pen += 50.0
            if fq2_epr is None or fr2 is None or float(fq2_epr) >= float(fr2):
                pen += 50.0
            anchor_fq1 = _payload_float(anchor_payload, ("qubits", "Q1", "f01_epr_GHz")) if anchor_payload else None
            anchor_fq2 = _payload_float(anchor_payload, ("qubits", "Q2", "f01_epr_GHz")) if anchor_payload else None
            anchor_fr1 = _payload_float(anchor_payload, ("resonators", "readout1", "f_GHz")) if anchor_payload else None
            anchor_fr2 = _payload_float(anchor_payload, ("resonators", "readout2", "f_GHz")) if anchor_payload else None
            pen += 0.25 * _anchor_pen(fq1_epr, anchor_fq1, tol=t1_anchor_fq_tol_ghz)
            pen += 0.25 * _anchor_pen(fq2_epr, anchor_fq2, tol=t1_anchor_fq_tol_ghz)
            pen += 0.15 * _anchor_pen(fr1, anchor_fr1, tol=t1_anchor_fr_tol_ghz)
            pen += 0.15 * _anchor_pen(fr2, anchor_fr2, tol=t1_anchor_fr_tol_ghz)
            pen += 0.10 * _soft_window_pen(alpha1, alpha_min_mhz, alpha_max_mhz)
            pen += 0.10 * _soft_window_pen(alpha2, alpha_min_mhz, alpha_max_mhz)
            pen += 0.45 * _gap_growth_pen(
                gap1,
                anchor_gap1,
                slack=transmon_epr_gap_slack_ghz,
            )
            pen += 0.45 * _gap_growth_pen(
                gap2,
                anchor_gap2,
                slack=transmon_epr_gap_slack_ghz,
            )
        else:
            pen += _soft_window_pen(fq1_epr, fq_score_min_ghz, fq_score_max_ghz)
            pen += _soft_window_pen(fq2_epr, fq_score_min_ghz, fq_score_max_ghz)
            pen += 0.75 * _soft_window_pen(alpha1, alpha_min_mhz, alpha_max_mhz)
            pen += 0.75 * _soft_window_pen(alpha2, alpha_min_mhz, alpha_max_mhz)
            pen += 0.75 * _target_pen(fr1, fr1_target, scale=0.02)
            pen += 0.75 * _target_pen(fr2, fr2_target, scale=0.02)
            pen += 0.15 * _target_pen(abs(chi1) if chi1 is not None else None, chi1_target, scale=0.20)
            pen += 0.15 * _target_pen(abs(chi2) if chi2 is not None else None, chi2_target, scale=0.20)
            pen += 0.10 * _floor_pen(g1_mhz, g_target_mhz, scale=15.0, missing_pen=12.0)
            pen += 0.10 * _floor_pen(g2_mhz, g_target_mhz, scale=15.0, missing_pen=12.0)
            pen += 0.50 * _soft_window_pen(kappa1, kappa_min_hz, kappa_max_hz, missing_pen=8.0)
            pen += 0.50 * _soft_window_pen(kappa2, kappa_min_hz, kappa_max_hz, missing_pen=8.0)
            pen += 0.20 * _soft_window_pen(q_loaded1, q_loaded_min, q_loaded_max, missing_pen=6.0)
            pen += 0.20 * _soft_window_pen(q_loaded2, q_loaded_min, q_loaded_max, missing_pen=6.0)
            pen += 4.0 * _floor_pen(t1_1_us, t1_target_us, scale=t1_target_us, missing_pen=50.0)
            pen += 4.0 * _floor_pen(t1_2_us, t1_target_us, scale=t1_target_us, missing_pen=50.0)
            if fq1_epr is None or fr1 is None or float(fq1_epr) >= float(fr1):
                pen += 50.0
            if fq2_epr is None or fr2 is None or float(fq2_epr) >= float(fr2):
                pen += 50.0
            if anchor_payload is not None:
                pen += 0.50 * _anchor_pen(
                    fq1_epr,
                    _payload_float(anchor_payload, ("qubits", "Q1", "f01_epr_GHz")),
                    tol=t1_anchor_fq_tol_ghz,
                )
                pen += 0.50 * _anchor_pen(
                    fq2_epr,
                    _payload_float(anchor_payload, ("qubits", "Q2", "f01_epr_GHz")),
                    tol=t1_anchor_fq_tol_ghz,
                )
                pen += 0.50 * _anchor_pen(
                    fr1,
                    _payload_float(anchor_payload, ("resonators", "readout1", "f_GHz")),
                    tol=t1_anchor_fr_tol_ghz,
                )
                pen += 0.50 * _anchor_pen(
                    fr2,
                    _payload_float(anchor_payload, ("resonators", "readout2", "f_GHz")),
                    tol=t1_anchor_fr_tol_ghz,
                )
            pen += 0.50 * _gap_growth_pen(
                gap1,
                anchor_gap1,
                slack=transmon_epr_gap_slack_ghz,
            )
            pen += 0.50 * _gap_growth_pen(
                gap2,
                anchor_gap2,
                slack=transmon_epr_gap_slack_ghz,
            )

        details: TDict[str, Any] = {
            "stage": stage,
            "fq1_transmon_GHz": fq1_t,
            "fq2_transmon_GHz": fq2_t,
            "fq1_epr_GHz": fq1_epr,
            "fq2_epr_GHz": fq2_epr,
            "fq_score_window_GHz": [fq_score_min_ghz, fq_score_max_ghz],
            "alpha1_MHz": alpha1,
            "alpha2_MHz": alpha2,
            "g1_MHz": g1_mhz,
            "g2_MHz": g2_mhz,
            "fr1_GHz": fr1,
            "fr2_GHz": fr2,
            "chi1_MHz": chi1,
            "chi2_MHz": chi2,
            "transmon_epr_gap1_GHz": gap1,
            "transmon_epr_gap2_GHz": gap2,
            "kappa1_Hz": kappa1,
            "kappa2_Hz": kappa2,
            "Q_loaded1": q_loaded1,
            "Q_loaded2": q_loaded2,
            "T1_qubit1_us": t1_1_us,
            "T1_qubit2_us": t1_2_us,
            "hard_errors": hard_errors or None,
        }
        return float(pen), details

    def _clone_p2(src: Layout2QParams) -> Layout2QParams:
        return Layout2QParams(
            shared=src.shared,
            ro1_dx_mm=src.ro1_dx_mm,
            ro1_dy_mm=src.ro1_dy_mm,
            ro2_dx_mm=src.ro2_dx_mm,
            ro2_dy_mm=src.ro2_dy_mm,
            tee_offset_mm=src.tee_offset_mm,
            ro1_total_length_mm=src.ro1_total_length_mm,
            ro2_total_length_mm=src.ro2_total_length_mm,
            ro1_tee_cap_gap_um=src.ro1_tee_cap_gap_um,
            ro2_tee_cap_gap_um=src.ro2_tee_cap_gap_um,
            ro1_tee_cap_width_um=src.ro1_tee_cap_width_um,
            ro2_tee_cap_width_um=src.ro2_tee_cap_width_um,
            ro1_tee_cap_distance_um=src.ro1_tee_cap_distance_um,
            ro2_tee_cap_distance_um=src.ro2_tee_cap_distance_um,
            ro1_tee_finger_length_um=src.ro1_tee_finger_length_um,
            ro2_tee_finger_length_um=src.ro2_tee_finger_length_um,
            ro1_tee_finger_count=src.ro1_tee_finger_count,
            ro2_tee_finger_count=src.ro2_tee_finger_count,
            q1_pad_width_um=src.q1_pad_width_um,
            q2_pad_width_um=src.q2_pad_width_um,
            q1_pocket_height_um=src.q1_pocket_height_um,
            q2_pocket_height_um=src.q2_pocket_height_um,
            q1_ro_pad_w_um=src.q1_ro_pad_w_um,
            q2_ro_pad_w_um=src.q2_ro_pad_w_um,
            q1_ro_pad_h_um=src.q1_ro_pad_h_um,
            q2_ro_pad_h_um=src.q2_ro_pad_h_um,
            q1_ro_pad_gap_um=src.q1_ro_pad_gap_um,
            q2_ro_pad_gap_um=src.q2_ro_pad_gap_um,
            q1_bus_pad_w_um=src.q1_bus_pad_w_um,
            q2_bus_pad_w_um=src.q2_bus_pad_w_um,
            q1_bus_pad_h_um=src.q1_bus_pad_h_um,
            q2_bus_pad_h_um=src.q2_bus_pad_h_um,
            q1_bus_pad_gap_um=src.q1_bus_pad_gap_um,
            q2_bus_pad_gap_um=src.q2_bus_pad_gap_um,
            lp1_dx_mm=src.lp1_dx_mm,
            lp2_dx_mm=src.lp2_dx_mm,
            ro1_purcell_stub_length_mm=src.ro1_purcell_stub_length_mm,
            ro2_purcell_stub_length_mm=src.ro2_purcell_stub_length_mm,
            ro1_purcell_stub_offset_mm=src.ro1_purcell_stub_offset_mm,
            ro2_purcell_stub_offset_mm=src.ro2_purcell_stub_offset_mm,
            ro1_purcell_stub_width_um=src.ro1_purcell_stub_width_um,
            ro2_purcell_stub_width_um=src.ro2_purcell_stub_width_um,
            ro1_purcell_stub_gap_um=src.ro1_purcell_stub_gap_um,
            ro2_purcell_stub_gap_um=src.ro2_purcell_stub_gap_um,
            swap_tee_ports=src.swap_tee_ports,
            bus_width=src.bus_width,
            bus_gap=src.bus_gap,
            bus_pad_gap_um=src.bus_pad_gap_um,
            bus_lead_start=src.bus_lead_start,
            bus_lead_end=src.bus_lead_end,
            bus_fillet=src.bus_fillet,
            bus_spacing=src.bus_spacing,
            bus_total_length=src.bus_total_length,
        )

    def _mutate_float_value(
        value: float,
        *,
        sigma: float,
        lo: float,
        hi: float,
        rng: np.random.Generator,
        scale: float,
    ) -> float:
        return float(np.clip(float(value) + rng.normal(0.0, float(sigma) * float(scale)), float(lo), float(hi)))

    def _mutate_int_value(
        value: int,
        *,
        sigma: int,
        lo: int,
        hi: int,
        rng: np.random.Generator,
        scale: float,
    ) -> int:
        span = max(1, int(round(int(sigma) * float(scale))))
        return int(np.clip(int(value) + int(rng.integers(-span, span + 1)), int(lo), int(hi)))

    def _mutate_log_value(
        value: float,
        *,
        sigma_decades: float,
        lo: float,
        hi: float,
        rng: np.random.Generator,
        scale: float,
    ) -> float:
        anchor = max(float(value), float(lo))
        return float(np.clip(anchor * (10.0 ** rng.normal(0.0, float(sigma_decades) * float(scale))), float(lo), float(hi)))

    def _mutate_p2(
        src: Layout2QParams,
        *,
        rng: np.random.Generator,
        stage: str = "full",
        allow_bus_pad_gap: bool = False,
    ) -> Layout2QParams:
        out = _clone_p2(src)
        local_scale = 0.50 if stage == "full" else 1.0

        if stage in {"fq", "full"}:
            out.q1_pad_width_um = _mutate_float_value(
                float(src.q1_pad_width_um or 425.0),
                sigma=18.0,
                lo=100.0,
                hi=2000.0,
                rng=rng,
                scale=local_scale,
            )
            out.q2_pad_width_um = _mutate_float_value(
                float(src.q2_pad_width_um or 425.0),
                sigma=18.0,
                lo=100.0,
                hi=2000.0,
                rng=rng,
                scale=local_scale,
            )
            out.q1_pocket_height_um = _mutate_float_value(
                float(src.q1_pocket_height_um or 650.0),
                sigma=20.0,
                lo=100.0,
                hi=3000.0,
                rng=rng,
                scale=local_scale,
            )
            out.q2_pocket_height_um = _mutate_float_value(
                float(src.q2_pocket_height_um or 650.0),
                sigma=20.0,
                lo=100.0,
                hi=3000.0,
                rng=rng,
                scale=local_scale,
            )

        if stage in {"fr", "full"}:
            out.ro1_total_length_mm = _mutate_float_value(
                float(src.ro1_total_length_mm or RO_L_MM),
                sigma=0.22,
                lo=3.0,
                hi=20.0,
                rng=rng,
                scale=local_scale,
            )
            out.ro2_total_length_mm = _mutate_float_value(
                float(src.ro2_total_length_mm or RO_L_MM),
                sigma=0.22,
                lo=3.0,
                hi=20.0,
                rng=rng,
                scale=local_scale,
            )

        if stage in {"chi", "full"}:
            out.q1_ro_pad_w_um = _mutate_float_value(
                float(src.q1_ro_pad_w_um or 360.0),
                sigma=12.0,
                lo=260.0,
                hi=420.0,
                rng=rng,
                scale=local_scale,
            )
            out.q2_ro_pad_w_um = _mutate_float_value(
                float(src.q2_ro_pad_w_um or 520.0),
                sigma=18.0,
                lo=360.0,
                hi=700.0,
                rng=rng,
                scale=local_scale,
            )
            out.q1_ro_pad_h_um = _mutate_float_value(
                float(src.q1_ro_pad_h_um or 360.0),
                sigma=12.0,
                lo=260.0,
                hi=420.0,
                rng=rng,
                scale=local_scale,
            )
            out.q2_ro_pad_h_um = _mutate_float_value(
                float(src.q2_ro_pad_h_um or 520.0),
                sigma=18.0,
                lo=360.0,
                hi=700.0,
                rng=rng,
                scale=local_scale,
            )
            out.q1_ro_pad_gap_um = _mutate_float_value(
                float(src.q1_ro_pad_gap_um or Q_RO_PAD_GAP_UM),
                sigma=0.10,
                lo=0.4,
                hi=1.5,
                rng=rng,
                scale=local_scale,
            )
            out.q2_ro_pad_gap_um = _mutate_float_value(
                float(src.q2_ro_pad_gap_um or Q_RO_PAD_GAP_UM),
                sigma=0.10,
                lo=0.7,
                hi=2.0,
                rng=rng,
                scale=local_scale,
            )
            out.ro1_tee_cap_gap_um = _mutate_float_value(
                float(src.ro1_tee_cap_gap_um or TEE_CAP_GAP_UM),
                sigma=0.08,
                lo=0.9,
                hi=2.4,
                rng=rng,
                scale=local_scale,
            )
            out.ro2_tee_cap_gap_um = _mutate_float_value(
                float(src.ro2_tee_cap_gap_um or TEE_CAP_GAP_UM),
                sigma=0.08,
                lo=0.9,
                hi=2.4,
                rng=rng,
                scale=local_scale,
            )
            out.ro1_tee_cap_width_um = _mutate_float_value(
                float(src.ro1_tee_cap_width_um or TEE_CAP_WIDTH_UM),
                sigma=0.8,
                lo=12.0,
                hi=22.0,
                rng=rng,
                scale=local_scale,
            )
            out.ro2_tee_cap_width_um = _mutate_float_value(
                float(src.ro2_tee_cap_width_um or TEE_CAP_WIDTH_UM),
                sigma=0.8,
                lo=12.0,
                hi=22.0,
                rng=rng,
                scale=local_scale,
            )
            out.ro1_tee_cap_distance_um = _mutate_float_value(
                float(src.ro1_tee_cap_distance_um or TEE_CAP_DISTANCE_UM),
                sigma=3.0,
                lo=32.0,
                hi=60.0,
                rng=rng,
                scale=local_scale,
            )
            out.ro2_tee_cap_distance_um = _mutate_float_value(
                float(src.ro2_tee_cap_distance_um or TEE_CAP_DISTANCE_UM),
                sigma=3.0,
                lo=32.0,
                hi=60.0,
                rng=rng,
                scale=local_scale,
            )
            out.ro1_tee_finger_length_um = _mutate_int_value(
                int(src.ro1_tee_finger_length_um or TEE_FINGER_LENGTH_UM),
                sigma=5,
                lo=140,
                hi=190,
                rng=rng,
                scale=local_scale,
            )
            out.ro2_tee_finger_length_um = _mutate_int_value(
                int(src.ro2_tee_finger_length_um or TEE_FINGER_LENGTH_UM),
                sigma=5,
                lo=140,
                hi=190,
                rng=rng,
                scale=local_scale,
            )
            out.ro1_tee_finger_count = _mutate_int_value(
                int(src.ro1_tee_finger_count or TEE_FINGER_COUNT),
                sigma=1,
                lo=10,
                hi=14,
                rng=rng,
                scale=local_scale,
            )
            out.ro2_tee_finger_count = _mutate_int_value(
                int(src.ro2_tee_finger_count or TEE_FINGER_COUNT),
                sigma=1,
                lo=10,
                hi=14,
                rng=rng,
                scale=local_scale,
            )
            if allow_bus_pad_gap:
                base_bus_pad_gap = (
                    float(src.bus_pad_gap_um)
                    if src.bus_pad_gap_um is not None
                    else 0.5 * float((src.q1_ro_pad_gap_um or Q_RO_PAD_GAP_UM) + (src.q2_ro_pad_gap_um or Q_RO_PAD_GAP_UM))
                )
                out.bus_pad_gap_um = _mutate_float_value(
                    base_bus_pad_gap,
                    sigma=0.10,
                    lo=0.2,
                    hi=30.0,
                    rng=rng,
                    scale=local_scale,
                )

        if stage in {"t1", "full"}:
            out.ro1_purcell_stub_length_mm = _mutate_float_value(
                float(src.ro1_purcell_stub_length_mm or PURCELL_STUB_LENGTH_MM),
                sigma=0.20,
                lo=0.0,
                hi=50.0,
                rng=rng,
                scale=local_scale,
            )
            out.ro2_purcell_stub_length_mm = _mutate_float_value(
                float(src.ro2_purcell_stub_length_mm or PURCELL_STUB_LENGTH_MM),
                sigma=0.20,
                lo=0.0,
                hi=50.0,
                rng=rng,
                scale=local_scale,
            )
            out.ro1_purcell_stub_width_um = _mutate_float_value(
                float(src.ro1_purcell_stub_width_um or PURCELL_STUB_WIDTH_UM),
                sigma=0.8,
                lo=0.1,
                hi=500.0,
                rng=rng,
                scale=local_scale,
            )
            out.ro2_purcell_stub_width_um = _mutate_float_value(
                float(src.ro2_purcell_stub_width_um or PURCELL_STUB_WIDTH_UM),
                sigma=0.8,
                lo=0.1,
                hi=500.0,
                rng=rng,
                scale=local_scale,
            )
            out.ro1_purcell_stub_gap_um = _mutate_float_value(
                float(src.ro1_purcell_stub_gap_um or PURCELL_STUB_GAP_UM),
                sigma=0.30,
                lo=0.1,
                hi=500.0,
                rng=rng,
                scale=local_scale,
            )
            out.ro2_purcell_stub_gap_um = _mutate_float_value(
                float(src.ro2_purcell_stub_gap_um or PURCELL_STUB_GAP_UM),
                sigma=0.30,
                lo=0.1,
                hi=500.0,
                rng=rng,
                scale=local_scale,
            )

        return out

    def _mutate_state(
        src: TDict[str, Any],
        *,
        rng: np.random.Generator,
        stage: str,
    ) -> TDict[str, Any]:
        local_scale = 0.50 if stage == "full" else 1.0
        out: TDict[str, Any] = {
            "p2": _mutate_p2(
                cast(Layout2QParams, src["p2"]),
                rng=rng,
                stage=stage,
                allow_bus_pad_gap=allow_bus_pad_gap_opt,
            ),
            "lj1_nh": float(src["lj1_nh"]),
            "cj1_fF": float(src["cj1_fF"]),
            "lj2_nh": float(src["lj2_nh"]),
            "cj2_fF": float(src["cj2_fF"]),
            "tan_delta_main": float(src["tan_delta_main"]),
            "q1_effective_cin_fF": float(src["q1_effective_cin_fF"]),
            "q2_effective_cin_fF": float(src["q2_effective_cin_fF"]),
            "q1_target_kext_hz": float(src["q1_target_kext_hz"]),
            "q2_target_kext_hz": float(src["q2_target_kext_hz"]),
        }
        if stage in {"fq", "full"}:
            out["lj1_nh"] = _mutate_float_value(
                float(src["lj1_nh"]),
                sigma=0.30,
                lo=0.1,
                hi=float(src["lj1_nh"]),
                rng=rng,
                scale=local_scale,
            )
            out["lj2_nh"] = _mutate_float_value(
                float(src["lj2_nh"]),
                sigma=0.30,
                lo=0.1,
                hi=float(src["lj2_nh"]),
                rng=rng,
                scale=local_scale,
            )
            out["cj1_fF"] = _mutate_float_value(
                float(src["cj1_fF"]),
                sigma=0.20,
                lo=0.0,
                hi=100.0,
                rng=rng,
                scale=local_scale,
            )
            out["cj2_fF"] = _mutate_float_value(
                float(src["cj2_fF"]),
                sigma=0.20,
                lo=0.0,
                hi=100.0,
                rng=rng,
                scale=local_scale,
            )
        if stage in {"t1", "full"}:
            out["tan_delta_main"] = _mutate_log_value(
                float(src["tan_delta_main"]),
                sigma_decades=0.12,
                lo=1e-7,
                hi=1e-1,
                rng=rng,
                scale=local_scale,
            )
        return out

    def _state_to_dict(state: TDict[str, Any]) -> TDict[str, Any]:
        return {
            "Q1_X_MM": float(state.get("q1_x_mm", Q1_X_MM)),
            "Q1_Y_MM": float(state.get("q1_y_mm", Q1_Y_MM)),
            "Q2_X_MM": float(state.get("q2_x_mm", Q2_X_MM)),
            "Q2_Y_MM": float(state.get("q2_y_mm", Q2_Y_MM)),
            "layout": _layout2qparams_to_dict(cast(Layout2QParams, state["p2"])),
            "Lj1_nH": float(state["lj1_nh"]),
            "Cj1_fF": float(state["cj1_fF"]),
            "Lj2_nH": float(state["lj2_nh"]),
            "Cj2_fF": float(state["cj2_fF"]),
            "TAN_DELTA_MAIN": float(state["tan_delta_main"]),
            "Q1_EFFECTIVE_CIN_FF": float(state["q1_effective_cin_fF"]),
            "Q2_EFFECTIVE_CIN_FF": float(state["q2_effective_cin_fF"]),
            "Q1_TARGET_KEXT_HZ": float(state["q1_target_kext_hz"]),
            "Q2_TARGET_KEXT_HZ": float(state["q2_target_kext_hz"]),
        }

    def _metric_snapshot(payload: TDict[str, Any]) -> TDict[str, Any]:
        return {
            "Q1": {
                "f01_epr_GHz": _payload_float(payload, ("qubits", "Q1", "f01_epr_GHz")),
                "f01_transmon_GHz": _payload_float(payload, ("qubits", "Q1", "f01_transmon_GHz")),
                "alpha_epr_MHz": _payload_float(payload, ("qubits", "Q1", "alpha_epr_MHz")),
                "chi_MHz": _payload_float(payload, ("qubits", "Q1", "chi_MHz")),
                "T1_est_us": _payload_float(payload, ("chip", "T1_qubit1_est_us")),
            },
            "Q2": {
                "f01_epr_GHz": _payload_float(payload, ("qubits", "Q2", "f01_epr_GHz")),
                "f01_transmon_GHz": _payload_float(payload, ("qubits", "Q2", "f01_transmon_GHz")),
                "alpha_epr_MHz": _payload_float(payload, ("qubits", "Q2", "alpha_epr_MHz")),
                "chi_MHz": _payload_float(payload, ("qubits", "Q2", "chi_MHz")),
                "T1_est_us": _payload_float(payload, ("chip", "T1_qubit2_est_us")),
            },
            "readout1": {
                "f_GHz": _payload_float(payload, ("resonators", "readout1", "f_GHz")),
                "kappa_i_over_2pi_Hz": _payload_float(payload, ("resonators", "readout1", "kappa_i_over_2pi_Hz")),
                "kappa_ext_over_2pi_Hz": _payload_float(payload, ("resonators", "readout1", "external", "kappa_over_2pi_Hz")),
                "kappa_over_2pi_Hz": _payload_float(payload, ("resonators", "readout1", "kappa_over_2pi_Hz")),
                "Q_loaded": _payload_float(payload, ("resonators", "readout1", "Q_loaded")),
            },
            "readout2": {
                "f_GHz": _payload_float(payload, ("resonators", "readout2", "f_GHz")),
                "kappa_i_over_2pi_Hz": _payload_float(payload, ("resonators", "readout2", "kappa_i_over_2pi_Hz")),
                "kappa_ext_over_2pi_Hz": _payload_float(payload, ("resonators", "readout2", "external", "kappa_over_2pi_Hz")),
                "kappa_over_2pi_Hz": _payload_float(payload, ("resonators", "readout2", "kappa_over_2pi_Hz")),
                "Q_loaded": _payload_float(payload, ("resonators", "readout2", "Q_loaded")),
            },
        }

    def _print_metric_snapshot(label: str, payload: TDict[str, Any]) -> None:
        snap = _metric_snapshot(payload)
        q1 = snap["Q1"]
        q2 = snap["Q2"]
        r1 = snap["readout1"]
        r2 = snap["readout2"]
        print(
            f"{label} Q1: f01_epr_GHz={q1['f01_epr_GHz']} alpha_epr_MHz={q1['alpha_epr_MHz']} "
            f"chi_MHz={q1['chi_MHz']} T1_est_us={q1['T1_est_us']}",
            flush=True,
        )
        print(
            f"{label} Q2: f01_epr_GHz={q2['f01_epr_GHz']} alpha_epr_MHz={q2['alpha_epr_MHz']} "
            f"chi_MHz={q2['chi_MHz']} T1_est_us={q2['T1_est_us']}",
            flush=True,
        )
        print(
            f"{label} readout1: f_GHz={r1['f_GHz']} kappa_over_2pi_Hz={r1['kappa_over_2pi_Hz']} "
            f"kappa_i_over_2pi_Hz={r1['kappa_i_over_2pi_Hz']} "
            f"kappa_ext_over_2pi_Hz={r1['kappa_ext_over_2pi_Hz']} Q_loaded={r1['Q_loaded']}",
            flush=True,
        )
        print(
            f"{label} readout2: f_GHz={r2['f_GHz']} kappa_over_2pi_Hz={r2['kappa_over_2pi_Hz']} "
            f"kappa_i_over_2pi_Hz={r2['kappa_i_over_2pi_Hz']} "
            f"kappa_ext_over_2pi_Hz={r2['kappa_ext_over_2pi_Hz']} Q_loaded={r2['Q_loaded']}",
            flush=True,
        )

    def _save_named_json(filename: str, data: TDict[str, Any]) -> None:
        path = dataset_root / filename
        if path.exists():
            stem = path.stem
            suffix = path.suffix
            for idx in range(1, 10000):
                candidate = dataset_root / f"{stem}_{idx:04d}{suffix}"
                if not candidate.exists():
                    path = candidate
                    break
            else:
                raise FileExistsError(f"No safe non-overwriting name available for {path}")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(base.to_jsonable(data), handle, indent=2, ensure_ascii=False)

    def _num_for_record(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            out = float(value)
        except Exception:
            return None
        return out if math.isfinite(out) else None

    def _q_record_row(payload: TDict[str, Any], qname: str, rname: str, chi_over_key: str) -> TDict[str, Any]:
        q = ((payload.get("qubits") or {}).get(qname) or {})
        r = ((payload.get("resonators") or {}).get(rname) or {})
        g_ghz = (((q.get("dispersive") or {}).get("g_GHz")) if isinstance(q.get("dispersive"), dict) else None)
        return {
            "fr_GHz": _num_for_record(r.get("f_GHz")),
            "fq_GHz": _num_for_record(q.get("f01_epr_GHz")),
            "chi_MHz": _num_for_record(q.get("chi_MHz")),
            "g_MHz": None if _num_for_record(g_ghz) is None else float(cast(float, _num_for_record(g_ghz))) * 1000.0,
            "alpha_MHz": _num_for_record(q.get("alpha_epr_MHz")),
            "kint_Hz": _num_for_record(r.get("kappa_internal_over_2pi_Hz")),
            "kext_Hz": _num_for_record(r.get("kappa_external_over_2pi_Hz")),
            "k_total_Hz": _num_for_record(r.get("kappa_total_over_2pi_Hz")),
            "T1_us": _num_for_record(q.get("T1_est_us")),
            "chi_over_k": _num_for_record((payload.get("chip") or {}).get(chi_over_key)),
        }

    def _changed_record(state: TDict[str, Any]) -> TDict[str, Any]:
        p2r = cast(Layout2QParams, state["p2"])
        return {
            "METAL_2Q_USE_QI_FALLBACK": 1,
            "METAL_2Q_QI_FALLBACK": float(base.QI_FALLBACK),
            "METAL_2Q_DO_Q3D": 1 if bool(do_q3d) else 0,
            "METAL_2Q_HFSS_N_MODES": _env_int("METAL_2Q_HFSS_N_MODES", 10, lo=4, hi=20),
            "METAL_2Q_HFSS_MAX_PASSES": _env_int("METAL_2Q_HFSS_MAX_PASSES", 10, lo=1, hi=25),
            "METAL_2Q_HFSS_MIN_PASSES": _env_int("METAL_2Q_HFSS_MIN_PASSES", 1, lo=1, hi=20),
            "METAL_2Q_HFSS_MIN_CONVERGED": _env_int("METAL_2Q_HFSS_MIN_CONVERGED", 1, lo=1, hi=5),
            "METAL_2Q_HFSS_MIN_FREQ_GHZ": _env_float("METAL_2Q_HFSS_MIN_FREQ_GHZ", 1.0, lo=0.1, hi=20.0),
            "METAL_2Q_HFSS_MAX_DELTA_F_GHZ": _env_float("METAL_2Q_HFSS_MAX_DELTA_F_GHZ", 0.1, lo=0.001, hi=1.0),
            "METAL_2Q_HFSS_PCT_REFINEMENT": _env_int("METAL_2Q_HFSS_PCT_REFINEMENT", 30, lo=1, hi=100),
            "METAL_2Q_HFSS_MAX_MESH_LENGTH_JJ": os.environ.get("METAL_2Q_HFSS_MAX_MESH_LENGTH_JJ", "7um"),
            "METAL_2Q_HFSS_MAX_MESH_LENGTH_PORT": os.environ.get("METAL_2Q_HFSS_MAX_MESH_LENGTH_PORT", "7um"),
            "METAL_2Q_ANSYS_REUSE_PROJECT_FILE": 1 if _env_bool("METAL_2Q_ANSYS_REUSE_PROJECT_FILE", True) else 0,
            "METAL_2Q_ANSYS_PROJECT_FILE": os.environ.get(
                "METAL_2Q_ANSYS_PROJECT_FILE",
                DEFAULT_2Q_ANSYS_PROJECT_FILE,
            ),
            "METAL_2Q_PICK_Q_MIN_GHZ": _env_float("METAL_2Q_PICK_Q_MIN_GHZ", 4.0, lo=0.1, hi=20.0),
        "METAL_2Q_PICK_R_MAX_GHZ": _env_float("METAL_2Q_PICK_R_MAX_GHZ", 6.5, lo=0.1, hi=20.0),
            "METAL_2Q_PICK_R_PM_MAX": _env_float("METAL_2Q_PICK_R_PM_MAX", 0.05, lo=0.0, hi=1.0),
            "Q1_TARGET_KEXT_HZ": float(state["q1_target_kext_hz"]),
            "Q2_TARGET_KEXT_HZ": float(state["q2_target_kext_hz"]),
            "CHIP_SIZE_X_MM": float(CHIP_SIZE_X_MM),
            "CHIP_SIZE_Y_MM": float(CHIP_SIZE_Y_MM),
            "Q1_X_MM": float(state.get("q1_x_mm", Q1_X_MM)),
            "Q1_Y_MM": float(state.get("q1_y_mm", Q1_Y_MM)),
            "Q2_X_MM": float(state.get("q2_x_mm", Q2_X_MM)),
            "Q2_Y_MM": float(state.get("q2_y_mm", Q2_Y_MM)),
            "FEED_L_MM": float(DEFAULT_FEED_L_MM),
            "FEED_SPACING_UM": float(DEFAULT_FEED_SPACING_UM),
            "RO1_L_MM": float(p2r.ro1_total_length_mm),
            "RO2_L_MM": float(p2r.ro2_total_length_mm),
            "LJ1_NH": float(state["lj1_nh"]),
            "LJ2_NH": float(state["lj2_nh"]),
            "CJ2_FF": float(state["cj2_fF"]),
            "Q1_RO_PAD_W_UM": float(p2r.q1_ro_pad_w_um or 0.0),
            "Q1_RO_PAD_H_UM": float(p2r.q1_ro_pad_h_um or 0.0),
            "Q1_RO_PAD_GAP_UM": float(p2r.q1_ro_pad_gap_um or 0.0),
            "Q2_RO_PAD_W_UM": float(p2r.q2_ro_pad_w_um or 0.0),
            "Q2_RO_PAD_H_UM": float(p2r.q2_ro_pad_h_um or 0.0),
            "Q2_RO_PAD_GAP_UM": float(p2r.q2_ro_pad_gap_um or 0.0),
            "RO1_TEE_CAP_GAP_UM": float(p2r.ro1_tee_cap_gap_um or 0.0),
            "RO2_TEE_CAP_GAP_UM": float(p2r.ro2_tee_cap_gap_um or 0.0),
            "RO1_TEE_CAP_WIDTH_UM": float(p2r.ro1_tee_cap_width_um or 0.0),
            "RO2_TEE_CAP_WIDTH_UM": float(p2r.ro2_tee_cap_width_um or 0.0),
            "RO1_TEE_CAP_DISTANCE_UM": float(p2r.ro1_tee_cap_distance_um or 0.0),
            "RO2_TEE_CAP_DISTANCE_UM": float(p2r.ro2_tee_cap_distance_um or 0.0),
            "RO1_TEE_FINGER_LENGTH_UM": int(p2r.ro1_tee_finger_length_um or 0),
            "RO2_TEE_FINGER_LENGTH_UM": int(p2r.ro2_tee_finger_length_um or 0),
            "RO1_TEE_FINGER_COUNT": int(p2r.ro1_tee_finger_count or 0),
            "RO2_TEE_FINGER_COUNT": int(p2r.ro2_tee_finger_count or 0),
            "RO1_PURCELL_STUB_LENGTH_MM": float(p2r.ro1_purcell_stub_length_mm or 0.0),
            "RO2_PURCELL_STUB_LENGTH_MM": float(p2r.ro2_purcell_stub_length_mm or 0.0),
            "RO1_PURCELL_STUB_OFFSET_MM": float(p2r.ro1_purcell_stub_offset_mm or 0.0),
            "RO2_PURCELL_STUB_OFFSET_MM": float(p2r.ro2_purcell_stub_offset_mm or 0.0),
            "RO1_PURCELL_STUB_WIDTH_UM": float(p2r.ro1_purcell_stub_width_um or 0.0),
            "RO2_PURCELL_STUB_WIDTH_UM": float(p2r.ro2_purcell_stub_width_um or 0.0),
            "RO1_PURCELL_STUB_GAP_UM": float(p2r.ro1_purcell_stub_gap_um or 0.0),
            "RO2_PURCELL_STUB_GAP_UM": float(p2r.ro2_purcell_stub_gap_um or 0.0),
            "RO1_DX_MM": float(p2r.ro1_dx_mm),
            "RO1_DY_MM": float(p2r.ro1_dy_mm),
            "RO2_DX_MM": float(p2r.ro2_dx_mm),
            "RO2_DY_MM": float(p2r.ro2_dy_mm),
            "SWAP_TEE_PORTS": 1 if bool(p2r.swap_tee_ports) else 0,
        }

    def _record_metric(q: Optional[TDict[str, Any]]) -> str:
        if not q or q.get("fq_GHz") is None:
            return "no result"

        def fmt(value: Any, digits: int = 3) -> str:
            n = _num_for_record(value)
            return "null" if n is None else f"{n:.{digits}f}"

        return (
            f"fr={fmt(q.get('fr_GHz'))}, fq={fmt(q.get('fq_GHz'))}, chi={fmt(q.get('chi_MHz'))}, "
            f"g={fmt(q.get('g_MHz'))}, alpha={fmt(q.get('alpha_MHz'))}, "
            f"kint={fmt(q.get('kint_Hz'))}, kext={fmt(q.get('kext_Hz'))}, "
            f"T1={fmt(q.get('T1_us'), 4)}, chi/k={fmt(q.get('chi_over_k'))}"
        )

    def _update_2q_run_indexes(records_root: Path) -> None:
        runs: List[TDict[str, Any]] = []
        for path in sorted((records_root / "runs").glob("2Q-RUN-*.json")):
            with path.open("r", encoding="utf-8-sig") as handle:
                runs.append(cast(TDict[str, Any], json.load(handle)))
        (records_root / "RUN_INDEX.json").write_text(
            json.dumps(base.to_jsonable(runs), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        lines = [
            "# 2Q Run Index",
            "",
            "## Units",
            "",
            "- fr and fq: GHz.",
            "- chi, g, and alpha: MHz.",
            "- kint and kext: Hz.",
            "- T1: us.",
            "",
            "## Runs",
            "",
            "| ID | title | status | valid | changed design/run params | Q1 metrics | Q2 metrics | run record | command | source payload |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for run in runs:
            rid = str(run.get("id") or "")
            run_matches = list((records_root / "runs").glob(f"{rid}_*.json"))
            cmd_matches = list((records_root / "commands").glob(f"{rid}_*.ps1"))
            run_file = run_matches[0].name if run_matches else ""
            cmd_file = cmd_matches[0].name if cmd_matches else ""
            changed = ", ".join(f"{k}={v}" for k, v in (run.get("changed") or {}).items())
            valid = run.get("valid")
            valid_s = "n/a" if valid is None else str(valid)
            lines.append(
                f"| {rid} | {run.get('title')} | {run.get('status')} | {valid_s} | {changed} | "
                f"{_record_metric(cast(Optional[TDict[str, Any]], run.get('Q1')))} | "
                f"{_record_metric(cast(Optional[TDict[str, Any]], run.get('Q2')))} | runs/{run_file} | "
                f"commands/{cmd_file} | {run.get('payload_json')} |"
            )
        lines.extend(
            [
                "",
                "## Current Goal Baseline",
                "",
                "2Q-RUN-0060 is the current Q3D-real-kext tuning baseline requested by the user: tee gap 5um, original finger length/counts, HFSS 10/10, Q3D on, target kext disabled.",
            ]
        )
        (records_root / "RUN_INDEX.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _auto_record_run(payload: TDict[str, Any], state: TDict[str, Any], *, title: str, purpose: str) -> None:
        records_root = Path("records") / "2qubit"
        runs_dir = records_root / "runs"
        commands_dir = records_root / "commands"
        runs_dir.mkdir(parents=True, exist_ok=True)
        commands_dir.mkdir(parents=True, exist_ok=True)
        existing_ids: List[int] = []
        for path in runs_dir.glob("2Q-RUN-*.json"):
            try:
                existing_ids.append(int(path.name.split("_", 1)[0].replace("2Q-RUN-", "")))
            except Exception:
                continue
        rid = f"2Q-RUN-{(max(existing_ids) + 1 if existing_ids else 1):04d}"
        safe_title = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in title).strip("_") or "run"
        if len(safe_title) > 140:
            safe_title = safe_title[:140].rstrip("_.-")
        run_path = runs_dir / f"{rid}_{safe_title}.json"
        cmd_path = commands_dir / f"{rid}_{safe_title}.ps1"
        if run_path.exists() or cmd_path.exists():
            raise FileExistsError(f"Auto record path already exists for {rid}_{safe_title}")

        mp = ((payload.get("meta") or {}).get("mode_picker") or {})
        record = {
            "id": rid,
            "title": safe_title,
            "sample": (payload.get("meta") or {}).get("sample_id"),
            "purpose": purpose,
            "command": "python metal_KDD_2qubit.py",
            "changed": _changed_record(state),
            "payload_json": str((payload.get("meta") or {}).get("json_filename") or ""),
            "recorded_at": base.now_utc_iso(),
            "status": payload.get("status"),
            "valid": bool(mp.get("valid")),
            "invalid_reasons": list(mp.get("invalid_reasons") or []),
            "junctions": (payload.get("meta") or {}).get("junctions"),
            "hfss_setup": (payload.get("meta") or {}).get("hfss_setup_used"),
            "layout_key_params": _layout2qparams_to_dict(cast(Layout2QParams, state["p2"])),
            "modes_f_GHz": [
                _num_for_record(x)
                for x in ((((payload.get("resonators") or {}).get("readout1") or {}).get("modes_f_GHz")) or [])
            ],
            "mode_picker": mp,
            "Q1": _q_record_row(payload, "Q1", "readout1", "chi1_over_kappa1"),
            "Q2": _q_record_row(payload, "Q2", "readout2", "chi2_over_kappa2"),
            "hfss_last_pass_delta_GHz": (payload.get("meta") or {}).get("hfss_last_pass_delta_GHz"),
            "errors": {
                "analysis_error": (payload.get("meta") or {}).get("analysis_error"),
                "hfss_error": (payload.get("meta") or {}).get("hfss_error"),
                "mode_picker_error": (payload.get("meta") or {}).get("mode_picker_error"),
                "q3d_error": (payload.get("meta") or {}).get("q3d_error"),
            },
        }
        run_path.write_text(json.dumps(base.to_jsonable(record), indent=2, ensure_ascii=False), encoding="utf-8")
        cmd_path.write_text("python metal_KDD_2qubit.py\n", encoding="utf-8")
        _update_2q_run_indexes(records_root)
        print(f"[Record] Saved {run_path}", flush=True)

    def _save_stage_result(stage_name: str, result: TDict[str, Any]) -> None:
        _save_named_json(
            f"best_{stage_name}.json",
            {
                "stage": stage_name,
                "run_id": result["run_id"],
                "score": result["score"],
                "score_details": result["details"],
                "metrics": _metric_snapshot(cast(TDict[str, Any], result["payload"])),
                "state": _state_to_dict(cast(TDict[str, Any], result["state"])),
                "payload_json_filename": ((cast(TDict[str, Any], result["payload"]).get("meta") or {}).get("json_filename")),
                "payload": result["payload"],
            },
        )

    def _batch_index_path() -> Path:
        return Path("records") / "2qubit" / "batch_near50_position_sweep_index.json"

    def _batch_point_plan(total: int) -> List[TDict[str, float]]:
        qx_values = [-0.0104167, -0.00625, 0.0104167, 0.0145833, 0.01875, 0.0229167]
        qy_values = [2.9791667, 2.9875, 3.0208333, 3.0291667, 3.0375]
        ro_dx_values = [3.175, 3.225]
        points: List[TDict[str, float]] = []
        for ro_dx in ro_dx_values:
            for qysep in qy_values:
                for qxsep in qx_values:
                    points.append(
                        {
                            "qxsep_mm": round(float(qxsep), 7),
                            "qysep_mm": round(float(qysep), 7),
                            "ro_dx_mm": round(float(ro_dx), 7),
                            "q1_x_mm": round(-0.5 * float(qxsep), 7),
                            "q2_x_mm": round(0.5 * float(qxsep), 7),
                            "q1_y_mm": round(0.5 * float(qysep), 7),
                            "q2_y_mm": round(-0.5 * float(qysep), 7),
                        }
                    )
        return points[: int(total)]

    def _load_batch_index(total: int, batch_size: int, points: List[TDict[str, float]]) -> TDict[str, Any]:
        path = _batch_index_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        expected = {
            "name": "batch_near50_position_sweep",
            "total": int(total),
            "batch_size": int(batch_size),
            "cursor": 0,
            "points": points,
            "completed": [],
        }
        if not path.exists():
            path.write_text(json.dumps(base.to_jsonable(expected), indent=2, ensure_ascii=False), encoding="utf-8")
            return expected
        try:
            data = cast(TDict[str, Any], json.loads(path.read_text(encoding="utf-8-sig")))
        except Exception:
            backup = path.with_name(f"{path.stem}_{time.strftime('%Y%m%d_%H%M%S')}.unreadable.json")
            path.rename(backup)
            path.write_text(json.dumps(base.to_jsonable(expected), indent=2, ensure_ascii=False), encoding="utf-8")
            return expected
        if int(data.get("total") or 0) != int(total) or len(data.get("points") or []) != int(total):
            data["total"] = int(total)
            data["batch_size"] = int(batch_size)
            data["points"] = points
            data.setdefault("cursor", 0)
            data.setdefault("completed", [])
        data["batch_size"] = int(batch_size)
        return data

    def _save_batch_index(data: TDict[str, Any]) -> None:
        path = _batch_index_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        data["updated_utc"] = base.now_utc_iso()
        path.write_text(json.dumps(base.to_jsonable(data), indent=2, ensure_ascii=False), encoding="utf-8")

    def _batch_state(base_state: TDict[str, Any], point: TDict[str, float]) -> TDict[str, Any]:
        state = copy.deepcopy(base_state)
        p2_local = _clone_p2(cast(Layout2QParams, state["p2"]))
        p2_local.ro1_dx_mm = float(point["ro_dx_mm"])
        p2_local.ro2_dx_mm = float(point["ro_dx_mm"])
        state["p2"] = p2_local
        state["q1_x_mm"] = float(point["q1_x_mm"])
        state["q1_y_mm"] = float(point["q1_y_mm"])
        state["q2_x_mm"] = float(point["q2_x_mm"])
        state["q2_y_mm"] = float(point["q2_y_mm"])
        return state

    def _batch_sample_id(idx: int, point: TDict[str, float]) -> str:
        def tag(v: float) -> str:
            s = f"{float(v):.7f}".rstrip("0").rstrip(".")
            return s.replace("-", "m").replace(".", "p")

        return (
            f"batchnear50_2q_{idx:03d}"
            f"_qx{tag(point['qxsep_mm'])}"
            f"_qy{tag(point['qysep_mm'])}"
            f"_rdx{tag(point['ro_dx_mm'])}"
        )

    def _run_position_batch(base_state: TDict[str, Any]) -> TDict[str, Any]:
        total = _env_int("METAL_2Q_BATCH_TOTAL", 60, lo=1, hi=10_000)
        batch_size = _env_int("METAL_2Q_BATCH_SIZE", 2, lo=1, hi=20)
        points = _batch_point_plan(total)
        index = _load_batch_index(total, batch_size, points)
        cursor = int(index.get("cursor") or 0)
        completed = list(index.get("completed") or [])
        completed_indices = {int(item.get("index")) for item in completed if isinstance(item, dict) and item.get("index") is not None}
        json_out_dir = dataset_root / (os.environ.get("METAL_2Q_JSON_SUBDIR", "json_tmp").strip().strip("/\\") or "json_tmp")
        gds_out_dir = dataset_root / (os.environ.get("METAL_2Q_GDS_SUBDIR", "gds_tmp").strip().strip("/\\") or "gds_tmp")

        batch_results: List[TDict[str, Any]] = []
        processed = 0
        idx = cursor
        while idx < len(points) and processed < batch_size:
            if idx in completed_indices:
                idx += 1
                continue
            point = cast(TDict[str, float], points[idx])
            run_id = _batch_sample_id(idx, point)
            expected_json = json_out_dir / f"{run_id}.json"
            if expected_json.exists():
                completed.append({"index": idx, "run_id": run_id, "status": "exists", "json": expected_json.name})
                index["completed"] = completed
                index["cursor"] = idx + 1
                _save_batch_index(index)
                idx += 1
                continue
            expected_gds = gds_out_dir / f"{run_id}.gds"
            if expected_gds.exists():
                base_run_id = run_id
                for retry_idx in range(1, 1000):
                    retry_run_id = f"{base_run_id}_retry{retry_idx:03d}"
                    if not (json_out_dir / f"{retry_run_id}.json").exists() and not (gds_out_dir / f"{retry_run_id}.gds").exists():
                        run_id = retry_run_id
                        expected_json = json_out_dir / f"{run_id}.json"
                        break
                else:
                    raise RuntimeError(f"No safe retry run_id available for batch index {idx}")

            state = _batch_state(base_state, point)
            print(
                f"[BatchNear50] running {idx + 1}/{len(points)} run_id={run_id} "
                f"qxsep={point['qxsep_mm']} qysep={point['qysep_mm']} ro_dx={point['ro_dx_mm']}",
                flush=True,
            )
            try:
                payload = _run_state(state, run_id=run_id)
                _print_metric_snapshot(f"[BatchNear50 {idx:03d}]", payload)
                _auto_record_run(
                    payload,
                    state,
                    title=f"batchnear50_{idx:03d}",
                    purpose=(
                        "batch near50 interpolated position sweep around the selected 2Q design range; "
                        "outputs are intentionally written to json_tmp/gds_tmp"
                    ),
                )
                status = str(payload.get("status") or "")
                entry = {
                    "index": idx,
                    "run_id": run_id,
                    "status": status,
                    "json": ((payload.get("meta") or {}).get("json_filename")),
                    "gds": Path(str((payload.get("meta") or {}).get("gds_path") or "")).name,
                    "point": point,
                    "metrics": _metric_snapshot(payload),
                }
                batch_results.append(entry)
                completed.append(entry)
            except Exception as exc:
                entry = {
                    "index": idx,
                    "run_id": run_id,
                    "status": "failed",
                    "error": str(exc),
                    "point": point,
                }
                batch_results.append(entry)
                completed.append(entry)
                print(f"[BatchNear50] failed index={idx} run_id={run_id} error={exc}", flush=True)
            index["completed"] = completed
            index["cursor"] = idx + 1
            _save_batch_index(index)
            processed += 1
            idx += 1

        _save_named_json(
            f"batchnear50_position_sweep_{time.strftime('%Y%m%d_%H%M%S')}.json",
            {
                "batch_size": batch_size,
                "total": len(points),
                "start_cursor": cursor,
                "end_cursor": int(index.get("cursor") or 0),
                "processed_this_run": processed,
                "results": batch_results,
                "index_path": str(_batch_index_path()),
                "json_subdir": os.environ.get("METAL_2Q_JSON_SUBDIR", "json_tmp"),
                "gds_subdir": os.environ.get("METAL_2Q_GDS_SUBDIR", "gds_tmp"),
            },
        )
        print(
            f"[BatchNear50] processed_this_run={processed} cursor={index.get('cursor')}/{len(points)} "
            f"completed={len(completed)}",
            flush=True,
        )
        return {
            "processed": processed,
            "cursor": int(index.get("cursor") or 0),
            "total": len(points),
            "results": batch_results,
        }

    def _assert_baseline_ready(payload: TDict[str, Any]) -> None:
        hard_errors = _hard_error_summary(payload)
        if not hard_errors:
            return
        if any("deterministic_geometry_error" in item for item in hard_errors):
            raise RuntimeError(
                f"Baseline geometry failed; staged search aborted. Details: {'; '.join(hard_errors)}"
            )
        raise RuntimeError(f"Baseline failed; staged search aborted. Details: {'; '.join(hard_errors)}")

    def _run_state(
        state: TDict[str, Any],
        *,
        run_id: str,
    ) -> TDict[str, Any]:
        base.TAN_DELTA_MAIN = float(state["tan_delta_main"])
        design = base.designs.DesignPlanar({}, True)
        populate_design_2qubit(
            design,
            q1_x_mm=float(state.get("q1_x_mm", Q1_X_MM)),
            q1_y_mm=float(state.get("q1_y_mm", Q1_Y_MM)),
            q2_x_mm=float(state.get("q2_x_mm", Q2_X_MM)),
            q2_y_mm=float(state.get("q2_y_mm", Q2_Y_MM)),
            p2=cast(Layout2QParams, state["p2"]),
            lj_vars=("Lj1", "Lj2"),
            cj_vars=("Cj1", "Cj2"),
        )
        return run_analysis_pipeline_2q(
            design,
            run_id,
            root_path=str(dataset_root),
            do_hfss=bool(do_hfss),
            do_q3d=bool(do_q3d),
            lj1_nh=float(state["lj1_nh"]),
            cj1_fF_user=float(state["cj1_fF"]),
            lj2_nh=float(state["lj2_nh"]),
            cj2_fF_user=float(state["cj2_fF"]),
            q1_effective_cin_fF=float(state["q1_effective_cin_fF"]),
            q2_effective_cin_fF=float(state["q2_effective_cin_fF"]),
            q1_target_kext_hz=float(state["q1_target_kext_hz"]),
            q2_target_kext_hz=float(state["q2_target_kext_hz"]),
            save_json=True,
            layout_params=cast(Layout2QParams, state["p2"]),
        )

    def _search_stage(
        stage_name: str,
        start_state: TDict[str, Any],
        *,
        rng: np.random.Generator,
        anchor_payload: Optional[TDict[str, Any]] = None,
    ) -> TDict[str, Any]:
        best: Optional[TDict[str, Any]] = None
        for i in range(int(autotune_tries)):
            candidate_state = start_state if i == 0 else _mutate_state(start_state, rng=rng, stage=stage_name)
            signature = {
                "stage": stage_name,
                "i": int(i),
                "state": _state_to_dict(candidate_state),
            }
            run_tag = base.short_hash_tag(
                json.dumps(base.to_jsonable(signature), ensure_ascii=True, separators=(",", ":")),
                n=8,
            )
            run_id = f"{sample_id}__{stage_name}__try{i:04d}__{run_tag}"
            try:
                payload = _run_state(candidate_state, run_id=run_id)
                score, details = _score_payload(payload, stage=stage_name, anchor_payload=anchor_payload)
                print(
                    f"[Stage {stage_name} Try {i+1}/{autotune_tries}] pen={score:.6f} run_id={run_id} details={details}",
                    flush=True,
                )
            except Exception as exc:
                payload = build_chip_summary_multiq(sample_id=run_id)
                payload["status"] = "failed"
                payload["meta"]["exception"] = str(exc)
                score = 1.0e12
                details = {"stage": stage_name, "exception": str(exc)}
                print(
                    f"[Stage {stage_name} Try {i+1}/{autotune_tries}] failed run_id={run_id} error={exc}",
                    flush=True,
                )

            if best is None or float(score) < float(best["score"]):
                best = {
                    "stage": stage_name,
                    "run_id": run_id,
                    "score": float(score),
                    "details": details,
                    "payload": payload,
                    "state": candidate_state,
                }

        if best is None:
            raise RuntimeError(f"Stage {stage_name} produced no candidates.")

        print(
            f"[Stage {stage_name}] best_pen={best['score']:.6f} best_run_id={best['run_id']} best_details={best['details']}",
            flush=True,
        )
        return best

    sample_id = os.environ.get("METAL_2Q_SAMPLE_ID", "chip_2q_example").strip() or "chip_2q_example"
    try:
        rng = np.random.default_rng(int(autotune_seed) or None)
        current_state: TDict[str, Any] = {
            "p2": p2,
            "q1_x_mm": float(Q1_X_MM),
            "q1_y_mm": float(Q1_Y_MM),
            "q2_x_mm": float(Q2_X_MM),
            "q2_y_mm": float(Q2_Y_MM),
            "lj1_nh": float(LJ1_NH),
            "cj1_fF": float(CJ1_FF),
            "lj2_nh": float(LJ2_NH),
            "cj2_fF": float(CJ2_FF),
            "tan_delta_main": float(TAN_DELTA_MAIN),
            "q1_effective_cin_fF": float(Q1_EFFECTIVE_CIN_FF),
            "q2_effective_cin_fF": float(Q2_EFFECTIVE_CIN_FF),
            "q1_target_kext_hz": float(Q1_TARGET_KEXT_HZ),
            "q2_target_kext_hz": float(Q2_TARGET_KEXT_HZ),
        }

        if _env_bool("METAL_2Q_BATCH_POSITION_SWEEP", False):
            batch_result = _run_position_batch(current_state)
            print(f"[Done] batch_position_sweep result={batch_result}", flush=True)
            raise SystemExit(0)

        baseline_run_id = f"{sample_id}__baseline"
        baseline_payload = _run_state(current_state, run_id=baseline_run_id)
        _assert_baseline_ready(baseline_payload)
        _print_metric_snapshot("[Baseline]", baseline_payload)
        _auto_record_run(
            baseline_payload,
            current_state,
            title=f"auto_{DEFAULT_SAMPLE_PREFIX}",
            purpose=(
                "auto-recorded baseline run from metal_KDD_2qubit.py while tuning Q3D real kext toward "
                "single-digit MHz with fr<7, fq>=3.8, chi>1, and g near 100 when feasible"
            ),
        )
        _save_named_json(
            "baseline_metrics.json",
            {
                "run_id": baseline_run_id,
                "metrics": _metric_snapshot(baseline_payload),
                "state": _state_to_dict(current_state),
                "payload_json_filename": ((baseline_payload.get("meta") or {}).get("json_filename")),
                "payload": baseline_payload,
            },
        )

        current_payload = baseline_payload
        final_run_id = baseline_run_id
        for stage_name in stage_plan_map[opt_stage]:
            result = _search_stage(stage_name, current_state, rng=rng, anchor_payload=current_payload)
            _save_stage_result(stage_name, result)
            current_state = cast(TDict[str, Any], result["state"])
            current_payload = cast(TDict[str, Any], result["payload"])
            final_run_id = str(result["run_id"])
            _print_metric_snapshot(f"[Stage {stage_name} best]", current_payload)

        print(
            f"[Done] opt_stage={opt_stage} final_run_id={final_run_id} final_metrics={_metric_snapshot(current_payload)}",
            flush=True,
        )
    finally:
        base._mpl_cleanup()
