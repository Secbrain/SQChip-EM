# -*- coding: utf-8 -*-
"""8-qubit dataset generator (qiskit-metal + pyEPR).

Layout: 2x4 grid arrangement
    Q1 --- Q2 --- Q3 --- Q4
    |      |      |      |
    Q5 --- Q6 --- Q7 --- Q8

Bus couplers:
    Horizontal: Q1-Q2, Q2-Q3, Q3-Q4, Q5-Q6, Q6-Q7, Q7-Q8
    Vertical: Q1-Q5, Q2-Q6, Q3-Q7, Q4-Q8
Each qubit has its own readout resonator.

Sweep parameters: dx_h (horizontal spacing), dy_v (vertical spacing)
"""

from __future__ import annotations

import os
import csv
import json
import math
import time
import gc
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Dict as TDict, List, Optional, Tuple

import numpy as np
import pandas as pd

import metal_KDD as base
from qiskit_metal.qlibrary.qubits.transmon_pocket_6 import TransmonPocket6
from qiskit_metal.qlibrary.tlines.straight_path import RouteStraight


# -------------------------
# layout
# -------------------------
@dataclass
class Layout8QParams:
    """8Q layout settings."""

    shared: base.LayoutParams = None

    bus_width: str = "10um"
    bus_gap: str = "6um"
    bus_lead_start: str = "250um"
    bus_lead_end: str = "250um"
    bus_fillet: str = "50um"
    bus_spacing: str = "250um"
    bus_total_length: str = "5.0mm"

    def __post_init__(self):
        if self.shared is None:
            self.shared = base.LayoutParams()


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
    readout_loc_W: int = +1,
    readout_loc_H: int = +1,
):
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
                pad_width=p.ro_pad_w,
                pad_height=p.ro_pad_h,
                pad_gap=p.ro_pad_gap,
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


def _get_pin_frame(design, component: str, pin: str) -> Tuple[float, float, float, float, float, float]:
    """Return (x, y, nx, ny, tx, ty) for a pin.

    Values are in mm in the planar design coordinate system.
    """
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
):
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
            trace_width=p.prime_width,
            trace_gap=p.prime_gap,
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

    base.RouteMeander(
        design,
        f"RO_FEED{suffix}",
        options=base.Dict(
            total_length=p.feed_total_length,
            fillet=p.feed_fillet,
            trace_width=p.prime_width,
            trace_gap=p.prime_gap,
            meander=base.Dict(spacing=p.feed_spacing),
            lead=base.Dict(start_straight=p.feed_lead_start, end_straight=p.feed_lead_end),
            pin_inputs=base.Dict(
                start_pin=base.Dict(component=lp.name, pin="tie"),
                end_pin=base.Dict(component=tee.name, pin=prime_pin),
            ),
        ),
        type="CPW",
    )

    base.sanitize_center_readout_pin(design, qname, q_pin)
    RouteStraight(
        design,
        f"RO_RES{suffix}",
        options=base.Dict(
            fillet=p.ro_fillet,
            trace_width=p.second_width,
            trace_gap=p.second_gap,
            lead=base.Dict(start_straight=p.ro_lead_start, end_straight=p.ro_lead_end),
            pin_inputs=base.Dict(
                start_pin=base.Dict(component=tee.name, pin="second_end"),
                end_pin=base.Dict(component=qname, pin=q_pin),
            ),
        ),
    )

    return lp, tee


def _add_bus(
    design,
    name: str,
    q_start: str,
    pin_start: str,
    q_end: str,
    pin_end: str,
    p8: Layout8QParams,
    *,
    asymmetry: Optional[str] = None,
    lead_start: Optional[str] = None,
    lead_end: Optional[str] = None,
):
    """Add a bus coupler between two qubit pads."""
    meander = base.Dict(spacing=p8.bus_spacing)
    if asymmetry is not None:
        meander["asymmetry"] = asymmetry
    lead = base.Dict(
        start_straight=(lead_start if lead_start is not None else p8.bus_lead_start),
        end_straight=(lead_end if lead_end is not None else p8.bus_lead_end),
    )
    base.RouteMeander(
        design,
        name,
        options=base.Dict(
            total_length=p8.bus_total_length,
            fillet=p8.bus_fillet,
            trace_width=p8.bus_width,
            trace_gap=p8.bus_gap,
            meander=meander,
            lead=lead,
            pin_inputs=base.Dict(
                start_pin=base.Dict(component=q_start, pin=pin_start),
                end_pin=base.Dict(component=q_end, pin=pin_end),
            ),
        ),
        type="CPW",
    )


def populate_design_8qubit(
    design,
    *,
    dx_h_mm: float = 3.0,
    dy_v_mm: float = 3.0,
    p8: Layout8QParams,
    lj_vars: Tuple[str, ...] = ("Lj1", "Lj2", "Lj3", "Lj4", "Lj5", "Lj6", "Lj7", "Lj8"),
    cj_vars: Tuple[str, ...] = ("Cj1", "Cj2", "Cj3", "Cj4", "Cj5", "Cj6", "Cj7", "Cj8"),
) -> None:
    """Populate planar design with 8 qubits in a 2x4 grid.

    Layout:
        Q1 --- Q2 --- Q3 --- Q4   (top row, y = +dy_v/2)
        |      |      |      |
        Q5 --- Q6 --- Q7 --- Q8   (bottom row, y = -dy_v/2)

    Bus couplers:
        Horizontal: Q1-Q2, Q2-Q3, Q3-Q4, Q5-Q6, Q6-Q7, Q7-Q8
        Vertical: Q1-Q5, Q2-Q6, Q3-Q7, Q4-Q8

    Readout pads:
        Q1: top-left, Q2: top, Q3: top, Q4: top-right
        Q5: bottom-left, Q6: bottom, Q7: bottom, Q8: bottom-right
    """

    design.overwrite_enabled = True
    p = p8.shared

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

    half_dy = dy_v_mm / 2.0

    # Positions: 2x4 grid centered at origin
    # Top row: Q1, Q2, Q3, Q4 at y = +half_dy
    # Bottom row: Q5, Q6, Q7, Q8 at y = -half_dy
    # X positions: -1.5*dx_h, -0.5*dx_h, +0.5*dx_h, +1.5*dx_h
    positions = {
        "Q1": (-1.5 * dx_h_mm, +half_dy),
        "Q2": (-0.5 * dx_h_mm, +half_dy),
        "Q3": (+0.5 * dx_h_mm, +half_dy),
        "Q4": (+1.5 * dx_h_mm, +half_dy),
        "Q5": (-1.5 * dx_h_mm, -half_dy),
        "Q6": (-0.5 * dx_h_mm, -half_dy),
        "Q7": (+0.5 * dx_h_mm, -half_dy),
        "Q8": (+1.5 * dx_h_mm, -half_dy),
    }

    # Q1 (top-left): readout top-left, bus pads: right (to Q2), down (to Q5)
    _make_transmon(
        design, "Q1", x_mm=positions["Q1"][0], y_mm=positions["Q1"][1],
        p=p, lj_var=lj_vars[0], cj_var=cj_vars[0],
        bus_pads=[
            ("bus_right", +1, -1),  # toward Q2
            ("bus_down", -1, -1),   # toward Q5
        ],
        readout_loc_W=-1, readout_loc_H=+1,
    )

    # Q2 (top, 2nd from left): readout top, bus pads: left (to Q1), right (to Q3), down (to Q6)
    _make_transmon(
        design, "Q2", x_mm=positions["Q2"][0], y_mm=positions["Q2"][1],
        p=p, lj_var=lj_vars[1], cj_var=cj_vars[1],
        bus_pads=[
            ("bus_left", -1, -1),   # toward Q1
            ("bus_right", +1, -1),  # toward Q3
            ("bus_down", 0, -1),    # toward Q6
        ],
        readout_loc_W=0, readout_loc_H=+1,
    )

    # Q3 (top, 3rd from left): readout top, bus pads: left (to Q2), right (to Q4), down (to Q7)
    _make_transmon(
        design, "Q3", x_mm=positions["Q3"][0], y_mm=positions["Q3"][1],
        p=p, lj_var=lj_vars[2], cj_var=cj_vars[2],
        bus_pads=[
            ("bus_left", -1, -1),   # toward Q2
            ("bus_right", +1, -1),  # toward Q4
            ("bus_down", 0, -1),    # toward Q7
        ],
        readout_loc_W=0, readout_loc_H=+1,
    )

    # Q4 (top-right): readout top-right, bus pads: left (to Q3), down (to Q8)
    _make_transmon(
        design, "Q4", x_mm=positions["Q4"][0], y_mm=positions["Q4"][1],
        p=p, lj_var=lj_vars[3], cj_var=cj_vars[3],
        bus_pads=[
            ("bus_left", -1, -1),   # toward Q3
            ("bus_down", +1, -1),   # toward Q8
        ],
        readout_loc_W=+1, readout_loc_H=+1,
    )

    # Q5 (bottom-left): readout bottom-left, bus pads: right (to Q6), up (to Q1)
    _make_transmon(
        design, "Q5", x_mm=positions["Q5"][0], y_mm=positions["Q5"][1],
        p=p, lj_var=lj_vars[4], cj_var=cj_vars[4],
        bus_pads=[
            ("bus_right", +1, +1),  # toward Q6
            ("bus_up", -1, +1),     # toward Q1
        ],
        readout_loc_W=-1, readout_loc_H=-1,
    )

    # Q6 (bottom, 2nd from left): readout bottom, bus pads: left (to Q5), right (to Q7), up (to Q2)
    _make_transmon(
        design, "Q6", x_mm=positions["Q6"][0], y_mm=positions["Q6"][1],
        p=p, lj_var=lj_vars[5], cj_var=cj_vars[5],
        bus_pads=[
            ("bus_left", -1, +1),   # toward Q5
            ("bus_right", +1, +1),  # toward Q7
            ("bus_up", 0, +1),      # toward Q2
        ],
        readout_loc_W=0, readout_loc_H=-1,
    )

    # Q7 (bottom, 3rd from left): readout bottom, bus pads: left (to Q6), right (to Q8), up (to Q3)
    _make_transmon(
        design, "Q7", x_mm=positions["Q7"][0], y_mm=positions["Q7"][1],
        p=p, lj_var=lj_vars[6], cj_var=cj_vars[6],
        bus_pads=[
            ("bus_left", -1, +1),   # toward Q6
            ("bus_right", +1, +1),  # toward Q8
            ("bus_up", 0, +1),      # toward Q3
        ],
        readout_loc_W=0, readout_loc_H=-1,
    )

    # Q8 (bottom-right): readout bottom-right, bus pads: left (to Q7), up (to Q4)
    _make_transmon(
        design, "Q8", x_mm=positions["Q8"][0], y_mm=positions["Q8"][1],
        p=p, lj_var=lj_vars[7], cj_var=cj_vars[7],
        bus_pads=[
            ("bus_left", -1, +1),   # toward Q7
            ("bus_up", +1, +1),     # toward Q4
        ],
        readout_loc_W=+1, readout_loc_H=-1,
    )

    design.rebuild()

    # Readout chains
    tee_offset = 0.45
    # Keep RO_TEE bodies aligned to the readout pad normal (2Q-style),
    # to avoid a rotated/diagonal RO_RES segment.

    for i in range(1, 9):
        q = f"Q{i}"
        px, py, nx, ny, tx, ty = _get_pin_frame(design, q, "readout")
        ori = _tee_orientation_from_readout_normal(nx, ny)
        cx = px + nx * tee_offset
        cy = py + ny * tee_offset

        # Keep launchpads on top/bottom edges (like the existing 8Q style).
        lp_dx = float(getattr(p, "lp_dx", 0.0) or 0.0)
        chip_y = base._value_to_mm(getattr(p, "chip_size_y", 0.0))
        half_y = (chip_y / 2.0) if chip_y else 0.0
        edge_margin_mm = 0.5
        lp_dir = 1 if float(cy) >= 0 else -1
        if half_y > 0:
            max_dy = (half_y - edge_margin_mm) - abs(float(cy))
            if max_dy > 0:
                lp_dx = min(lp_dx, float(max_dy))
        if lp_dx <= 0:
            lp_dx = float(getattr(p, "lp_dx", 0.0) or 0.0)

        lp_x = float(cx)
        lp_y = float(cy) + float(lp_dir) * float(lp_dx)

        prime_pin_override = None
        if int(float(ori)) % 360 in (0, 180):
            # For horizontal prime pins, connect on the outer x-side.
            prime_pin_override = "prime_start" if float(cx) < 0 else "prime_end"

        _add_readout_chain(
            design,
            suffix=str(i),
            qname=q,
            q_pin="readout",
            coupler_x_mm=cx,
            coupler_y_mm=cy,
            p=p,
            tee_orientation=ori,
            lp_pos_x_mm=lp_x,
            lp_pos_y_mm=lp_y,
            lp_orientation=None,
            prime_pin_override=prime_pin_override,
            lp_direction=lp_dir,
            lp_dx_mm=lp_dx,
        )

    # Bus couplers
    # Horizontal (top row)
    _add_bus(design, "Bus_12", "Q1", "bus_right", "Q2", "bus_left", p8)
    _add_bus(design, "Bus_23", "Q2", "bus_right", "Q3", "bus_left", p8)
    _add_bus(design, "Bus_34", "Q3", "bus_right", "Q4", "bus_left", p8)
    # Horizontal (bottom row)
    _add_bus(design, "Bus_56", "Q5", "bus_right", "Q6", "bus_left", p8)
    _add_bus(design, "Bus_67", "Q6", "bus_right", "Q7", "bus_left", p8)
    _add_bus(design, "Bus_78", "Q7", "bus_right", "Q8", "bus_left", p8)
    # Vertical
    _add_bus(
        design,
        "Bus_15",
        "Q1",
        "bus_down",
        "Q5",
        "bus_up",
        p8,
    )
    _add_bus(
        design,
        "Bus_26",
        "Q2",
        "bus_down",
        "Q6",
        "bus_up",
        p8,
    )
    _add_bus(
        design,
        "Bus_37",
        "Q3",
        "bus_down",
        "Q7",
        "bus_up",
        p8,
    )
    _add_bus(
        design,
        "Bus_48",
        "Q4",
        "bus_down",
        "Q8",
        "bus_up",
        p8,
    )

    design.rebuild()


# -------------------------
# payload
# -------------------------
NUM_Q = 8


def _empty_readout_block() -> TDict[str, Any]:
    return {
        "f_GHz": None, "K_MHz": None, "modes_f_GHz": None,
        "picked_qubit_mode_index": None, "picked_res_mode_index": None,
        "Qi": None, "kappa_i_over_2pi_Hz": None,
        "external": {
            "Cin_fF": None, "pair": None, "res_node": None, "feed_node": None,
            "kappa_Hz": None, "kappa_over_2pi_Hz": None, "Qe": None,
        },
        "kappa_over_2pi_Hz": None, "Q_loaded": None, "warnings": [],
        "T1_photon_us": None, "kappa_i_over_kappa_e": None,
    }


def _empty_qubit_block() -> TDict[str, Any]:
    return {
        "Lj_H": None, "Cj_fF": None, "C_eff_fF": None,
        "f01_epr_GHz": None, "alpha_epr_MHz": None, "chi_MHz": None,
        "dispersive": {"chi_GHz": None, "Delta_GHz": None, "g_GHz": None},
        "energies": {"Ec_GHz": None, "Ej_GHz": None, "Ej_over_Ec": None},
        "f01_transmon_GHz": None, "Q_dielectric_main": None,
        "T1_dielectric_us": None, "T1_purcell_us": None,
        "T1_est_us": None, "T2_est_us": None,
    }


def build_chip_summary_8q(*, sample_id: str) -> TDict[str, Any]:
    chip = {}
    for i in range(1, NUM_Q + 1):
        chip[f"T1_qubit{i}_est_us"] = None
        chip[f"T2_qubit{i}_est_us"] = None

    chip.update(
        {
            # Horizontal couplings (top row)
            "chi12_MHz": None,
            "chi23_MHz": None,
            "chi34_MHz": None,
            # Horizontal couplings (bottom row)
            "chi56_MHz": None,
            "chi67_MHz": None,
            "chi78_MHz": None,
            # Vertical couplings
            "chi15_MHz": None,
            "chi26_MHz": None,
            "chi37_MHz": None,
            "chi48_MHz": None,
            # Cross couplings (diagonal, for reference)
            "chi16_MHz": None,
            "chi25_MHz": None,
            "chi27_MHz": None,
            "chi36_MHz": None,
            "chi38_MHz": None,
            "chi47_MHz": None,
        }
    )

    for i in range(1, NUM_Q + 1):
        chip[f"chi{i}_over_kappa{i}"] = None

    return {
        "meta": {
            "created_utc": base.now_utc_iso(), "updated_utc": None,
            "sample_id": sample_id,
            "units": {"f": "GHz", "kappa": "Hz", "C": "fF", "K": "MHz", "T": "us"},
        },
        "resonators": {f"readout{i}": _empty_readout_block() for i in range(1, NUM_Q+1)},
        "qubits": {f"Q{i}": _empty_qubit_block() for i in range(1, NUM_Q+1)},
        "q3d": {"internal": {}, "external": {}},
        "chip": chip,
        "status": "init",
    }


def _estimate_T1T2(payload: TDict[str, Any]) -> None:
    for i in range(1, NUM_Q+1):
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
                k_over_2pi = float(kappa_over_2pi)
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


def postprocess_8q(payload: TDict[str, Any]) -> TDict[str, Any]:
    payload["meta"]["updated_utc"] = base.now_utc_iso()
    for i in range(1, NUM_Q+1):
        _postprocess_one_readout(payload["resonators"][f"readout{i}"])
        _postprocess_one_qubit(payload["qubits"][f"Q{i}"])

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

    for i in range(1, NUM_Q + 1):
        payload["chip"][f"chi{i}_over_kappa{i}"] = _chi_over_kappa(
            payload["qubits"][f"Q{i}"],
            payload["resonators"][f"readout{i}"],
        )

    _estimate_T1T2(payload)
    return payload


# -------------------------
# mode picking
# -------------------------
def _pick_n_qubits_and_n_resonators(chi_MHz: np.ndarray, n_q: int) -> Tuple[List[int], List[int]]:
    """Heuristic mode picker for N-qubit system."""
    chi = np.array(chi_MHz, dtype=float)
    n = chi.shape[0]
    diag = np.abs(np.diag(chi))
    if n < 2 * n_q:
        raise ValueError(f"Need >={2*n_q} modes for {n_q}Q+{n_q}R picking, got n={n}")

    q_sorted = list(np.argsort(-diag))
    idx_q = [int(q_sorted[i]) for i in range(n_q)]

    remaining = [i for i in range(n) if i not in idx_q]

    def score_res(qi: int, ri: int) -> float:
        return abs(float(chi[qi, ri])) / (1e-9 + abs(float(chi[ri, ri])))

    idx_r = []
    for qi in idx_q:
        r = max(remaining, key=lambda r: score_res(qi, r))
        idx_r.append(int(r))
        remaining = [i for i in remaining if i != r]

    return idx_q, idx_r


def _calc_g_from_chi(*, chi_MHz: float, Delta_GHz: float, alpha_MHz: float) -> Optional[float]:
    try:
        Delta_MHz = float(Delta_GHz) * 1e3
        if abs(float(alpha_MHz)) <= 1e-9:
            return None
        g_MHz = math.sqrt(abs(float(chi_MHz) * Delta_MHz * (Delta_MHz + float(alpha_MHz)) / float(alpha_MHz)))
        return g_MHz / 1e3
    except Exception:
        return None


def _qi_for_mode(res: dict, idx_mode: int, *, warnings: List[str]) -> Optional[float]:
    Qi_val = None

    p_val = base.extract_participation(res, idx_mode, kind="dielectrics_bulk", name="main")
    Qi_from_p = base.qi_from_p_tandelta(p_val, base.TAN_DELTA_MAIN)
    if Qi_from_p is not None:
        Qi_val = float(Qi_from_p)
        warnings.append(f"Qi from p*tan_delta: p={float(p_val):.3g}, Qi={Qi_val:.3g}")

    if Qi_val is None:
        Qi_raw = base.extract_qdielectric_main(res, idx_mode)
        if Qi_raw is not None and np.isfinite(Qi_raw) and Qi_raw > 0:
            p_inferred = 1.0 / (float(Qi_raw) * float(base.EPR_TAN_DELTA_ASSUMED))
            if 0.0001 <= p_inferred <= 0.99:
                Qi_val = 1.0 / (p_inferred * float(base.TAN_DELTA_MAIN))
            else:
                Qi_val = float(Qi_raw)

    if Qi_val is None and base.USE_QI_FALLBACK:
        Qi_val = float(base.QI_FALLBACK)

    if Qi_val is not None and base.QI_CLAMP_TO_FALLBACK and base.USE_QI_FALLBACK and Qi_val > float(base.QI_FALLBACK):
        Qi_val = float(base.QI_FALLBACK)

    return float(Qi_val) if Qi_val is not None else None


def _qdielectric_for_mode(res: dict, idx_mode: int) -> Optional[float]:
    p_val = base.extract_participation(res, idx_mode, kind="dielectrics_bulk", name="main")
    q_from_p = base.qi_from_p_tandelta(p_val, base.TAN_DELTA_MAIN)
    if q_from_p is not None:
        return float(q_from_p)
    q_raw = base.extract_qdielectric_main(res, idx_mode)
    if q_raw is not None and np.isfinite(q_raw) and q_raw > 0:
        return float(q_raw)
    return None


def _configure_n_junctions_best_effort(eig, *, n_q: int, lj_vars: Tuple, cj_vars: Tuple) -> None:
    try:
        pinfo = eig.sim.renderer.pinfo
    except Exception:
        return

    try:
        if hasattr(eig, "del_junction") and hasattr(eig, "add_junction"):
            eig.del_junction()
            for i in range(n_q):
                jj_name = f"jj{i+1}"
                qname = f"Q{i+1}"
                patterns = [
                    (f"JJ_rect_Lj_{qname}_rect_jj", f"JJ_Lj_{qname}_rect_jj_"),
                    (f"JJ_rect_{lj_vars[i]}_{qname}_rect_jj", f"JJ_{lj_vars[i]}_{qname}_rect_jj_"),
                ]
                for rect, line in patterns:
                    try:
                        eig.add_junction(jj_name, lj_vars[i], cj_vars[i], rect=rect, line=line)
                        try:
                            pinfo.validate_junction_info()
                        except Exception:
                            pass
                        break
                    except Exception:
                        continue
    except Exception:
        pass


# -------------------------
# analysis
# -------------------------
def run_analysis_pipeline_8q(
    design,
    sample_id: str,
    *,
    root_path: str,
    do_hfss: bool,
    do_q3d: bool,
    lj_nh: Tuple[float, ...],
    cj_fF_user: Tuple[float, ...],
    inputs_sweep: Optional[TDict[str, Any]] = None,
    inputs_qubits: Optional[TDict[str, Any]] = None,
) -> TDict[str, Any]:
    root = Path(root_path)
    (root / "gds").mkdir(parents=True, exist_ok=True)
    (root / "json").mkdir(parents=True, exist_ok=True)

    payload = build_chip_summary_8q(sample_id=sample_id)
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

    lj_vars = tuple(f"Lj{i}" for i in range(1, NUM_Q+1))
    cj_vars = tuple(f"Cj{i}" for i in range(1, NUM_Q+1))

    # 1) GDS
    try:
        gds_path = root / "gds" / f"{sample_id}.gds"
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
                target = workdir / f"EIG8Q_{sim_tag}_a{attempt}.aedt"

                print(f"[HFSS] attempt={attempt} Starting HFSS...", flush=True)
                base._ansys_best_effort_reset()
                hfss.start()
                time.sleep(0.8)

                base._ansys_prepare_project(hfss, target)

                setup_vars = base.Dict()
                for i in range(NUM_Q):
                    setup_vars[f"Lj{i+1}"] = f"{float(lj_nh[i])} nH"
                    setup_vars[f"Cj{i+1}"] = f"{float(cj_fF_user[i])} fF"

                eig.sim.setup.vars = setup_vars
                # Keep the HFSS solve light for smoke/deployment runs.
                eig.sim.setup.n_modes = 2 * NUM_Q
                eig.sim.setup.max_passes = 8
                eig.sim.setup.min_freq_ghz = 1.0

                all_components = list(design.components.keys())
                print(f"[HFSS] Running simulation with {len(all_components)} components", flush=True)
                eig.sim.run(
                    name=f"Eig8Q_{sim_tag}_a{attempt}",
                    components=all_components,
                    open_terminations=[],
                    box_plus_buffer=True,
                )

                _configure_n_junctions_best_effort(eig, n_q=NUM_Q, lj_vars=lj_vars, cj_vars=cj_vars)

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
                for i in range(1, NUM_Q+1):
                    payload["resonators"][f"readout{i}"]["modes_f_GHz"] = base.to_jsonable(freqs)

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

                idx_q, idx_r = _pick_n_qubits_and_n_resonators(chi_matrix, NUM_Q)

                for i in range(NUM_Q):
                    qname = f"Q{i+1}"
                    roname = f"readout{i+1}"
                    ro = payload["resonators"][roname]
                    q = payload["qubits"][qname]

                    f_q = float(freqs[idx_q[i]])
                    f_r = float(freqs[idx_r[i]])
                    alpha_MHz = -abs(float(chi_matrix[idx_q[i], idx_q[i]]))
                    chi_qr_MHz = float(chi_matrix[idx_q[i], idx_r[i]])
                    K_MHz = float(chi_matrix[idx_r[i], idx_r[i]])
                    Delta = f_q - f_r
                    g_GHz = _calc_g_from_chi(chi_MHz=chi_qr_MHz, Delta_GHz=Delta, alpha_MHz=alpha_MHz)

                    ro["f_GHz"] = f_r
                    ro["K_MHz"] = K_MHz
                    ro["picked_qubit_mode_index"] = idx_q[i]
                    ro["picked_res_mode_index"] = idx_r[i]

                    q["f01_epr_GHz"] = f_q
                    q["alpha_epr_MHz"] = alpha_MHz
                    q["chi_MHz"] = chi_qr_MHz
                    q["dispersive"]["chi_GHz"] = chi_qr_MHz / 1e3
                    q["dispersive"]["Delta_GHz"] = Delta
                    q["dispersive"]["g_GHz"] = g_GHz

                    ro_w = ro.get("warnings") or []
                    Qi = _qi_for_mode(res, idx_r[i], warnings=ro_w)
                    ro["Qi"] = Qi
                    ro["kappa_i_over_2pi_Hz"] = (f_r * 1e9) / float(Qi) if Qi else None

                    qQ = _qdielectric_for_mode(res, idx_q[i])
                    if qQ is not None:
                        q["Q_dielectric_main"] = float(qQ)

                payload["coupling"] = base.build_coupling_pairs(
                    chi_matrix_MHz=chi_matrix,
                    freqs_GHz=freqs,
                    idx_qubits=idx_q,
                    idx_readouts=idx_r,
                    qubit_names=[f"Q{i}" for i in range(1, NUM_Q + 1)],
                    readout_names=[f"readout{i}" for i in range(1, NUM_Q + 1)],
                )

                # Qubit-qubit cross-chi
                # idx: Q1=0, Q2=1, Q3=2, Q4=3, Q5=4, Q6=5, Q7=6, Q8=7
                # Horizontal (top row)
                payload["chip"]["chi12_MHz"] = float(chi_matrix[idx_q[0], idx_q[1]])
                payload["chip"]["chi23_MHz"] = float(chi_matrix[idx_q[1], idx_q[2]])
                payload["chip"]["chi34_MHz"] = float(chi_matrix[idx_q[2], idx_q[3]])
                # Horizontal (bottom row)
                payload["chip"]["chi56_MHz"] = float(chi_matrix[idx_q[4], idx_q[5]])
                payload["chip"]["chi67_MHz"] = float(chi_matrix[idx_q[5], idx_q[6]])
                payload["chip"]["chi78_MHz"] = float(chi_matrix[idx_q[6], idx_q[7]])
                # Vertical
                payload["chip"]["chi15_MHz"] = float(chi_matrix[idx_q[0], idx_q[4]])
                payload["chip"]["chi26_MHz"] = float(chi_matrix[idx_q[1], idx_q[5]])
                payload["chip"]["chi37_MHz"] = float(chi_matrix[idx_q[2], idx_q[6]])
                payload["chip"]["chi48_MHz"] = float(chi_matrix[idx_q[3], idx_q[7]])
                # Cross (diagonal)
                payload["chip"]["chi16_MHz"] = float(chi_matrix[idx_q[0], idx_q[5]])
                payload["chip"]["chi25_MHz"] = float(chi_matrix[idx_q[1], idx_q[4]])
                payload["chip"]["chi27_MHz"] = float(chi_matrix[idx_q[1], idx_q[6]])
                payload["chip"]["chi36_MHz"] = float(chi_matrix[idx_q[2], idx_q[5]])
                payload["chip"]["chi38_MHz"] = float(chi_matrix[idx_q[2], idx_q[7]])
                payload["chip"]["chi47_MHz"] = float(chi_matrix[idx_q[3], idx_q[6]])

                # store sol columns for debugging participation extraction
                try:
                    sol = res.get("sol", None)
                    if isinstance(sol, pd.DataFrame):
                        payload["meta"]["epr_sol_columns"] = [str(c) for c in sol.columns]
                except Exception:
                    pass

                break

            except Exception as e:
                payload["meta"]["hfss_error"] = str(e)
                try:
                    details = base._format_com_error(e)
                except Exception:
                    details = ""
                if details:
                    print(f"[HFSS] attempt={attempt} Error: {e} {details}", flush=True)
                else:
                    print(f"[HFSS] attempt={attempt} Error: {e}", flush=True)
                base._print_ansys_messages(locals().get("hfss"), "HFSS", level=2)
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
    Ceff = {i: None for i in range(1, NUM_Q+1)}
    cj_fF = {i: float(cj_fF_user[i-1]) for i in range(1, NUM_Q+1)}

    if do_q3d:
        for attempt in [1, 2]:
            lom = None
            try:
                lom = base.LOManalysis(design, "q3d")
                q3d = lom.sim.renderer

                if q3d is None:
                    raise RuntimeError("Q3D renderer is not available.")

                workdir = base._ansys_short_workdir(root)
                target = workdir / f"Q3D8Q_{sim_tag}_a{attempt}.aedt"

                print(f"[Q3D] attempt={attempt} Starting Q3D...", flush=True)
                base._ansys_best_effort_reset()
                q3d.start()
                time.sleep(0.8)

                base._ansys_prepare_project(q3d, target)

                try:
                    lom.sim.renderer_initialized = True
                except Exception:
                    pass

                lom.sim.setup.freq_ghz = 5.0
                lom.sim.setup.max_passes = 6
                print(f"[Q3D] Running simulation...", flush=True)
                lom.sim.run(name=f"Q3D8Q_{sim_tag}_a{attempt}", components=list(design.components.keys()))

                Cmat = lom.sim.capacitance_matrix

                def _ceff_for_qubit(qname: str, cj_fF_local: float) -> Optional[float]:
                    pt = f"pad_top_{qname}"
                    pb = f"pad_bot_{qname}"
                    if pt in Cmat.index and pb in Cmat.index:
                        C2 = Cmat.loc[[pt, pb], [pt, pb]].values.astype(float)
                        C_mode_fF = base.diff_mode_ceff_from_2x2_maxwell_fF(C2)
                        if C_mode_fF is not None and np.isfinite(C_mode_fF):
                            return float(C_mode_fF) + float(cj_fF_local)
                    return None

                for i in range(1, NUM_Q+1):
                    Ceff[i] = _ceff_for_qubit(f"Q{i}", cj_fF[i])

                payload["q3d"]["internal"]["capacitances_fF"] = {
                    "units": "fF",
                    "nodes": list(Cmat.index),
                    "capacitance_matrix_fF": base.to_jsonable(Cmat),
                }

                cin = {}
                for i in range(1, NUM_Q+1):
                    cin[i] = base.pick_cin_prefer_tee_bodies(Cmat, tee_name=f"RO_TEE{i}")
                    payload["resonators"][f"readout{i}"]["external"].update(cin[i])

                payload["q3d"]["external"] = {
                    "units": "fF",
                    **{f"readout{i}": cin[i] for i in range(1, NUM_Q+1)},
                }

                from qiskit_metal.analyses.em.kappa_calculation import kappa_in

                for i in range(1, NUM_Q+1):
                    ro = payload["resonators"][f"readout{i}"]
                    if cin[i].get("Cin_fF") and ro.get("f_GHz"):
                        fr_hz = float(ro["f_GHz"]) * 1e9
                        Cin_F = float(cin[i]["Cin_fF"]) * 1e-15
                        ke_over_2pi_hz = float(kappa_in(fr_hz, Cin_F, fr_hz))
                        ro["external"]["kappa_over_2pi_Hz"] = ke_over_2pi_hz
                        ro["external"]["kappa_Hz"] = 2.0 * math.pi * ke_over_2pi_hz
                        ro["external"]["Qe"] = fr_hz / ke_over_2pi_hz if ke_over_2pi_hz > 0 else None

                break

            except Exception as e:
                payload["meta"]["q3d_error"] = str(e)
                print(f"[Q3D] attempt={attempt} Error: {e}", flush=True)
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

    for i in range(1, NUM_Q+1):
        payload["qubits"][f"Q{i}"]["Lj_H"] = float(lj_nh[i-1]) * 1e-9
        payload["qubits"][f"Q{i}"]["Cj_fF"] = float(cj_fF[i])
        payload["qubits"][f"Q{i}"]["C_eff_fF"] = Ceff[i]

    payload = postprocess_8q(payload)
    payload["status"] = "completed"

    json_path = Path(root_path) / "json" / f"{sample_id}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(base.to_jsonable(base.export_payload_v3(payload)), f, indent=2, ensure_ascii=False)

    return payload


# -------------------------
# main
# -------------------------
if __name__ == "__main__":
    dataset_root = Path("./data/sqchip_em_8q")
    dataset_root.mkdir(parents=True, exist_ok=True)

    # Junction params (same for all qubits)
    LJ_NH = tuple([12.0] * NUM_Q)
    # Match Qiskit Metal / pyEPR common default: ignore JJ capacitance in EM (use Cj=0).
    CJ_FF = tuple([0.0] * NUM_Q)

    TEE_FINGER_LENGTH_UM = 160
    TEE_FINGER_COUNT = 12
    TEE_CAP_GAP_UM = 1.2
    TEE_CAP_WIDTH_UM = 18.0
    TEE_CAP_DISTANCE_UM = 40.0

    RO_L_MM = 8.5
    Q_RO_PAD_W_UM = 125
    Q_RO_PAD_H_UM = 30
    Q_RO_PAD_GAP_UM = 15

    # Sweep parameters: horizontal and vertical spacing between adjacent qubits
    DX_H_LIST = np.linspace(2.5, 4.0, 7)   # horizontal spacing (mm)
    DY_V_LIST = np.linspace(2.5, 4.0, 7)   # vertical spacing (mm)

    RUN_ONE_EXAMPLE = True
    MAX_SAMPLES = 1 if RUN_ONE_EXAMPLE else None
    RESUME_FROM_SUMMARY = False if RUN_ONE_EXAMPLE else True

    csv_headers = [
        "sample_id",
        "dx_h_mm", "dy_v_mm",
        "fq1_GHz", "fq2_GHz", "fq3_GHz", "fq4_GHz", "fq5_GHz", "fq6_GHz", "fq7_GHz", "fq8_GHz",
        "fr1_GHz", "fr2_GHz", "fr3_GHz", "fr4_GHz", "fr5_GHz", "fr6_GHz", "fr7_GHz", "fr8_GHz",
        "chi12_MHz", "chi23_MHz", "chi34_MHz", "chi56_MHz", "chi67_MHz", "chi78_MHz",
        "chi15_MHz", "chi26_MHz", "chi37_MHz", "chi48_MHz",
        "status", "error",
    ]

    summary_csv_path = dataset_root / "summary_8q.csv"
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

    lp_shared = base.LayoutParams(
        chip_size_x="25mm",
        chip_size_y="18mm",
        cpw_width="10um",
        cpw_gap="6um",
        feed_fillet="25um",
        ro_fillet="25um",
        # Avoid RouteStraight lead backtracking that can create self-overlapping polylines.
        ro_lead_start="80um",
        ro_lead_end="80um",
        ro_pad_w=f"{Q_RO_PAD_W_UM}um",
        ro_pad_h=f"{Q_RO_PAD_H_UM}um",
        ro_pad_gap=f"{Q_RO_PAD_GAP_UM}um",
        ro_total_length=f"{RO_L_MM}mm",
        finger_length=f"{TEE_FINGER_LENGTH_UM}um",
        finger_count=str(int(TEE_FINGER_COUNT)),
        cap_gap=f"{TEE_CAP_GAP_UM}um",
        cap_width=f"{TEE_CAP_WIDTH_UM}um",
        cap_distance=f"{TEE_CAP_DISTANCE_UM}um",
    )

    sample_count = 0
    for dx_h, dy_v in product(DX_H_LIST, DY_V_LIST):
        if MAX_SAMPLES is not None and sample_count >= MAX_SAMPLES:
            break

        sample_id = (
            "chip_8q_"
            f"dxh{base.fmt_mm_id(float(dx_h))}_dyv{base.fmt_mm_id(float(dy_v))}"
            f"__Lj{str(float(LJ_NH[0])).replace('.', 'p')}nH_Cj{str(float(CJ_FF[0])).replace('.', 'p')}fF"
            f"__roL{str(float(RO_L_MM)).replace('.', 'p')}mm"
            f"__tee_fl{int(TEE_FINGER_LENGTH_UM)}um_fc{int(TEE_FINGER_COUNT)}_cg{str(float(TEE_CAP_GAP_UM)).replace('.', 'p')}um"
            f"__rpg{str(float(Q_RO_PAD_GAP_UM)).replace('.', 'p')}um"
        )

        if sample_id in existing_ids:
            print(f"[SKIP] {sample_id} already exists")
            continue

        # Keep bus compact; too much extra length makes meanders sprawl.
        bus_length_mm = max(4.0, max(dx_h, dy_v) + 0.8)

        p8 = Layout8QParams(
            shared=lp_shared,
            bus_total_length=f"{bus_length_mm:.2f}mm",
        )

        try:
            print(f"\n[START] {sample_id} dx_h={dx_h}, dy_v={dy_v}", flush=True)
            design = base.designs.DesignPlanar({}, True)
            populate_design_8qubit(
                design,
                dx_h_mm=dx_h,
                dy_v_mm=dy_v,
                p8=p8,
            )

            result = run_analysis_pipeline_8q(
                design,
                sample_id,
                root_path=str(dataset_root),
                do_hfss=True,
                do_q3d=True,
                lj_nh=LJ_NH,
                cj_fF_user=CJ_FF,
                inputs_sweep={"dx_h_mm": float(dx_h), "dy_v_mm": float(dy_v)},
                inputs_qubits={
                    "Q1": {"x_mm": float(-1.5 * dx_h), "y_mm": float(+dy_v / 2.0)},
                    "Q2": {"x_mm": float(-0.5 * dx_h), "y_mm": float(+dy_v / 2.0)},
                    "Q3": {"x_mm": float(+0.5 * dx_h), "y_mm": float(+dy_v / 2.0)},
                    "Q4": {"x_mm": float(+1.5 * dx_h), "y_mm": float(+dy_v / 2.0)},
                    "Q5": {"x_mm": float(-1.5 * dx_h), "y_mm": float(-dy_v / 2.0)},
                    "Q6": {"x_mm": float(-0.5 * dx_h), "y_mm": float(-dy_v / 2.0)},
                    "Q7": {"x_mm": float(+0.5 * dx_h), "y_mm": float(-dy_v / 2.0)},
                    "Q8": {"x_mm": float(+1.5 * dx_h), "y_mm": float(-dy_v / 2.0)},
                },
            )

            chip = result.get("chip", {})
            qubits = result.get("qubits", {})
            resonators = result.get("resonators", {})

            out = dict(
                sample_id=sample_id,
                dx_h_mm=dx_h,
                dy_v_mm=dy_v,
                fq1_GHz=qubits.get("Q1", {}).get("f01_epr_GHz"),
                fq2_GHz=qubits.get("Q2", {}).get("f01_epr_GHz"),
                fq3_GHz=qubits.get("Q3", {}).get("f01_epr_GHz"),
                fq4_GHz=qubits.get("Q4", {}).get("f01_epr_GHz"),
                fq5_GHz=qubits.get("Q5", {}).get("f01_epr_GHz"),
                fq6_GHz=qubits.get("Q6", {}).get("f01_epr_GHz"),
                fq7_GHz=qubits.get("Q7", {}).get("f01_epr_GHz"),
                fq8_GHz=qubits.get("Q8", {}).get("f01_epr_GHz"),
                fr1_GHz=resonators.get("readout1", {}).get("f_GHz"),
                fr2_GHz=resonators.get("readout2", {}).get("f_GHz"),
                fr3_GHz=resonators.get("readout3", {}).get("f_GHz"),
                fr4_GHz=resonators.get("readout4", {}).get("f_GHz"),
                fr5_GHz=resonators.get("readout5", {}).get("f_GHz"),
                fr6_GHz=resonators.get("readout6", {}).get("f_GHz"),
                fr7_GHz=resonators.get("readout7", {}).get("f_GHz"),
                fr8_GHz=resonators.get("readout8", {}).get("f_GHz"),
                chi12_MHz=chip.get("chi12_MHz"),
                chi23_MHz=chip.get("chi23_MHz"),
                chi34_MHz=chip.get("chi34_MHz"),
                chi56_MHz=chip.get("chi56_MHz"),
                chi67_MHz=chip.get("chi67_MHz"),
                chi78_MHz=chip.get("chi78_MHz"),
                chi15_MHz=chip.get("chi15_MHz"),
                chi26_MHz=chip.get("chi26_MHz"),
                chi37_MHz=chip.get("chi37_MHz"),
                chi48_MHz=chip.get("chi48_MHz"),
                status=result.get("status", ""),
                error=(
                    result.get("meta", {}).get("hfss_error", "")
                    or result.get("meta", {}).get("q3d_error", "")
                    or result.get("meta", {}).get("gds_error", "")
                    or ""
                ),
            )
            write_row(out)
            sample_count += 1

        except Exception as e:
            out = dict(
                sample_id=sample_id,
                dx_h_mm=dx_h, dy_v_mm=dy_v,
                status="ERROR", error=str(e),
            )
            write_row(out)
            sample_count += 1
        finally:
            base._mpl_cleanup()

    print(f"\n[Done] 8Q dataset generation completed. Total samples: {sample_count}")
