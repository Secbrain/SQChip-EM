#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#######
# -*- coding: utf-8 -*-
# Headless guard: qiskit-metal/pyEPR may trigger matplotlib GUI windows.
# Default: disable GUI popups during long parameter sweeps.
import os
import sys


def _ensure_utf8_stdio() -> None:
    """Prevent UnicodeEncodeError on Windows consoles (e.g. GBK code page).

    pyEPR/Ansys logs may include Unicode symbols; ensure stdout/stderr can emit them.
    """

    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


_ensure_utf8_stdio()

_DISABLE_MPL_GUI = os.environ.get("METAL_DISABLE_MPL_GUI", "1") != "0"
if _DISABLE_MPL_GUI:
    # Must be set BEFORE any library imports matplotlib.
    os.environ.setdefault("MPLBACKEND", "Agg")
    # Best-effort: if matplotlib gets imported anyway, keep it non-interactive.
    try:
        import matplotlib  # type: ignore

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as _plt  # type: ignore

        _plt.ioff()
        def _noop_show(*args, **kwargs):
            _plt.close("all")
        _plt.show = _noop_show
        _plt.pause = lambda *args, **kwargs: None
    except Exception:
        pass


def _mpl_cleanup() -> None:
    if not _DISABLE_MPL_GUI:
        return
    try:
        import matplotlib.pyplot as _plt  # type: ignore
        _plt.close("all")
    except Exception:
        pass

import csv
import json
import math
import re
import time
import tempfile
import hashlib
import gc
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any, Dict as TDict, List, Optional

import numpy as np
import pandas as pd

from qiskit_metal import Dict, designs
from qiskit_metal.analyses.quantization import EPRanalysis, LOManalysis
from qiskit_metal.qlibrary.couplers.cap_n_interdigital_tee import CapNInterdigitalTee
from qiskit_metal.qlibrary.qubits.transmon_pocket import TransmonPocket
from qiskit_metal.qlibrary.tlines.meandered import RouteMeander

try:
    from qiskit_metal.qlibrary.terminations.launchpad_wb import LaunchpadWirebond
    _LAUNCHPAD_CLS = LaunchpadWirebond
except Exception:
    from qiskit_metal.qlibrary.terminations.launchpad_wb_coupled import LaunchpadWirebondCoupled
    _LAUNCHPAD_CLS = LaunchpadWirebondCoupled


# -------------------------
# knobs
# -------------------------
# GDS锛氬鏋滀綘鍙槸瑕佽窇浠跨湡/鍙傛暟鎵紝寤鸿鐩存帴 SAFE锛岄伩锟?gdstk 甯冨皵杩愮畻鍋跺彂锟?GDS_FORCE_SAFE_MODE = True
GDS_DISABLE_CHEESE = True

# 鐩爣锛氳 魏i/2蟺 ~ 1 MHz锛坒r~7.1GHz => Qi ~ 7.1e3锟?# 寤鸿锛氬埆锟?fallback 甯告暟锛涗笅闈㈡垜浠紭鍏堢敤 participation * tan未 璁＄畻 Qi
QI_FALLBACK = float(os.environ.get("METAL_QI_FALLBACK", "2500.0"))  # 鏈€鍚庡厹搴曪紙浣犲鏋滃畬鍏ㄤ笉鎯崇敤锛屽彲锟?USE_QI_FALLBACK=False锟?USE_QI_FALLBACK = True

# 鏄惁鎶婁粠 EPR 鍙栧埌锟?Qi 闄愬埗锟?QI_FALLBACK 浠ュ唴
QI_CLAMP_TO_FALLBACK = False

# 浠嬭川鎹熻€楋紙褰卞搷 kappa_i锛夈€傚彲閫氳繃鐜鍙橀噺瑕嗙洊锛歁ETAL_TAN_DELTA_MAIN
# 鍏稿瀷鍊硷細锟?~1e-4, 钃濆疂锟?~4e-4
TAN_DELTA_MAIN = float(os.environ.get("METAL_TAN_DELTA_MAIN", "4e-4"))

# 濡傛灉 EPR 娌℃湁杈撳嚭 participation 鍒楋紝灏辩敤杩欎釜鍋囧畾鍊煎弽锟?p锟?# 鍙€氳繃鐜鍙橀噺瑕嗙洊锛歁ETAL_EPR_TAN_DELTA_ASSUMED
EPR_TAN_DELTA_ASSUMED = float(os.environ.get("METAL_EPR_TAN_DELTA_ASSUMED", "4e-4"))

_E_CHARGE = 1.602176634e-19
_H_PLANCK = 6.62607015e-34
_PHI0 = _H_PLANCK / (2 * _E_CHARGE)


# -------------------------
# utils
# -------------------------
def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def to_jsonable(x: Any) -> Any:
    if isinstance(x, (pd.DataFrame, pd.Series)):
        return x.to_dict()
    if hasattr(x, "tolist"):
        return x.tolist()
    if hasattr(x, "item"):
        return x.item()
    if isinstance(x, complex):
        return {"re": float(x.real), "im": float(x.imag)}
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [to_jsonable(v) for v in x]
    return x


_QR_KEY_RE_1 = re.compile(r"^Q(?P<qidx>\d+)readout(?P<ridx>\d+)$")
_QR_KEY_RE_2 = re.compile(r"^Q(?P<qidx>\d+)R(?P<ridx>\d+)$")


def _export_qr_diagonal(qr: Any) -> TDict[str, Any]:
    """Keep only direct Qn<->Rn pairs and rename keys to QnRn."""

    if not isinstance(qr, dict):
        return {}
    out: TDict[str, Any] = {}
    for k, v in qr.items():
        ks = str(k)
        m = _QR_KEY_RE_1.match(ks) or _QR_KEY_RE_2.match(ks)
        if not m:
            continue
        try:
            qidx = int(m.group("qidx"))
            ridx = int(m.group("ridx"))
        except Exception:
            continue
        if qidx != ridx:
            continue
        out[f"Q{qidx}R{ridx}"] = v
    return out


def export_payload_v3(payload: Any) -> TDict[str, Any]:
    """Export payload in a stable, dataset-friendly schema.

    Top-level key order:
      meta, position, qubit, resonators, q3d, coupling, chip, status

    - Preserve `meta` exactly (no field deletion).
    - Rename `inputs`/`positions` -> `position`.
    - Rename `qubits` -> `qubit`.
    - Move `resonators.*.external` under `q3d.external.resonators.*`.
    - Keep Q3D external info under `q3d.external`.
    - Reduce coupling to only {QQ, QR}, and in QR keep only diagonal Qn<->Rn
      pairs (Qnreadoutn) renamed as QnRn.
    """

    if not isinstance(payload, dict):
        return {
            "meta": {},
            "position": {},
            "qubit": {},
            "resonators": {},
            "q3d": {},
            "coupling": {},
            "chip": {},
            "status": None,
        }

    meta_val = payload.get("meta")
    meta: TDict[str, Any] = meta_val if isinstance(meta_val, dict) else {}

    # Inputs/positions
    position: TDict[str, Any] = {}
    for k in ("position", "inputs", "positions"):
        v = payload.get(k)
        if isinstance(v, dict):
            position = v
            break

    # Qubits
    qubit_val = payload.get("qubit")
    if not isinstance(qubit_val, dict):
        qubit_val = payload.get("qubits")
    qubit: TDict[str, Any] = qubit_val if isinstance(qubit_val, dict) else {}

    # Resonators (strip embedded external)
    resonators_val = payload.get("resonators")
    resonators_in: TDict[str, Any] = resonators_val if isinstance(resonators_val, dict) else {}
    resonators_out: TDict[str, Any] = {}
    ext_res: TDict[str, Any] = {}
    for rname, rdata in resonators_in.items():
        if isinstance(rdata, dict):
            ext = rdata.get("external")
            if isinstance(ext, dict):
                ext_res[str(rname)] = ext
            resonators_out[str(rname)] = {kk: vv for kk, vv in rdata.items() if kk != "external"}
        else:
            resonators_out[str(rname)] = rdata

    # Q3D (strip embedded external)
    q3d_val = payload.get("q3d")
    q3d_in: TDict[str, Any] = q3d_val if isinstance(q3d_val, dict) else {}
    q3d_out: TDict[str, Any] = {kk: vv for kk, vv in q3d_in.items() if kk != "external"}
    q3d_ext = q3d_in.get("external") if isinstance(q3d_in, dict) else None

    # Backward-compat: accept previous v3 that stored external at top-level.
    ext_top = payload.get("external")
    if isinstance(ext_top, dict):
        if not ext_res and isinstance(ext_top.get("resonators"), dict):
            ext_res.update({str(k): v for k, v in ext_top["resonators"].items()})
        if not isinstance(q3d_ext, dict) and isinstance(ext_top.get("q3d"), dict):
            q3d_ext = ext_top["q3d"]

    # Attach external info under q3d.
    q3d_external: TDict[str, Any] = {}
    if isinstance(q3d_ext, dict):
        q3d_external.update(q3d_ext)
    if ext_res:
        q3d_external["resonators"] = ext_res
    # Keep schema stable: always include q3d.external (possibly empty).
    q3d_out["external"] = q3d_external

    # Coupling reduced.
    # Note: for 1-qubit datasets, keep coupling present but empty.
    coupling_val = payload.get("coupling")
    coupling_in: TDict[str, Any] = coupling_val if isinstance(coupling_val, dict) else {}
    qq_val = coupling_in.get("QQ")
    qq: TDict[str, Any] = qq_val if isinstance(qq_val, dict) else {}
    qr: TDict[str, Any] = _export_qr_diagonal(coupling_in.get("QR"))
    if len(qubit) <= 1:
        coupling_out: TDict[str, Any] = {}
    else:
        coupling_out = {"QQ": qq, "QR": qr}

    chip_val = payload.get("chip")
    chip: TDict[str, Any] = chip_val if isinstance(chip_val, dict) else {}
    status = payload.get("status")

    # Stable top-level order
    out: TDict[str, Any] = {}
    out["meta"] = meta
    out["position"] = position
    out["qubit"] = qubit
    out["resonators"] = resonators_out
    out["q3d"] = q3d_out
    out["coupling"] = coupling_out
    out["chip"] = chip
    out["status"] = status
    return out


def fmt_mm_id(x_mm: float, ndigits: int = 3) -> str:
    s = f"{x_mm:.{ndigits}f}"
    return s.replace("-", "m").replace(".", "p")


def short_hash_tag(sample_id: str, n: int = 8) -> str:
    """鍙粰 Ansys 鐢紝灏介噺鐭紝閬垮厤璺緞杩囬暱"""
    return hashlib.md5(sample_id.encode("utf-8")).hexdigest()[:n]


def short_tag(sample_id: str, max_len: int = 40) -> str:
    """浣犲師鏉ョ殑锛氱敤浜庢枃浠跺悕/璁板綍"""
    h = hashlib.md5(sample_id.encode("utf-8")).hexdigest()[:10]
    base = sample_id
    if len(base) > max_len - 11:
        base = base[: max_len - 11]
    base = "".join([c if (c.isalnum() or c in "_-") else "_" for c in base])
    return f"{base}_{h}"


def calc_g_over_2pi_MHz(*, chi_MHz: float, Delta_GHz: float, alpha_MHz: float) -> float:
    """Compute g/2pi in MHz from chi, Delta, alpha (all in frequency units).

    Returns a non-negative float. If inputs are degenerate, returns 0.0.
    """

    try:
        chi = float(chi_MHz)
        Delta_MHz = float(Delta_GHz) * 1e3
        alpha = float(alpha_MHz)

        if not np.isfinite(chi) or not np.isfinite(Delta_MHz) or not np.isfinite(alpha):
            return 0.0
        if abs(alpha) < 1e-12:
            return 0.0

        val = chi * Delta_MHz * (Delta_MHz + alpha) / alpha
        if not np.isfinite(val):
            return 0.0

        g_MHz = math.sqrt(abs(float(val)))
        if not np.isfinite(g_MHz):
            return 0.0
        return float(g_MHz)
    except Exception:
        return 0.0


def build_couplings_matrices(
    *,
    chi_matrix_MHz: np.ndarray,
    freqs_GHz: np.ndarray,
    idx_qubits: List[int],
    idx_readouts: List[int],
    qubit_names: Optional[List[str]] = None,
    readout_names: Optional[List[str]] = None,
) -> TDict[str, Any]:
    """Build scalable coupling matrices for N qubits and M readout modes.

    chi_matrix_MHz: full mode-mode chi (MHz)
    freqs_GHz: full eigenmode frequencies (GHz)
    idx_qubits: indices into mode list for qubit-like modes (len=N)
    idx_readouts: indices into mode list for readout-like modes (len=M)
    """

    chi = np.asarray(chi_matrix_MHz, dtype=float)
    f = np.asarray(freqs_GHz, dtype=float)

    iq = [int(i) for i in list(idx_qubits)]
    ir = [int(i) for i in list(idx_readouts)]
    nq = len(iq)
    nr = len(ir)

    qnames = qubit_names if qubit_names is not None else [f"Q{i+1}" for i in range(nq)]
    rnames = readout_names if readout_names is not None else [f"readout{i+1}" for i in range(nr)]

    chi_qq = chi[np.ix_(iq, iq)] if nq else np.zeros((0, 0), dtype=float)
    chi_qr = chi[np.ix_(iq, ir)] if (nq and nr) else np.zeros((nq, nr), dtype=float)

    fq = f[iq] if nq else np.zeros((0,), dtype=float)
    fr = f[ir] if nr else np.zeros((0,), dtype=float)
    Delta_qr = (fq.reshape((nq, 1)) - fr.reshape((1, nr))) if (nq and nr) else np.zeros((nq, nr), dtype=float)

    alpha_q = np.array([-abs(float(chi[i, i])) for i in iq], dtype=float) if nq else np.zeros((0,), dtype=float)
    g_qr = np.zeros((nq, nr), dtype=float)
    for i in range(nq):
        for j in range(nr):
            g_qr[i, j] = calc_g_over_2pi_MHz(
                chi_MHz=float(chi_qr[i, j]),
                Delta_GHz=float(Delta_qr[i, j]),
                alpha_MHz=float(alpha_q[i]),
            )

    return {
        "qubits": list(qnames),
        "readouts": list(rnames),
        "mode_index": {
            "qubits": {str(qnames[i]): int(iq[i]) for i in range(nq)},
            "readouts": {str(rnames[j]): int(ir[j]) for j in range(nr)},
        },
        "f_qubits_GHz": to_jsonable(fq),
        "f_readouts_GHz": to_jsonable(fr),
        "alpha_qubits_MHz": to_jsonable(alpha_q),
        "chi_qq_MHz": to_jsonable(chi_qq),
        "chi_qr_MHz": to_jsonable(chi_qr),
        "Delta_qr_GHz": to_jsonable(Delta_qr),
        "g_qr_over_2pi_MHz": to_jsonable(g_qr),
    }


def _finite_float(x: Any, default: float = 0.0) -> float:
    """Convert to finite float (no NaN/inf/None)."""

    try:
        v = float(x)
        if not np.isfinite(v):
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def build_coupling_pairs(
    *,
    chi_matrix_MHz: np.ndarray,
    freqs_GHz: np.ndarray,
    idx_qubits: List[int],
    idx_readouts: List[int],
    qubit_names: Optional[List[str]] = None,
    readout_names: Optional[List[str]] = None,
) -> TDict[str, Any]:
    """Build human-friendly couplings keyed like Q1Q2 / Q1readout1.

    - QQ: only i<j pairs
    - QR: all qubit-readout pairs

    All numeric fields are finite floats (no null/NaN/inf).
    """

    chi = np.asarray(chi_matrix_MHz, dtype=float)
    f = np.asarray(freqs_GHz, dtype=float)

    iq = [int(i) for i in list(idx_qubits)]
    ir = [int(i) for i in list(idx_readouts)]
    nq = len(iq)
    nr = len(ir)

    qnames = qubit_names if qubit_names is not None else [f"Q{i+1}" for i in range(nq)]
    rnames = readout_names if readout_names is not None else [f"readout{i+1}" for i in range(nr)]

    # Slices
    chi_qq = chi[np.ix_(iq, iq)] if nq else np.zeros((0, 0), dtype=float)
    chi_qr = chi[np.ix_(iq, ir)] if (nq and nr) else np.zeros((nq, nr), dtype=float)
    fq = f[iq] if nq else np.zeros((0,), dtype=float)
    fr = f[ir] if nr else np.zeros((0,), dtype=float)

    # Derived
    Delta_qr = (fq.reshape((nq, 1)) - fr.reshape((1, nr))) if (nq and nr) else np.zeros((nq, nr), dtype=float)
    alpha_q = np.array([-abs(float(chi[i, i])) for i in iq], dtype=float) if nq else np.zeros((0,), dtype=float)

    out: TDict[str, Any] = {
        "qubits": list(qnames),
        "readouts": list(rnames),
        "mode_index": {
            "qubits": {str(qnames[i]): int(iq[i]) for i in range(nq)},
            "readouts": {str(rnames[j]): int(ir[j]) for j in range(nr)},
        },
        "QQ": {},
        "QR": {},
    }

    qq: TDict[str, Any] = {}
    for i in range(nq):
        for j in range(i + 1, nq):
            key = f"{qnames[i]}{qnames[j]}"
            qq[key] = {
                "chi_MHz": _finite_float(chi_qq[i, j], 0.0),
                "Delta_GHz": _finite_float(float(fq[i]) - float(fq[j]), 0.0),
            }
    out["QQ"] = qq

    qr: TDict[str, Any] = {}
    for i in range(nq):
        for j in range(nr):
            key = f"{qnames[i]}{rnames[j]}"
            chi_MHz = _finite_float(chi_qr[i, j], 0.0)
            Delta_GHz = _finite_float(Delta_qr[i, j], 0.0)
            g_MHz = calc_g_over_2pi_MHz(chi_MHz=chi_MHz, Delta_GHz=Delta_GHz, alpha_MHz=_finite_float(alpha_q[i], 0.0))
            qr[key] = {
                "chi_MHz": chi_MHz,
                "Delta_GHz": Delta_GHz,
                "g_over_2pi_MHz": _finite_float(g_MHz, 0.0),
            }
    out["QR"] = qr

    return out


def ec_over_h_hz_from_c(C_F: float) -> Optional[float]:
    if C_F is None or C_F <= 0:
        return None
    return (_E_CHARGE * _E_CHARGE) / (2.0 * C_F * _H_PLANCK)


def ej_over_h_hz_from_lj(Lj_H: float) -> Optional[float]:
    if Lj_H is None or Lj_H <= 0:
        return None
    return ((_PHI0 / (2.0 * math.pi)) ** 2) / (Lj_H * _H_PLANCK)


def transmon_f01_hz(EJ_over_h_hz: float, EC_over_h_hz: float) -> Optional[float]:
    if EJ_over_h_hz is None or EC_over_h_hz is None:
        return None
    if EJ_over_h_hz <= 0 or EC_over_h_hz <= 0:
        return None
    return math.sqrt(8.0 * EJ_over_h_hz * EC_over_h_hz) - EC_over_h_hz


def _safe_us(x_s: Optional[float]) -> Optional[float]:
    if x_s is None:
        return None
    try:
        x = float(x_s)
        if not np.isfinite(x) or x <= 0:
            return None
        return x * 1e6
    except Exception:
        return None


def _extract_mode_scalar(res: dict, mode_index: int, keys: List[str]) -> Optional[float]:
    """
    鏀硅繘锛氭敮锟?MultiIndex锛坧yEPR 甯歌 Lj->mode 锟?MultiIndex锟?    """
    for k in keys:
        if k not in res:
            continue
        obj = res.get(k)
        try:
            if isinstance(obj, pd.Series):
                if isinstance(obj.index, pd.MultiIndex):
                    try:
                        xs = obj.xs(mode_index, level=-1)
                        return float(np.array(xs.values).ravel()[0])
                    except Exception:
                        pass
                if mode_index in obj.index:
                    return float(obj.loc[mode_index])
                return float(obj.iloc[mode_index])

            if isinstance(obj, pd.DataFrame):
                if isinstance(obj.index, pd.MultiIndex):
                    try:
                        xs = obj.xs(mode_index, level=-1)
                        return float(np.array(xs.values).ravel()[0])
                    except Exception:
                        pass
                if mode_index in obj.index:
                    row = obj.loc[mode_index]
                    return float(np.array(getattr(row, "values", row)).ravel()[0])
                row = obj.iloc[mode_index]
                return float(np.array(row.values).ravel()[0])

            if isinstance(obj, (int, float, np.number)):
                return float(obj)

        except Exception:
            pass
    return None


def extract_qdielectric_main(res: dict, mode_index: int) -> Optional[float]:
    """
    鍏煎涓ょ缁撴瀯锟?      1) res 椤跺眰鐩存帴锟?Qdielectric_main/Qdielectric/Q_diel
      2) res['sol'] DataFrame 閲屾湁锟?Qdielectric_main/Qdielectric/Q_diel
    """
    v = _extract_mode_scalar(res, mode_index, keys=["Qdielectric_main", "Qdielectric", "Q_diel"])
    try:
        if v is not None:
            v = float(v)
            if np.isfinite(v) and v > 0:
                return v
    except Exception:
        pass

    sol = res.get("sol", None)
    if isinstance(sol, pd.DataFrame):
        for col in ["Qdielectric_main", "Qdielectric", "Q_diel"]:
            if col not in sol.columns:
                continue
            try:
                if mode_index in sol.index:
                    v2 = sol.loc[mode_index, col]
                else:
                    v2 = sol.iloc[int(mode_index)][col]
                v2 = float(v2)
                if np.isfinite(v2) and v2 > 0:
                    return v2
            except Exception:
                pass

    return None


def extract_participation(
    res: dict,
    mode_index: int,
    *,
    kind: str = "dielectrics_bulk",
    name: str = "main",
) -> Optional[float]:
    """
    锟?pyEPR results 閲屾姄 participation p锛岀敤 p*tan未 锟?Qi锟?    涓昏锟?res['sol'] DataFrame 鐨勫垪閲屽仛鈥滄ā绯婂尮閰嶁€濓拷?
    浣犺窇閫氬悗鍙互鎵撳嵃 res['sol'].columns锛岃繘涓€姝ユ妸鍖归厤瑙勫垯鏀剁揣锟?    """
    sol = res.get("sol", None)
    if isinstance(sol, pd.DataFrame):
        cols = [str(c) for c in sol.columns]
        kind_l = kind.lower()
        name_l = name.lower()

        # 鍏堢簿涓€鐐癸細鍒楀悕閲屽悓鏃跺寘锟?kind + name锛屽苟涓旂湅璧锋潵锟?participation
        candidates = []
        for c in cols:
            lc = c.lower()
            if (kind_l in lc) and (name_l in lc) and (("participation" in lc) or lc.startswith("p") or "_p" in lc):
                candidates.append(c)

        # 鍐嶅鏉惧厹锟?        if not candidates:
            for c in cols:
                lc = c.lower()
                if (kind_l in lc) and (name_l in lc) and ("p" in lc):
                    candidates.append(c)

        for c in candidates:
            try:
                v = sol.loc[mode_index, c] if mode_index in sol.index else sol.iloc[int(mode_index)][c]
                v = float(v)
                if np.isfinite(v) and v > 0:
                    return v
            except Exception:
                pass

    return None


def qi_from_p_tandelta(p: Optional[float], tan_delta: float) -> Optional[float]:
    try:
        if p is None:
            return None
        p = float(p)
        tan_delta = float(tan_delta)
        if not np.isfinite(p) or p <= 0:
            return None
        if not np.isfinite(tan_delta) or tan_delta <= 0:
            return None
        return 1.0 / (p * tan_delta)
    except Exception:
        return None


# -------------------------
# data payload
# -------------------------
def build_chip_summary(*, sample_id: str) -> TDict[str, Any]:
    return {
        "meta": {
            "created_utc": now_utc_iso(),
            "updated_utc": None,
            "sample_id": sample_id,
            "units": {"f": "GHz", "kappa": "Hz", "C": "fF", "K": "MHz", "T": "us"},
        },
        "resonators": {
            "readout": {
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
        },
        "qubits": {
            "Q1": {
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
        },
        "q3d": {"internal": {}, "external": {}},
        "chip": {
            "T1_qubit_est_us": None,
            "T2_qubit_est_us": None,
            "T1_qubit_dielectric_us": None,
            "T1_qubit_purcell_us": None,
            "chi_over_kappa": None,
            "readout_f_GHz": None,
            "readout_kappa_over_2pi_Hz": None,
            "readout_kappa_over_2pi_MHz": None,
            "readout_Cin_fF": None,
            "qubit_f_GHz": None,
            "chi_MHz": None,
            "g_over_2pi_MHz": None,
        },
        "status": "init",
    }


def _estimate_T1T2(payload: TDict[str, Any]) -> None:
    ro = payload.get("resonators", {}).get("readout", {}) or {}
    q1 = payload.get("qubits", {}).get("Q1", {}) or {}
    fq_GHz = q1.get("f01_epr_GHz")

    kappa_over_2pi = ro.get("kappa_over_2pi_Hz")
    if kappa_over_2pi:
        try:
            T1_photon_s = 1.0 / (2.0 * math.pi * float(kappa_over_2pi))
            ro["T1_photon_us"] = _safe_us(T1_photon_s)
        except Exception:
            pass

    T1_diel_s = None
    Qq = q1.get("Q_dielectric_main")
    if Qq is not None and fq_GHz is not None:
        try:
            fq_hz = float(fq_GHz) * 1e9
            Qqf = float(Qq)
            if fq_hz > 0 and Qqf > 0:
                T1_diel_s = Qqf / (2.0 * math.pi * fq_hz)
                q1["T1_dielectric_us"] = _safe_us(T1_diel_s)
        except Exception:
            pass

    T1_purcell_s = None
    g_GHz = (q1.get("dispersive") or {}).get("g_GHz")
    Delta_GHz = (q1.get("dispersive") or {}).get("Delta_GHz")
    if g_GHz is not None and Delta_GHz is not None and kappa_over_2pi:
        try:
            g_hz = float(g_GHz) * 1e9
            d_hz = float(Delta_GHz) * 1e9
            k_over_2pi = float(kappa_over_2pi)
            if abs(d_hz) > 0 and g_hz > 0 and k_over_2pi > 0:
                Gamma_p = (g_hz / d_hz) ** 2 * (2.0 * math.pi * k_over_2pi)
                if Gamma_p > 0:
                    T1_purcell_s = 1.0 / Gamma_p
                    q1["T1_purcell_us"] = _safe_us(T1_purcell_s)
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
        q1["T1_est_us"] = _safe_us(T1_est_s)
        q1["T2_est_us"] = _safe_us(2.0 * T1_est_s)
        payload["chip"]["T1_qubit_est_us"] = q1["T1_est_us"]
        payload["chip"]["T2_qubit_est_us"] = q1["T2_est_us"]
        payload["chip"]["T1_qubit_dielectric_us"] = q1.get("T1_dielectric_us")
        payload["chip"]["T1_qubit_purcell_us"] = q1.get("T1_purcell_us")


def postprocess(payload: TDict[str, Any]) -> TDict[str, Any]:
    payload["meta"]["updated_utc"] = now_utc_iso()
    ro = payload["resonators"]["readout"]
    q1 = payload["qubits"]["Q1"]
    
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

    C_eff_fF = q1.get("C_eff_fF")
    Lj_H = q1.get("Lj_H")
    EC_hz = ec_over_h_hz_from_c(float(C_eff_fF) * 1e-15) if C_eff_fF else None
    EJ_hz = ej_over_h_hz_from_lj(float(Lj_H)) if Lj_H else None

    if EC_hz:
        q1["energies"]["Ec_GHz"] = EC_hz / 1e9
    if EJ_hz:
        q1["energies"]["Ej_GHz"] = EJ_hz / 1e9
    if EC_hz and EJ_hz:
        q1["energies"]["Ej_over_Ec"] = EJ_hz / EC_hz
        f01 = transmon_f01_hz(EJ_hz, EC_hz)
        q1["f01_transmon_GHz"] = (f01 / 1e9) if f01 else None

    chi_mhz = q1.get("chi_MHz")
    kappa_over_2pi_hz = ro.get("kappa_over_2pi_Hz")
    if chi_mhz is not None and kappa_over_2pi_hz:
        payload["chip"]["chi_over_kappa"] = (abs(float(chi_mhz)) * 1e6) / float(kappa_over_2pi_hz)

    payload["chip"]["readout_f_GHz"] = ro.get("f_GHz")
    payload["chip"]["readout_kappa_over_2pi_Hz"] = ro.get("kappa_over_2pi_Hz")
    payload["chip"]["readout_kappa_over_2pi_MHz"] = (
        float(ro["kappa_over_2pi_Hz"]) / 1e6 if ro.get("kappa_over_2pi_Hz") else None
    )
    payload["chip"]["readout_Cin_fF"] = (ro.get("external") or {}).get("Cin_fF")
    payload["chip"]["qubit_f_GHz"] = q1.get("f01_epr_GHz")
    payload["chip"]["chi_MHz"] = q1.get("chi_MHz")
    payload["chip"]["g_over_2pi_MHz"] = (
        float(q1["dispersive"]["g_GHz"]) * 1e3 if q1.get("dispersive", {}).get("g_GHz") else None
    )

    _estimate_T1T2(payload)
    return payload


# -------------------------
# Q3D helpers
# -------------------------
def diff_mode_ceff_from_2x2_maxwell_fF(C2_fF: np.ndarray) -> Optional[float]:
    try:
        C2 = np.array(C2_fF, dtype=float).reshape((2, 2))
        invC = np.linalg.pinv(C2)
        v = np.array([1.0, -1.0], dtype=float)
        denom = float(v @ invC @ v)
        if denom <= 0 or not np.isfinite(denom):
            return None
        return 1.0 / denom
    except Exception:
        return None


def pick_cin_prefer_tee_bodies(Cmat_fF: pd.DataFrame, tee_name: str = "RO_TEE") -> TDict[str, Any]:
    """
    鍏堝皾锟?cap_body_0 <-> cap_body_1
    濡傛灉 cap_body_1 涓嶅瓨鍦紙甯歌锛氬彟涓€锟?plate 鍚堝苟鍒颁簡 pad/trace node锛夛紝fallback锟?        锟?cap_body_0 涓庘€滆€﹀悎鏈€寮虹殑鍙︿竴涓妭鐐光€濆綋 Cin
    """
    a = f"cap_body_0_{tee_name}"
    b = f"cap_body_1_{tee_name}"
    if a in Cmat_fF.index and b in Cmat_fF.index:
        cin = abs(float(Cmat_fF.loc[a, b]))
        return {"Cin_fF": float(cin), "res_node": a, "feed_node": b, "pair": f"{a} <-> {b}"}

    if a in Cmat_fF.index:
        row = Cmat_fF.loc[a].copy()
        row = row.drop(labels=[a], errors="ignore")
        if len(row.index) > 0:
            j = str(row.abs().idxmax())
            cin = abs(float(Cmat_fF.loc[a, j]))
            return {"Cin_fF": float(cin), "res_node": j, "feed_node": a, "pair": f"{a} <-> {j} (fallback)"}

    return {"Cin_fF": 0.0, "res_node": None, "feed_node": None, "pair": None}


# -------------------------
# layout
# -------------------------
@dataclass
class LayoutParams:
    chip_size_x: str = "12mm"
    chip_size_y: str = "12mm"
    cpw_width: str = "10um"
    cpw_gap: str = "6um"

    feed_fillet: str = "25um"
    ro_fillet: str = "25um"

    pad_width: str = "520um"
    pocket_height: str = "780um"
    ro_pad_w: str = "160um"
    ro_pad_h: str = "140um"
    ro_pad_gap: str = "8um"

    lp_dx: float = 2.4

    prime_width: str = "14um"
    prime_gap: str = "6um"
    second_width: str = "10um"
    second_gap: str = "6um"
    cap_gap: str = "15um"
    cap_width: str = "10um"
    finger_length: str = "45um"
    finger_count: str = "3"
    cap_distance: str = "170um"

    feed_total_length: str = "5.5mm"
    feed_spacing: str = "280um"
    feed_lead_start: str = "250um"
    feed_lead_end: str = "250um"

    ro_total_length: str = "10.5mm"
    ro_spacing: str = "260um"
    ro_lead_start: str = "250um"
    ro_lead_end: str = "250um"


def _um_to_mm(u: str) -> float:
    return float(u.replace("um", "")) / 1000.0


def _mm_str_to_mm(s: str) -> float:
    return float(s.replace("mm", ""))


def _force_meander_total_len(total_len_mm: float, lead_s_um: str, lead_e_um: str, spacing_um: str) -> float:
    lead_mm = _um_to_mm(lead_s_um) + _um_to_mm(lead_e_um)
    spacing_mm = _um_to_mm(spacing_um)
    min_mm = lead_mm + 8.0 * spacing_mm
    return max(total_len_mm, min_mm + 1.0)


def _value_to_mm(value: object) -> float:
    try:
        if isinstance(value, str):
            if value.endswith("mm"):
                return float(value[:-2])
            if value.endswith("um"):
                return float(value[:-2]) / 1000.0
        return float(value)
    except Exception:
        return 0.0


def _is_finite_number(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def _pin_has_nan(pin: dict) -> bool:
    for key in ("normal", "tangent"):
        vec = pin.get(key)
        if vec is None or len(vec) < 2:
            return True
        if not _is_finite_number(vec[0]) or not _is_finite_number(vec[1]):
            return True
    points = pin.get("points")
    if not points:
        return True
    for point in points:
        if point is None or len(point) < 2:
            return True
        if not _is_finite_number(point[0]) or not _is_finite_number(point[1]):
            return True
    return False


def sanitize_center_readout_pin(design, component: str, pin_name: str = "readout") -> bool:
    try:
        pin = design.components[component].pins[pin_name]
    except Exception:
        return False

    if not _pin_has_nan(pin):
        return False

    middle = pin.get("middle")
    if middle is None or len(middle) < 2:
        return False

    mid_x = float(middle[0])
    mid_y = float(middle[1])
    if not _is_finite_number(mid_x) or not _is_finite_number(mid_y):
        return False

    comp = design.components[component]
    pos_y = _value_to_mm(comp.options.get("pos_y", 0.0))
    normal_sign = 1.0 if mid_y >= pos_y else -1.0

    width = pin.get("width", 0.01)
    width = float(width) if _is_finite_number(width) and float(width) > 0 else 0.01
    half_w = width / 2.0

    pin["normal"] = np.array([0.0, normal_sign], dtype=float)
    pin["tangent"] = np.array([1.0, 0.0], dtype=float)
    pin["points"] = [
        np.array([mid_x - half_w, mid_y], dtype=float),
        np.array([mid_x + half_w, mid_y], dtype=float),
    ]
    return True


def populate_design(design, q_x_mm: float, q_y_mm: float, *, p: LayoutParams) -> None:
    design.overwrite_enabled = True
    design.variables.update(Dict(cpw_width=p.cpw_width, cpw_gap=p.cpw_gap))
    design.chips.main.size["size_x"] = p.chip_size_x
    design.chips.main.size["size_y"] = p.chip_size_y

    # Assign tan_delta to the Qiskit Metal chip material when available.
    try:
        if hasattr(design.chips.main, "material") and isinstance(design.chips.main.material, dict):
            design.chips.main.material["tan_delta"] = float(TAN_DELTA_MAIN)
        elif hasattr(design.chips.main, "material"):
            setattr(design.chips.main.material, "tan_delta", float(TAN_DELTA_MAIN))
    except Exception:
        pass

    q1 = TransmonPocket(
        design,
        "Q1",
        options=dict(
            pos_x=f"{q_x_mm}mm",
            pos_y=f"{q_y_mm}mm",
            pad_width=p.pad_width,
            pocket_height=p.pocket_height,
            connection_pads=dict(
                readout=dict(
                    loc_W=+1,
                    loc_H=+1,
                    pad_width=p.ro_pad_w,
                    pad_height=p.ro_pad_h,
                    pad_gap=p.ro_pad_gap,
                )
            ),
        ),
    )
    design.components["Q1"].options["hfss_inductance"] = "Lj"
    design.components["Q1"].options["hfss_capacitance"] = "Cj"
    design.rebuild()

    try:
        qx_pin = float(design.components["Q1"].pins["readout"]["middle"][0])
        qy_pin = float(design.components["Q1"].pins["readout"]["middle"][1])
    except Exception:
        qx_pin, qy_pin = q_x_mm, q_y_mm

    coupler_x = qx_pin + 0.38
    coupler_y = qy_pin + 0.38

    lp = _LAUNCHPAD_CLS(
        design,
        "LP_RO",
        options=dict(
            pos_x=f"{coupler_x - p.lp_dx}mm",
            pos_y=f"{coupler_y}mm",
            orientation="0",
            trace_width=p.cpw_width,
            trace_gap=p.cpw_gap,
        ),
    )

    tee = CapNInterdigitalTee(
        design,
        "RO_TEE",
        options=dict(
            pos_x=f"{coupler_x}mm",
            pos_y=f"{coupler_y}mm",
            orientation="0",
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

    RouteMeander(
        design,
        "RO_FEED",
        options=Dict(
            total_length=p.feed_total_length,
            fillet=p.feed_fillet,
            trace_width=p.cpw_width,
            trace_gap=p.cpw_gap,
            meander=Dict(spacing=p.feed_spacing),
            lead=Dict(start_straight=p.feed_lead_start, end_straight=p.feed_lead_end),
            pin_inputs=Dict(
                start_pin=Dict(component=lp.name, pin="tie"),
                end_pin=Dict(component=tee.name, pin="prime_start"),
            ),
        ),
        type="CPW",
    )

    ro_mm = _mm_str_to_mm(p.ro_total_length)
    ro_mm = _force_meander_total_len(ro_mm, p.ro_lead_start, p.ro_lead_end, p.ro_spacing)

    RouteMeander(
        design,
        "RO_RES",
        options=Dict(
            total_length=f"{ro_mm:.3f}mm",
            fillet=p.ro_fillet,
            trace_width=p.cpw_width,
            trace_gap=p.cpw_gap,
            meander=Dict(spacing=p.ro_spacing),
            lead=Dict(start_straight=p.ro_lead_start, end_straight=p.ro_lead_end),
            pin_inputs=Dict(
                start_pin=Dict(component=tee.name, pin="second_end"),
                end_pin=Dict(component=q1.name, pin="readout"),
            ),
        ),
        type="CPW",
    )

    design.rebuild()


# -------------------------
# GDS export
# -------------------------
def _configure_gds_for_export(gds, *, safe_mode: bool) -> None:
    gds.options["corners"] = "natural"
    if GDS_DISABLE_CHEESE:
        try:
            gds.options["cheese"]["view_in_file"] = Dict(main={})
        except Exception:
            pass
        try:
            gds.options["no_cheese"]["view_in_file"] = Dict(main={})
        except Exception:
            pass
    gds.options["ground_plane"] = True
    if safe_mode:
        gds.options["negative_mask"] = Dict(main=[])


def _gds_file_ok(path: Path) -> bool:
    try:
        return path.exists() and path.is_file() and path.stat().st_size > 200
    except Exception:
        return False


def export_gds_robust(design, abs_gds_path: str) -> TDict[str, Any]:
    gds = design.renderers.gds
    abs_path = Path(abs_gds_path)
    abs_path.parent.mkdir(parents=True, exist_ok=True)

    errors: List[str] = []
    attempts = [
        ("safe", True),
        ("cpw", False),
        ("minimal", None),
    ]

    if not GDS_FORCE_SAFE_MODE:
        attempts = [("cpw", False), ("safe", True), ("minimal", None)]

    for mode, safe_mode in attempts:
        try:
            if safe_mode is None:
                gds.options["corners"] = "natural"
                try:
                    gds.options.pop("ground_plane", None)
                except Exception:
                    pass
                try:
                    gds.options.pop("negative_mask", None)
                except Exception:
                    pass
            else:
                _configure_gds_for_export(gds, safe_mode=bool(safe_mode))

            gds.export_to_gds(str(abs_path))
            if _gds_file_ok(abs_path):
                return {"mode": mode, "ok": True, "error": None}
            errors.append(f"{mode}: export completed but file missing/empty")
        except Exception as e:
            errors.append(f"{mode}: {type(e).__name__}: {e!r}")

    return {"mode": "failed", "ok": False, "error": "; ".join(errors)}


# -------------------------
# Ansys hard reset (best-effort)
# -------------------------
def _ansys_best_effort_reset():
    try:
        import pyEPR as epr
        try:
            if hasattr(epr.ansys, "release_desktop"):
                epr.ansys.release_desktop()
        except Exception:
            pass
        try:
            if hasattr(epr.ansys, "close_desktop"):
                epr.ansys.close_desktop()
        except Exception:
            pass
    except Exception:
        pass

    try:
        gc.collect()
    except Exception:
        pass

    if os.name == "nt":
        for exe in ("ansysedt.exe", "ansysedt64.exe", "ansysedtlite.exe"):
            try:
                subprocess.run(
                    ["taskkill", "/F", "/IM", exe, "/T"],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except Exception:
                pass
        try:
            time.sleep(1.0)
        except Exception:
            pass


def _ansys_wait_ready(renderer, *, timeout_s: float = 30.0, interval_s: float = 0.5) -> bool:
    if renderer is None:
        return False
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            if hasattr(renderer, "rdesktop") and renderer.rdesktop:
                renderer.rdesktop.project_count()
                return True
        except Exception:
            pass
        time.sleep(interval_s)
    return False


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


def _safe_ascii(obj) -> str:
    try:
        s = str(obj)
    except Exception:
        s = repr(obj)
    try:
        return s.encode("ascii", "backslashreplace").decode("ascii")
    except Exception:
        return repr(s)


def _format_com_error(err: Exception) -> str:
    parts = []
    hresult = getattr(err, "hresult", None)
    if isinstance(hresult, int):
        parts.append(f"hresult={hresult}({hex(hresult & 0xFFFFFFFF)})")
    elif hresult is not None:
        parts.append(f"hresult={_safe_ascii(hresult)}")
    args = getattr(err, "args", None)
    if args:
        parts.append(f"args={_safe_ascii(args)}")
    excepinfo = getattr(err, "excepinfo", None)
    if excepinfo:
        parts.append(f"excepinfo={_safe_ascii(excepinfo)}")
    return " ".join(parts)




def _ansys_short_workdir(root: Path) -> Path:
    """
    鍏抽敭锛氱敤寰堢煭鐨勭洰褰曪紝閬垮厤 Ansys results 鐩綍灞傜骇杩囨繁瀵艰嚧 0x80070003
    """
    try:
        return Path(tempfile.gettempdir()) / "ansys_work"
    except Exception:
        pass
    try:
        if os.name == "nt" and root.drive:
            return Path(root.drive + "\\ansys_work")
    except Exception:
        pass
    return root / "ansys_work"


def _ansys_save_active_project_as(renderer, target_aedt: Path) -> None:
    target_aedt.parent.mkdir(parents=True, exist_ok=True)
    try:
        pinfo = getattr(renderer, "pinfo", None)
        if pinfo is None or getattr(pinfo, "design", None) is None:
            return
        oDesktop = pinfo.design.parent.parent._desktop
        oProject = oDesktop.GetActiveProject()
        oProject.SaveAs(str(target_aedt), True)
    except Exception:
        pass


def _ansys_force_new_project_best_effort(renderer) -> None:
    try:
        if hasattr(renderer, "new_ansys_project"):
            renderer.new_ansys_project()
        if hasattr(renderer, "connect_ansys"):
            renderer.connect_ansys()
    except Exception:
        pass


def _ansys_prepare_project(renderer, target_aedt: Path) -> None:
    target_aedt.parent.mkdir(parents=True, exist_ok=True)

    prepared = False
    try:
        if hasattr(renderer, "rdesktop") and renderer.rdesktop:
            project = renderer.rdesktop.get_active_project()
            if not project:
                renderer.rdesktop.new_project()
                project = renderer.rdesktop.get_active_project()
            if project:
                project.SaveAs(str(target_aedt), True)
                prepared = True
    except Exception:
        prepared = False

    if not prepared:
        _ansys_force_new_project_best_effort(renderer)
        _ansys_save_active_project_as(renderer, target_aedt)


# -------------------------
# analysis
# -------------------------
def run_analysis_pipeline(
    design,
    sample_id: str,
    *,
    root_path: str,
    do_hfss: bool,
    do_q3d: bool,
    lj_nh: float,
    cj_fF_user: float,
    inputs_sweep: Optional[TDict[str, Any]] = None,
    inputs_qubits: Optional[TDict[str, Any]] = None,
) -> TDict[str, Any]:
    root = Path(root_path)
    (root / "gds").mkdir(parents=True, exist_ok=True)
    (root / "json").mkdir(parents=True, exist_ok=True)

    payload = build_chip_summary(sample_id=sample_id)
    payload["status"] = "running"

    payload["inputs"] = {
        "sweep": dict(inputs_sweep or {}),
        "qubits": dict(inputs_qubits or {}),
        "junction": {"Lj_nH": float(lj_nh), "Cj_fF": float(cj_fF_user)},
    }

    filename_base = sample_id
    payload["meta"]["filename_base"] = filename_base

    sim_tag = short_hash_tag(sample_id, n=8)
    payload["meta"]["ansys_sim_tag"] = sim_tag

    # 1) GDS
    try:
        gds_path = root / "gds" / f"{filename_base}.gds"
        abs_gds_path = str(gds_path.resolve())
        print(f"[GDS] Exporting to: {abs_gds_path}")
        info = export_gds_robust(design, abs_gds_path)
        payload["meta"]["gds_path"] = abs_gds_path
        payload["meta"]["gds_export_mode"] = info["mode"]
        payload["meta"]["gds_error"] = info.get("error") or ""
        print(f"[OK] [GDS] ok={info['ok']} mode={info['mode']}", flush=True)
    except Exception as e:
        payload["meta"]["gds_error"] = str(e)

    # -----------------
    # 2) HFSS/EPR
    # -----------------
    if do_hfss:
        for attempt in [1, 2]:
            eig = None
            step = "init"
            try:
                step = "epr_init"
                eig = EPRanalysis(design, "hfss")
                hfss = eig.sim.renderer

                workdir = _ansys_short_workdir(root)
                target = workdir / f"EIG_{sim_tag}_a{attempt}.aedt"

                step = "hfss.reset"
                _ansys_best_effort_reset()
                step = "hfss.start"
                hfss.start()
                time.sleep(0.8)
                step = "hfss.prepare_project"
                _ansys_prepare_project(hfss, target)

                eig.sim.setup.vars = Dict(Lj=f"{float(lj_nh)} nH", Cj=f"{float(cj_fF_user)} fF")
                eig.sim.setup.n_modes = 4
                eig.sim.setup.max_passes = 8
                eig.sim.setup.min_freq_ghz = 1.0

                all_components = list(design.components.keys())

                step = "hfss.run"
                eig.sim.run(
                    name=f"Eig_{sim_tag}_a{attempt}",
                    components=all_components,
                    open_terminations=[],
                    box_plus_buffer=True,
                )

                pinfo = hfss.pinfo
                jkeys = list(getattr(pinfo, "junctions", {}).keys())
                if jkeys:
                    jname = jkeys[0]
                    jinfo = dict(pinfo.junctions[jname])
                    jinfo["Lj_variable"] = "Lj"
                    jinfo["Cj_variable"] = "Cj"
                    pinfo.junctions[jname] = jinfo
                    eig.setup.junctions[jname] = jinfo
                else:
                    pinfo.junctions["jj"] = {
                        "Lj_variable": "Lj",
                        "rect": "JJ_rect_Lj_Q1_rect_jj",
                        "line": "JJ_Lj_Q1_rect_jj_",
                        "Cj_variable": "Cj",
                    }
                    pinfo.validate_junction_info()
                    eig.setup.junctions["jj"] = pinfo.junctions["jj"]

                eig.setup.dissipatives = {"dielectrics_bulk": ["main"]}
                
                # 鏄惧紡璁剧疆tan_delta鍒皊etup锟?                # pyEPR鍙兘闇€瑕佽繖鏍风殑閰嶇疆鎵嶈兘姝ｇ‘浣跨敤tan_delta
                try:
                    # 鏂规硶1: 鐩存帴鍦╠issipatives涓锟?                    if hasattr(eig.setup, 'dissipative'):
                        eig.setup.dissipative = {'dielectrics_bulk': {'main': TAN_DELTA_MAIN}}
                        print(f"[OK] [DEBUG] Set eig.setup.dissipative with tan_delta = {TAN_DELTA_MAIN}", flush=True)
                except Exception as e:
                    print(f"[WARN] [DEBUG] Method 1 failed: {_safe_ascii(e)}", flush=True)
                
                step = "hfss.run_epr"
                try:
                    import pyEPR.core_distributed_analysis as _cda

                    if not getattr(_cda, "_metal_safe_hfss_report_f_convergence", False):
                        _orig = _cda.DistributedAnalysis.hfss_report_f_convergence

                        def _safe_hfss_report_f_convergence(self, variation: str = "0", save_csv: bool = True):
                            try:
                                return _orig(self, variation=variation, save_csv=save_csv)
                            except Exception:
                                return None

                        _cda.DistributedAnalysis.hfss_report_f_convergence = _safe_hfss_report_f_convergence
                        _cda._metal_safe_hfss_report_f_convergence = True
                except Exception:
                    pass

                epr_error = None
                try:
                    eig.clear_data()
                    eig.get_stored_energy(no_junctions=False)
                    eig.run_analysis()
                    eig.spectrum_analysis(eig.setup.cos_trunc, eig.setup.fock_trunc)
                    try:
                        eig.report_hamiltonian(eig.setup.sweep_variable)
                    except Exception as e:
                        payload["meta"]["epr_report_error"] = _safe_ascii(e)
                except Exception as e:
                    epr_error = e
                    payload["meta"]["epr_error"] = _safe_ascii(e)

                freqs = eig.get_frequencies().iloc[:, 0].values
                payload["resonators"]["readout"]["modes_f_GHz"] = to_jsonable(freqs)

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

                idx_q = int(np.argmax(np.abs(np.diag(chi_matrix))))
                row = np.abs(chi_matrix[idx_q, :])
                row[idx_q] = -1.0
                idx_r = int(np.argmax(row))

                payload["coupling"] = build_coupling_pairs(
                    chi_matrix_MHz=chi_matrix,
                    freqs_GHz=freqs,
                    idx_qubits=[idx_q],
                    idx_readouts=[idx_r],
                    qubit_names=["Q1"],
                    readout_names=["readout"],
                )

                f_qubit = float(freqs[idx_q])
                f_res = float(freqs[idx_r])

                alpha_MHz = -abs(float(chi_matrix[idx_q, idx_q]))
                chi_MHz = float(chi_matrix[idx_q, idx_r])
                K_MHz = float(chi_matrix[idx_r, idx_r])
                Delta_GHz = f_qubit - f_res

                g_GHz = None
                try:
                    Delta_MHz = Delta_GHz * 1e3
                    if abs(alpha_MHz) > 1e-9:
                        g_MHz = math.sqrt(abs(chi_MHz * Delta_MHz * (Delta_MHz + alpha_MHz) / alpha_MHz))
                        g_GHz = g_MHz / 1e3
                except Exception:
                    g_GHz = None

                ro = payload["resonators"]["readout"]
                q1 = payload["qubits"]["Q1"]

                ro["f_GHz"] = f_res
                ro["K_MHz"] = K_MHz
                ro["picked_qubit_mode_index"] = idx_q
                ro["picked_res_mode_index"] = idx_r

                q1["f01_epr_GHz"] = f_qubit
                q1["alpha_epr_MHz"] = alpha_MHz
                q1["chi_MHz"] = chi_MHz
                q1["dispersive"]["chi_GHz"] = chi_MHz / 1e3
                q1["dispersive"]["Delta_GHz"] = Delta_GHz
                q1["dispersive"]["g_GHz"] = g_GHz

                # -------------------------
                # 锟?鍏抽敭淇敼锛氫笉鐢ㄢ€滃父锟?fallback鈥濅及 Qi
                # 浼樺厛锛氫粠 participation p 璁＄畻 Qi = 1/(p*tan未)
                # 娆￠€夛細EPR 杈撳嚭锟?Qdielectric_main
                # 鏈€鍚庯細鍙拷?fallback锛堝彲鍏虫帀锟?                # -------------------------
                Qi_val = None

                p_r = extract_participation(res, idx_r, kind="dielectrics_bulk", name="main")
                Qi_from_p = qi_from_p_tandelta(p_r, TAN_DELTA_MAIN)
                if Qi_from_p is not None:
                    Qi_val = float(Qi_from_p)
                    ro["warnings"].append(
                        f"Qi from p*tan未: p={float(p_r):.3g}, tan未={float(TAN_DELTA_MAIN):.3g}, Qi={Qi_val:.3g}"
                    )

                # 濡傛灉娌℃湁participation鏁版嵁锛屽皾璇曚粠EPR鐨凲dielectric鍙嶆帹
                if Qi_val is None:
                    Qi_raw = extract_qdielectric_main(res, idx_r)
                    if Qi_raw is not None and np.isfinite(Qi_raw) and Qi_raw > 0:
                        # 鍏抽敭淇敼锛氫粠EPR鐨凲dielectric鍙嶆帹participation锛岀劧鍚庣敤TAN_DELTA_MAIN閲嶆柊璁＄畻
                        # EPR浣跨敤鐨則an_delta鏈煡鏃讹紝鐢‥PR_TAN_DELTA_ASSUMED鍙嶆帹p
                        p_inferred = 1.0 / (float(Qi_raw) * float(EPR_TAN_DELTA_ASSUMED))
                        
                        if 0.0001 <= p_inferred <= 0.99:
                            # 鐢ㄥ弽鎺ㄧ殑p鍜屾垜浠缃殑TAN_DELTA_MAIN閲嶆柊璁＄畻Qi
                            Qi_val = 1.0 / (p_inferred * float(TAN_DELTA_MAIN))
                            ro["warnings"].append(
                                "Qi recalculated from inferred p="
                                f"{p_inferred:.3g}: tan_delta={float(TAN_DELTA_MAIN):.3g} "
                                f"(assumed EPR tan_delta={float(EPR_TAN_DELTA_ASSUMED):.3g}), Qi={Qi_val:.3g}"
                            )
                        else:
                            # 濡傛灉鏃犳硶鍙嶆帹锛岀洿鎺ヤ娇鐢‥PR鐨勫€间絾鍙戝嚭璀﹀憡
                            Qi_val = float(Qi_raw)
                            ro["warnings"].append(f"Qi from EPR Qdielectric_main (TAN_DELTA_MAIN may not be used): {Qi_val:.3g}")

                if Qi_val is None and USE_QI_FALLBACK:
                    Qi_val = float(QI_FALLBACK)
                    ro["warnings"].append(f"Qi fallback assumed: {QI_FALLBACK:g}")

                if Qi_val is not None and QI_CLAMP_TO_FALLBACK and Qi_val > float(QI_FALLBACK):
                    ro["warnings"].append(f"Qi clamped to {QI_FALLBACK:g} (raw {Qi_val:g})")
                    Qi_val = float(QI_FALLBACK)

                ro["Qi"] = float(Qi_val) if Qi_val is not None else None
                ro["kappa_i_over_2pi_Hz"] = (float(f_res) * 1e9) / float(Qi_val) if Qi_val else None

                # Qubit dielectric Q 鍚岀悊锛堝锟?p 鑳芥姄鍒板氨锟?p*tan未锛涘惁鍒欑敤 EPR 锟?Q锟?                q1Q_val = None
                p_q = extract_participation(res, idx_q, kind="dielectrics_bulk", name="main")
                q1Q_from_p = qi_from_p_tandelta(p_q, TAN_DELTA_MAIN)
                if q1Q_from_p is not None:
                    q1Q_val = float(q1Q_from_p)
                else:
                    q1Q_raw = extract_qdielectric_main(res, idx_q)
                    if q1Q_raw is not None and np.isfinite(q1Q_raw) and q1Q_raw > 0:
                        q1Q_val = float(q1Q_raw)

                if q1Q_val is not None:
                    q1["Q_dielectric_main"] = float(q1Q_val)

                # Keep EPR solution column names for diagnostics.
                try:
                    sol = res.get("sol", None)
                    if isinstance(sol, pd.DataFrame):
                        payload["meta"]["epr_sol_columns"] = [str(c) for c in sol.columns]
                except Exception:
                    pass

                break

            except Exception as e:
                payload["meta"]["hfss_error"] = f"{step}: {_safe_ascii(e)}"
                details = _format_com_error(e)
                if details:
                    print(f"[ERROR] [HFSS] attempt={attempt} step={step} error={_safe_ascii(e)} {details}")
                else:
                    print(f"[ERROR] [HFSS] attempt={attempt} step={step} error={_safe_ascii(e)}")
                if attempt == 1:
                    try:
                        if eig is not None:
                            eig.sim.close()
                    except Exception:
                        pass
                    _ansys_best_effort_reset()
                    time.sleep(2.0)
                    gc.collect()
                    continue
            finally:
                try:
                    if eig is not None:
                        eig.sim.close()
                except Exception:
                    pass

    # -----------------
    # 3) Q3D (FIX)
    # -----------------
    C_eff_total_fF = None
    cj_fF = float(cj_fF_user)

    if do_q3d:
        for attempt in [1, 2]:
            lom = None
            step = "init"
            try:
                step = "lom_init"
                lom = LOManalysis(design, "q3d")
                q3d = lom.sim.renderer

                workdir = _ansys_short_workdir(root)
                target = workdir / f"Q3D_{sim_tag}_a{attempt}.aedt"

                step = "q3d.reset"
                _ansys_best_effort_reset()
                step = "q3d.start"
                q3d.start()
                time.sleep(0.8)
                step = "q3d.prepare_project"
                _ansys_prepare_project(q3d, target)

                try:
                    lom.sim.renderer_initialized = True
                except Exception:
                    pass

                lom.sim.setup.freq_ghz = 5.0
                lom.sim.setup.max_passes = 8
                step = "q3d.run"
                lom.sim.run(name=f"Q3D_{sim_tag}_a{attempt}", components=list(design.components.keys()))

                step = "q3d.capacitance_matrix"
                Cmat = lom.sim.capacitance_matrix

                pad_top = "pad_top_Q1"
                pad_bot = "pad_bot_Q1"
                if pad_top in Cmat.index and pad_bot in Cmat.index:
                    C2 = Cmat.loc[[pad_top, pad_bot], [pad_top, pad_bot]].values.astype(float)
                    C_mode_fF = diff_mode_ceff_from_2x2_maxwell_fF(C2)
                    if C_mode_fF is not None and np.isfinite(C_mode_fF):
                        C_eff_total_fF = C_mode_fF + cj_fF

                payload["q3d"]["internal"]["capacitances_fF"] = {
                    "units": "fF",
                    "nodes": list(Cmat.index),
                    "capacitance_matrix_fF": to_jsonable(Cmat),
                }

                # Diagnostic: report strongest couplings and tee nodes
                try:
                    nodes = list(Cmat.index)
                    tee_nodes = [n for n in nodes if "RO_TEE" in n]
                    if tee_nodes:
                        print(f"[Q3D] tee nodes: {tee_nodes}", flush=True)
                    else:
                        print("[Q3D] tee nodes: (none)", flush=True)

                    Cabs = Cmat.abs().copy()
                    for n in nodes:
                        if n in Cabs.index and n in Cabs.columns:
                            Cabs.loc[n, n] = 0.0

                    max_pair = None
                    max_val = None
                    for i in nodes:
                        for j in nodes:
                            if i == j:
                                continue
                            v = float(Cabs.loc[i, j])
                            if (max_val is None) or (v > max_val):
                                max_val = v
                                max_pair = (i, j)
                    if max_pair and max_val is not None:
                        print(f"[Q3D] max |C|: {max_val:.6g} fF between {max_pair[0]} <-> {max_pair[1]}", flush=True)
                except Exception:
                    pass

                cin_pick = pick_cin_prefer_tee_bodies(Cmat, tee_name="RO_TEE")
                payload["q3d"]["external"] = {"units": "fF", **cin_pick}

                ro = payload["resonators"]["readout"]
                ro["external"]["Cin_fF"] = cin_pick["Cin_fF"]
                ro["external"]["pair"] = cin_pick["pair"]
                ro["external"]["res_node"] = cin_pick["res_node"]
                ro["external"]["feed_node"] = cin_pick["feed_node"]

                if cin_pick["Cin_fF"] > 0 and ro.get("f_GHz"):
                    from qiskit_metal.analyses.em.kappa_calculation import kappa_in
                    fr_hz = float(ro["f_GHz"]) * 1e9
                    Cin_F = float(cin_pick["Cin_fF"]) * 1e-15
                    kappa_hz = float(kappa_in(fr_hz, Cin_F, fr_hz))
                    ro["external"]["kappa_Hz"] = kappa_hz
                    ro["external"]["kappa_over_2pi_Hz"] = kappa_hz / (2.0 * math.pi) if kappa_hz > 0 else None
                    ro["external"]["Qe"] = fr_hz / kappa_hz if kappa_hz > 0 else None

                break

            except Exception as e:
                payload["meta"]["q3d_error"] = f"{step}: {_safe_ascii(e)}"
                details = _format_com_error(e)
                if details:
                    print(f"[ERROR] [Q3D] attempt={attempt} step={step} error={_safe_ascii(e)} {details}")
                else:
                    print(f"[ERROR] [Q3D] attempt={attempt} step={step} error={_safe_ascii(e)}")
                if attempt == 1:
                    try:
                        if lom is not None:
                            lom.sim.close()
                    except Exception:
                        pass
                    _ansys_best_effort_reset()
                    time.sleep(2.0)
                    gc.collect()
                    continue
            finally:
                try:
                    if lom is not None:
                        lom.sim.close()
                except Exception:
                    pass

    payload["qubits"]["Q1"]["Lj_H"] = float(lj_nh) * 1e-9
    payload["qubits"]["Q1"]["Cj_fF"] = cj_fF
    payload["qubits"]["Q1"]["C_eff_fF"] = C_eff_total_fF

    payload = postprocess(payload)
    payload["status"] = "completed"

    json_path = Path(root_path) / "json" / f"{filename_base}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(export_payload_v3(payload)), f, indent=2, ensure_ascii=False)
    print(f"馃搫 [JSON] Saved to {json_path}")

    return payload


# -------------------------
# main
# -------------------------
if __name__ == "__main__":
    dataset_root = Path("./data/sqchip_em_1q")
    dataset_root.mkdir(parents=True, exist_ok=True)

    LJ_NH = 12.0
    # Match Qiskit Metal / pyEPR common default: ignore JJ capacitance in EM (use Cj=0).
    CJ_FF = 0.0

    TEE_FINGER_LENGTH_UM = 160
    TEE_FINGER_COUNT = 12
    TEE_CAP_GAP_UM = 1.2
    TEE_CAP_WIDTH_UM = 18.0
    TEE_CAP_DISTANCE_UM = 40.0

    RO_L_MM = 8.5
    Q_RO_PAD_GAP_UM = 1.4

    TARGET_ABS_CHI_MHZ = 2.0
    MAX_SAMPLES = None  # None means no limit
    RESUME_FROM_SUMMARY = True

    csv_headers = [
        "sample_id","dx_mm","dy_mm","Lj_nH","Cj_fF",
        "tee_finger_length_um","tee_finger_count","tee_cap_gap_um","ro_L_mm",
        "fq_GHz","fr_GHz","Delta_GHz","g_over_2pi_MHz","chi_MHz",
        "kappa_over_2pi_MHz","chi_over_kappa","Cin_fF","Ceff_fF",
        "kappa_i_over_kappa_e",
        "hit_chi_ge_2MHz","T1_qubit_est_us","T2_qubit_est_us",
        "T1_qubit_dielectric_us","T1_qubit_purcell_us","status","error","warnings",
    ]

    summary_csv_path = dataset_root / "summary.csv"
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

    DX_LIST = np.round(np.linspace(-10, 10, 23), 3)
    DY_LIST = np.round(np.linspace(-10, 10, 23), 3)
    grid = list(product(DX_LIST, DY_LIST))

    lp = LayoutParams(
        chip_size_x="12mm",
        chip_size_y="12mm",
        cpw_width="10um",
        cpw_gap="6um",
        feed_fillet="25um",
        ro_fillet="25um",

        # 锟?澧炲ぇ readout pad锛堟洿寮虹數瀹硅€﹀悎锛実 寰€涓婅蛋锟?        ro_pad_w="270um",
        ro_pad_h="270um",
        ro_pad_gap=f"{Q_RO_PAD_GAP_UM}um",

        ro_total_length=f"{RO_L_MM}mm",

        finger_length=f"{TEE_FINGER_LENGTH_UM}um",
        finger_count=str(int(TEE_FINGER_COUNT)),
        cap_gap=f"{TEE_CAP_GAP_UM}um",
        cap_width=f"{TEE_CAP_WIDTH_UM}um",
        cap_distance=f"{TEE_CAP_DISTANCE_UM}um",
    )

    processed_new = 0
    for (dx, dy) in grid:
        if MAX_SAMPLES is not None and processed_new >= MAX_SAMPLES:
            break

        sample_id = (
            "chip_simple_"
            f"dx{fmt_mm_id(float(dx))}_dy{fmt_mm_id(float(dy))}"
            f"__Lj{str(float(LJ_NH)).replace('.', 'p')}nH"
            f"_Cj{str(float(CJ_FF)).replace('.', 'p')}fF"
            f"__roL{str(float(RO_L_MM)).replace('.', 'p')}mm"
            f"__tee_fl{int(TEE_FINGER_LENGTH_UM)}um"
            f"_fc{int(TEE_FINGER_COUNT)}"
            f"_cg{str(float(TEE_CAP_GAP_UM)).replace('.', 'p')}um"
            f"__rpg{str(float(Q_RO_PAD_GAP_UM)).replace('.', 'p')}um"
        )

        if RESUME_FROM_SUMMARY and sample_id in existing_ids:
            continue

        row_base = dict(
            sample_id=sample_id,
            dx_mm=float(dx),
            dy_mm=float(dy),
            Lj_nH=float(LJ_NH),
            Cj_fF=float(CJ_FF),
            tee_finger_length_um=int(TEE_FINGER_LENGTH_UM),
            tee_finger_count=int(TEE_FINGER_COUNT),
            tee_cap_gap_um=float(TEE_CAP_GAP_UM),
            ro_L_mm=float(RO_L_MM),
        )

        try:
            design = designs.DesignPlanar({}, True)
            populate_design(design, q_x_mm=float(dx), q_y_mm=float(dy), p=lp)

            result = run_analysis_pipeline(
                design,
                sample_id,
                root_path=str(dataset_root),
                do_hfss=True,
                do_q3d=True,
                lj_nh=float(LJ_NH),
                cj_fF_user=float(CJ_FF),
                inputs_sweep={"dx_mm": float(dx), "dy_mm": float(dy)},
                inputs_qubits={"Q1": {"x_mm": float(dx), "y_mm": float(dy)}},
            )

            ro = result.get("resonators", {}).get("readout", {})
            q1 = result.get("qubits", {}).get("Q1", {})
            chip = result.get("chip", {})

            chi = q1.get("chi_MHz")
            hit = (chi is not None) and (abs(float(chi)) >= TARGET_ABS_CHI_MHZ)

            warnings = ro.get("warnings") or []
            warnings_str = "; ".join([str(x) for x in warnings]) if warnings else ""

            out = dict(row_base)
            out.update(
                dict(
                    fq_GHz=q1.get("f01_epr_GHz"),
                    fr_GHz=ro.get("f_GHz"),
                    Delta_GHz=(q1.get("dispersive") or {}).get("Delta_GHz"),
                    g_over_2pi_MHz=chip.get("g_over_2pi_MHz"),
                    chi_MHz=chi,
                    kappa_over_2pi_MHz=chip.get("readout_kappa_over_2pi_MHz"),
                    chi_over_kappa=chip.get("chi_over_kappa"),
                    Cin_fF=(ro.get("external") or {}).get("Cin_fF"),
                    Ceff_fF=q1.get("C_eff_fF"),
                    kappa_i_over_kappa_e=ro.get("kappa_i_over_kappa_e"),
                    hit_chi_ge_2MHz=hit,
                    T1_qubit_est_us=chip.get("T1_qubit_est_us"),
                    T2_qubit_est_us=chip.get("T2_qubit_est_us"),
                    T1_qubit_dielectric_us=chip.get("T1_qubit_dielectric_us"),
                    T1_qubit_purcell_us=chip.get("T1_qubit_purcell_us"),
                    status=result.get("status", ""),
                    error=(
                        result.get("meta", {}).get("hfss_error", "")
                        or result.get("meta", {}).get("q3d_error", "")
                        or result.get("meta", {}).get("gds_error", "")
                        or ""
                    ),
                    warnings=warnings_str,
                )
            )
            write_row(out)
            if RESUME_FROM_SUMMARY:
                existing_ids.add(sample_id)

        except Exception as e:
            out = dict(row_base)
            out.update(dict(status="ERROR", error=str(e), warnings=""))
            write_row(out)
            if RESUME_FROM_SUMMARY:
                existing_ids.add(sample_id)
        finally:
            _mpl_cleanup()

        processed_new += 1

    print(f"[Done] processed_new={processed_new}")
