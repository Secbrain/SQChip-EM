from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_PATHS = [
    "README.md",
    "requirements.txt",
    "figure/SQChip_EM_Pipeline.png",
    "data/sqchip_em_1q/summary.csv",
    "data/sqchip_em_1q/json",
    "data/sqchip_em_1q/gds",
    "data/sqchip_em_2q/summary.csv",
    "data/sqchip_em_2q/json",
    "data/sqchip_em_2q/gds",
    "data/sqchip_em_3q",
    "data/sqchip_em_3q/json",
    "data/sqchip_em_3q/gds",
    "data/sqchip_em_4q",
    "data/sqchip_em_4q/json",
    "data/sqchip_em_4q/gds",
    "data/sqchip_em_6q",
    "data/sqchip_em_6q/json",
    "data/sqchip_em_6q/gds",
    "data/sqchip_em_8q",
    "data/sqchip_em_8q/json",
    "data/sqchip_em_8q/gds",
    "examples/poor_2q/json",
    "examples/poor_2q/gds",
    "examples/poor_2q/metadata",
    "task1_baseline",
    "task2_baseline",
    "task3_baseline",
]

SUMMARY_REQUIRED_COLUMNS = {
    "data/sqchip_em_1q/summary.csv": {
        "sample_id",
        "dx_mm",
        "dy_mm",
        "Lj_nH",
        "Cj_fF",
        "fq_GHz",
        "fr_GHz",
        "chi_MHz",
        "kappa_over_2pi_MHz",
        "status",
    },
    "data/sqchip_em_2q/summary.csv": {
        "sample_id",
        "q1_x_mm",
        "q1_y_mm",
        "q2_x_mm",
        "q2_y_mm",
        "Lj1_nH",
        "Lj2_nH",
        "fq1_GHz",
        "fq2_GHz",
        "status",
    },
}


def check_required_paths() -> list[str]:
    errors: list[str] = []
    for rel in REQUIRED_PATHS:
        path = ROOT / rel
        if not path.exists():
            errors.append(f"Missing required path: {rel}")
    return errors


def read_csv_header(path: Path) -> tuple[list[str], int]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader, [])
        rows = sum(1 for _ in reader)
    return header, rows


def check_summary_files() -> list[str]:
    errors: list[str] = []
    for rel, required in SUMMARY_REQUIRED_COLUMNS.items():
        path = ROOT / rel
        if not path.exists():
            continue
        header, rows = read_csv_header(path)
        missing = sorted(required - set(header))
        if missing:
            errors.append(f"{rel} missing columns: {', '.join(missing)}")
        if rows == 0:
            errors.append(f"{rel} has no data rows")
    return errors


def check_json_parseable() -> list[str]:
    errors: list[str] = []
    for folder in ["data/sqchip_em_1q/json", "data/sqchip_em_2q/json"]:
        path = ROOT / folder
        if not path.exists():
            continue
        for json_path in sorted(path.glob("*.json"))[:20]:
            try:
                json.loads(json_path.read_text(encoding="utf-8-sig"))
            except json.JSONDecodeError as exc:
                errors.append(f"Invalid JSON: {json_path.relative_to(ROOT)} ({exc})")
    return errors


FORBIDDEN_MARKER = "".join(["co", "dex"])


def check_no_forbidden_marker() -> list[str]:
    errors: list[str] = []
    for path in ROOT.rglob("*"):
        if path.is_file() and FORBIDDEN_MARKER in path.name.lower():
            errors.append(f"Forbidden marker in filename: {path.relative_to(ROOT)}")
    text_suffixes = {".py", ".csv", ".json", ".md", ".txt", ".cff", ".yml", ".yaml"}
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in text_suffixes:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if FORBIDDEN_MARKER in text.lower():
            errors.append(f"Forbidden marker in text: {path.relative_to(ROOT)}")
    return errors


def main() -> int:
    errors = []
    errors.extend(check_required_paths())
    errors.extend(check_summary_files())
    errors.extend(check_json_parseable())
    errors.extend(check_no_forbidden_marker())

    if errors:
        print("Repository validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("Repository validation passed.")
    for rel in SUMMARY_REQUIRED_COLUMNS:
        header, rows = read_csv_header(ROOT / rel)
        print(f"- {rel}: {rows} rows, {len(header)} columns")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
