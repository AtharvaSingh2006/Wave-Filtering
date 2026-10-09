"""Apply a conservative, paper-informed pre-screen to one event's SWS results.

This is deliberately not a replacement for CNN-SWS or a manual A/B/C/N rank.
The current measurements come from SWSPy's eigenvalue (EV) method, while the
Zhang & Gao CNN was trained on four Silver-Chan transverse-energy waveforms.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def as_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return math.nan


def nearest_null_axis_separation(fast_axis_deg: float, back_azimuth_deg: float) -> float:
    """Smallest line-angle separation to the event's fast or slow polarization.

    A null/near-null geometry occurs when the event back azimuth is close to
    either the fast axis or its orthogonal slow axis. Both axes are unoriented
    lines, so the separation is periodic over 90 degrees.
    """
    delta = (fast_axis_deg - back_azimuth_deg + 45.0) % 90.0 - 45.0
    return abs(delta)


def classify_row(row: dict[str, str], coverage: dict[str, str]) -> dict[str, str]:
    status = row.get("measurement_status", "")
    result = dict(row)
    result["sws_method"] = "SWSPy_EV"
    result["nearest_null_axis_separation_deg"] = ""
    result["paper_uncertainty_gate"] = "not_evaluated"
    result["cnn_probability_acceptable"] = ""
    result["cnn_quality_class"] = "not_run"

    if status != "computed":
        cover_status = coverage.get("status", "")
        if cover_status == "no_raw_waveform":
            label, reason = "no_waveform", "No raw waveform was downloaded for this station."
        elif cover_status == "no_complete_XKS_window":
            label, reason = "no_complete_window", "No complete 50-second XKS window was available."
        elif status == "failed":
            label, reason = "no_split_solution", row.get("error", "SWSPy did not return a solution.")
        else:
            label, reason = "not_assessed", "No finite SWSPy splitting estimate is available."
        result["interpretation_status"] = label
        result["interpretation_reason"] = reason
        return result

    phi = as_float(row, "phi_from_N")
    baz = as_float(row, "back_azimuth_deg")
    phi_err = as_float(row, "phi_err")
    dt = as_float(row, "dt")
    dt_err = as_float(row, "dt_err")
    separation = nearest_null_axis_separation(phi, baz)
    result["nearest_null_axis_separation_deg"] = f"{separation:.2f}"

    # Zhang & Gao (2022) excluded near-null geometries (<15 degrees from the
    # fast or slow polarization) and large-uncertainty / large-delay picks.
    # Apply these only as an exclusion/triage screen to EV results, not as a
    # formal CNN acceptance rank because the measurement methods differ.
    uncertainty_pass = (
        math.isfinite(phi_err) and phi_err < 15.0
        and math.isfinite(dt_err) and dt_err < 1.5
        and math.isfinite(dt) and dt <= 2.0
    )
    result["paper_uncertainty_gate"] = "pass" if uncertainty_pass else "fail"

    if uncertainty_pass and separation < 15.0:
        label = "near_null_geometry_excluded"
        reason = (
            f"Fast/slow orientation is only {separation:.1f}° from event back-azimuth; "
            "paper-based screen excludes this near-null geometry from fast-axis interpretation."
        )
    elif not uncertainty_pass:
        label = "unresolved_high_uncertainty"
        reason = (
            "Fails the paper's uncertainty/delay screening limits (σφ < 15°, "
            "σδt < 1.5 s, δt ≤ 2 s); the EV result is not a defensible fast-axis pick."
        )
    else:
        label = "manual_cnn_review_required"
        reason = (
            "Passes numeric pre-screen only; original/corrected R-T SNR, particle motion, "
            "contour uniqueness, and CNN/manual rank still need review."
        )
    result["interpretation_status"] = label
    result["interpretation_reason"] = reason
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-id", default="us10006exl")
    parser.add_argument("--results-dir", type=Path, default=Path("results/sws_1a_10h"))
    args = parser.parse_args()

    stem = f"1A_{args.event_id}"
    measurements = read_rows(args.results_dir / f"{stem}_sws_measurements.csv")
    coverage_rows = read_rows(args.results_dir / f"{stem}_processing_coverage.csv")
    coverage = {row.get("station", ""): row for row in coverage_rows}
    measurements_by_station = {row.get("station", ""): row for row in measurements}
    for row in coverage_rows:
        station = row.get("station", "")
        if station and station not in measurements_by_station:
            measurements_by_station[station] = {
                "event_id": args.event_id,
                "station": station,
                "phase": row.get("phase", ""),
                "arrival_utc": row.get("arrival_utc", ""),
                "measurement_status": "not_attempted",
                "error": row.get("error", ""),
            }
    screened = [
        classify_row(row, coverage.get(row.get("station", ""), {}))
        for row in measurements_by_station.values()
    ]
    output = args.results_dir / f"{stem}_paper_quality_screen.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in screened for key in row))
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(screened)

    counts: dict[str, int] = {}
    for row in screened:
        label = row["interpretation_status"]
        counts[label] = counts.get(label, 0) + 1
    print(f"Paper-informed triage (not a CNN rank): {output}")
    for label, count in sorted(counts.items()):
        print(f"  {label}: {count}")
    print("CNN A/B/C/N class: not run; SWSPy EV waveforms do not match the paper's Silver-Chan CNN inputs.")


if __name__ == "__main__":
    main()
