"""Map the 1A stations and the measured versus expected SKS polarization."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pyproj import Geod


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-id", default="us10006exl")
    parser.add_argument("--measurement-dir", type=Path, default=Path("results/sws_1a_10h"))
    parser.add_argument("--station-csv", type=Path, default=Path("results/network_1A_stations.csv"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    triage_path = args.measurement_dir / f"1A_{args.event_id}_paper_quality_screen.csv"
    triage = read_rows(triage_path)
    stations = read_rows(args.station_csv)
    triage_by_code = {row["station"]: row for row in triage}

    styles = {
        "near_null_geometry_excluded": {"marker": "D", "color": "#c2413a", "label": "Near-null geometry — exclude axis"},
        "unresolved_high_uncertainty": {"marker": "x", "color": "#d97706", "label": "Unresolved — uncertainty too large"},
        "manual_cnn_review_required": {"marker": "o", "color": "#2563eb", "label": "Manual/CNN review required"},
        "no_split_solution": {"marker": "x", "color": "#667085", "label": "No splitting estimate"},
        "no_complete_window": {"marker": "^", "color": "#98a2b3", "label": "No complete phase window"},
        "no_waveform": {"marker": "s", "color": "#98a2b3", "label": "No raw waveform"},
        "not_assessed": {"marker": "^", "color": "#98a2b3", "label": "Not assessed"},
    }
    counts: dict[str, int] = {}
    for row in triage:
        key = row.get("interpretation_status", "not_assessed")
        counts[key] = counts.get(key, 0) + 1

    projection = ccrs.UTM(zone=44)
    geographic = ccrs.PlateCarree()
    fig = plt.figure(figsize=(10, 9), layout="constrained")
    ax = fig.add_subplot(1, 1, 1, projection=projection)
    ax.set_extent([79.3, 82.1, 5.3, 10.2], crs=geographic)
    ax.add_feature(cfeature.OCEAN.with_scale("110m"), facecolor="#e8f2f8", zorder=0)
    ax.add_feature(cfeature.LAND.with_scale("110m"), facecolor="#f3f0e8", zorder=0)
    ax.coastlines(resolution="110m", linewidth=0.65, color="#4e5b62", zorder=1)
    grid = ax.gridlines(crs=geographic, draw_labels=True, linewidth=0.4, color="#777777", alpha=0.5, linestyle=":")
    grid.top_labels = False
    grid.right_labels = False

    legend_handles = []
    used_statuses: set[str] = set()
    label_offsets = [(6, 2), (6, 9), (6, -5), (-18, 8), (-18, -5), (11, 12), (11, -10), (-20, 13)]
    geod = Geod(ellps="WGS84")
    polarization_handles = [
        Line2D([], [], color="#1479b8", linewidth=2.0, label="Corrected E/N PCA (−2 to +8 s)"),
        Line2D([], [], color="#e07b24", linewidth=1.2, linestyle="--", label="Expected radial line (station→event BAZ)"),
        Line2D([], [], color="#b42318", marker="o", linestyle="None", markersize=5,
               label="PCA mismatch >15° or variance ratio >0.10"),
    ]
    for index, station in enumerate(stations):
        code = station["station"]
        lon, lat = float(station["longitude"]), float(station["latitude"])
        row = triage_by_code.get(code, {})
        status = row.get("interpretation_status", "not_assessed")
        style = styles.get(status, styles["not_assessed"])
        ax.scatter(
            [lon], [lat], transform=geographic, marker=style["marker"], s=35,
            color=style["color"], linewidths=1.1, zorder=4,
        )
        pca_pol = float(row.get("src_pol_numpy_from_N") or "nan")
        baz = float(row.get("back_azimuth_deg") or "nan")
        pol_delta = float(row.get("src_pol_deviation_from_radial_BAZ_deg") or "nan")
        if pca_pol == pca_pol and baz == baz:
            # A short geodesic line glyph makes the clockwise-from-N angle visible
            # without treating a particle-motion direction as a fast axis.
            half_length_m = 4500.0
            for angle_deg, color, linewidth, linestyle in (
                (pca_pol, "#1479b8", 2.2, "-"),
                (baz, "#e07b24", 1.25, "--"),
            ):
                lon1, lat1, _ = geod.fwd(lon, lat, angle_deg + 180.0, half_length_m)
                lon2, lat2, _ = geod.fwd(lon, lat, angle_deg, half_length_m)
                ax.plot([lon1, lon2], [lat1, lat2], transform=geographic, color=color,
                        linewidth=linewidth, linestyle=linestyle, zorder=6)
            linearity_ratio = float(row.get("src_pol_eigenvalue_minor_major_ratio") or "nan")
            if pol_delta > 15.0 or linearity_ratio > 0.10:
                ax.scatter([lon], [lat], transform=geographic, facecolors="none",
                           edgecolors="#b42318", linewidths=1.4, s=80, zorder=7)
        ax.annotate(
            code, xy=(lon, lat), xycoords=geographic._as_mpl_transform(ax),
            xytext=label_offsets[index % len(label_offsets)], textcoords="offset points",
            fontsize=5.6, color="#46535d", zorder=5,
        )
        if status not in used_statuses:
            legend_handles.append(Line2D(
                [], [], marker=style["marker"], linestyle="None", markersize=5,
                markerfacecolor=(style["color"] if style["marker"] in {"D", "o", "s"} else "none"),
                markeredgecolor=style["color"], label=style["label"],
            ))
            used_statuses.add(status)

    near_null = counts.get("near_null_geometry_excluded", 0)
    unresolved = counts.get("unresolved_high_uncertainty", 0)
    candidate = counts.get("manual_cnn_review_required", 0)
    station_legend = ax.legend(handles=legend_handles, loc="lower left", fontsize=7.0, framealpha=0.95)
    ax.add_artist(station_legend)
    ax.legend(handles=polarization_handles, loc="lower right", fontsize=7.0, framealpha=0.95)
    ax.set_title(
        f"1A station polarization check — {args.event_id}\n"
        f"{near_null} near-null geometry exclusions | {unresolved} unresolved | "
        f"{candidate} require manual/CNN review",
        fontsize=12,
        pad=10,
    )
    ax.text(
        0.5, -0.10,
        "Blue = NumPy PCA of corrected E/N particle motion in the primary −2 to +8 s XKS window; its angle stability across four windows is reported in the CSV/PDF. "
        "Orange dashed = expected radial polarization from independently checked WGS84 BAZ. "
        "Red rings flag >15° radial mismatch or a high ellipse ratio (>0.10). These lines are polarization axes, not splitting fast axes. "
        "Fast axes remain withheld where the SWSPy EV estimate is unresolved.",
        transform=ax.transAxes, ha="center", va="top", fontsize=7.5, color="#475569",
    )

    output = args.output or args.measurement_dir / f"1A_{args.event_id}_sws_station_map.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Mapped {len(stations)} stations with measured and expected polarization axes: {output}")


if __name__ == "__main__":
    main()
