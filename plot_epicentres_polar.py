"""Plot unique screened earthquake epicentres on an azimuthal polar map."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from obspy.geodetics import gps2dist_azimuth

from config import ACTIVE_END, ACTIVE_START, MIN_MAGNITUDE, STATION_COORDINATES
from screen_events import get_event_id, magnitude_for, origin_for


def read_unique_events(path: Path) -> list[dict[str, str]]:
    by_id: dict[str, dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            by_id.setdefault(row["event_id"], row)
    return sorted(by_id.values(), key=lambda row: row["event_time_utc"])


def fetch_global_catalog() -> list[dict[str, str]]:
    """Fetch the full magnitude-screened catalog, before distance/phase screening."""
    catalog = Client("USGS").get_events(
        starttime=UTCDateTime(ACTIVE_START),
        endtime=UTCDateTime(ACTIVE_END),
        minmagnitude=MIN_MAGNITUDE,
    )
    rows = []
    for event in catalog:
        origin = origin_for(event)
        magnitude = magnitude_for(event)
        if (
            origin is None
            or magnitude is None
            or origin.latitude is None
            or origin.longitude is None
        ):
            continue
        rows.append({
            "event_id": get_event_id(event),
            "event_time_utc": str(origin.time),
            "magnitude": f"{float(magnitude.mag):.2f}",
            "event_latitude": f"{float(origin.latitude):.5f}",
            "event_longitude": f"{float(origin.longitude):.5f}",
            "depth_km": f"{(origin.depth or 0.0) / 1000.0:.2f}",
        })
    return sorted(rows, key=lambda row: row["event_time_utc"])


def project_events(events: list[dict[str, str]], center_lat: float, center_lon: float):
    distances_deg = []
    azimuths_deg = []
    for event in events:
        distance_m, azimuth, _ = gps2dist_azimuth(
            center_lat,
            center_lon,
            float(event["event_latitude"]),
            float(event["event_longitude"]),
        )
        distances_deg.append(distance_m / 1000.0 / 111.195)
        azimuths_deg.append(azimuth)
    return np.deg2rad(azimuths_deg), distances_deg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-input", type=Path,
        default=Path("results/event_station_candidates_1A_active.csv"),
        help="XKS event-station candidate CSV; events are deduplicated by event_id",
    )
    parser.add_argument(
        "--catalog-csv", type=Path,
        default=Path("results/global_catalog_active.csv"),
        help="CSV cache of the full M>=6.3 USGS catalog for the active period",
    )
    parser.add_argument(
        "--refresh-catalog", action="store_true",
        help="Fetch the global catalog again instead of using the saved CSV",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/epicentres_polar_map.png"),
        help="Output PNG path",
    )
    args = parser.parse_args()

    candidates = read_unique_events(args.candidate_input)
    if not candidates:
        raise RuntimeError(f"No events found in {args.candidate_input}")

    if args.refresh_catalog or not args.catalog_csv.exists():
        all_events = fetch_global_catalog()
        args.catalog_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.catalog_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(all_events[0]))
            writer.writeheader()
            writer.writerows(all_events)
    else:
        all_events = read_unique_events(args.catalog_csv)
    if not all_events:
        raise RuntimeError("The USGS catalog contained no mappable events.")

    candidate_ids = {event["event_id"] for event in candidates}
    catalog_ids = {event["event_id"] for event in all_events}
    missing = candidate_ids - catalog_ids
    if missing:
        raise RuntimeError(
            "Candidate event IDs missing from the full USGS catalog: "
            + ", ".join(sorted(missing))
        )
    candidates = [event for event in all_events if event["event_id"] in candidate_ids]

    station_lats = [coords[0] for coords in STATION_COORDINATES.values()]
    station_lons = [coords[1] for coords in STATION_COORDINATES.values()]
    center_lat = float(np.mean(station_lats))
    center_lon = float(np.mean(station_lons))

    all_theta, all_distances = project_events(all_events, center_lat, center_lon)
    candidate_theta, candidate_distances = project_events(candidates, center_lat, center_lon)
    candidate_mags = np.asarray([float(event["magnitude"]) for event in candidates])
    fig = plt.figure(figsize=(14, 9), layout="constrained")
    grid = fig.add_gridspec(1, 2, width_ratios=[1.12, 0.88])
    ax = fig.add_subplot(grid[0, 0], projection="polar")
    listing = fig.add_subplot(grid[0, 1])
    listing.axis("off")

    ax.scatter(
        all_theta,
        all_distances,
        color="#9aa0a6",
        alpha=0.54,
        s=20,
        edgecolors="white",
        linewidths=0.35,
        zorder=2,
    )
    scatter = ax.scatter(
        candidate_theta,
        candidate_distances,
        c=candidate_mags,
        cmap="plasma",
        s=74,
        marker="*",
        edgecolors="black",
        linewidths=0.55,
        zorder=3,
    )
    for station, (lat, lon) in STATION_COORDINATES.items():
        distance_m, azimuth, _ = gps2dist_azimuth(center_lat, center_lon, lat, lon)
        ax.scatter(
            np.deg2rad(azimuth),
            distance_m / 1000.0 / 111.195,
            marker="^",
            s=38,
            color="#168c55",
            edgecolors="black",
            linewidths=0.5,
            zorder=5,
        )

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 180)
    ax.set_yticks(np.arange(30, 181, 30))
    ax.set_rlabel_position(225)
    ax.grid(True, alpha=0.55)
    ax.set_title(
        "Earthquake epicentres during the network period\n"
        "Azimuthal equidistant view from Sri Lanka",
        pad=22,
        fontsize=13,
    )
    ax.text(
        0.5,
        0.49,
        "PALK / MALK / HALK\nstation cluster",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=7,
        color="#146b42",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.5},
        zorder=6,
    )

    colorbar = fig.colorbar(scatter, ax=ax, pad=0.10, shrink=0.76)
    colorbar.set_label("XKS candidate magnitude")
    legend_handles = [
        Line2D(
            [], [], marker="o", linestyle="None", markersize=6,
            markerfacecolor="#9aa0a6", markeredgecolor="white",
            label=f"All catalog events M≥{MIN_MAGNITUDE:g} ({len(all_events)})",
        ),
        Line2D(
            [], [], marker="*", linestyle="None", markersize=10,
            markerfacecolor="#8a2be2", markeredgecolor="black",
            label=f"XKS candidates ({len(candidates)})",
        ),
        Line2D(
            [], [], marker="^", linestyle="None", markersize=7,
            markerfacecolor="#168c55", markeredgecolor="black",
            label="PALK / MALK / HALK stations",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower left",
        bbox_to_anchor=(-0.02, -0.13),
        fontsize=8,
    )

    listing.set_title("Mapped earthquakes", loc="left", fontsize=13, pad=14)
    listing.text(
        0.0,
        0.98,
        f"{len(all_events)} total M≥{MIN_MAGNITUDE:g} catalog events; "
        f"{len(candidates)} pass the 85°–140° XKS screen.\n"
        f"Map center: {center_lat:.3f}°N, {center_lon:.3f}°E.\n"
        "Stars mark XKS candidates; gray dots are the other catalog events.",
        transform=listing.transAxes,
        va="top",
        fontsize=8.5,
    )
    for index, event in enumerate(candidates):
        y = 0.83 - index * (0.79 / len(candidates))
        event_day = event["event_time_utc"][:10]
        listing.text(
            0.0,
            y,
            f"{event_day}  M{float(event['magnitude']):.1f}  "
            f"{event['event_id']}  ({float(event['event_latitude']):.2f}, "
            f"{float(event['event_longitude']):.2f})",
            transform=listing.transAxes,
            va="center",
            fontsize=8.5,
            family="monospace",
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(
        f"Mapped {len(all_events)} global events and highlighted "
        f"{len(candidates)} XKS candidates to {args.output}"
    )


if __name__ == "__main__":
    main()
