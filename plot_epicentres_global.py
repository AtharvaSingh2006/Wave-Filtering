"""Plot the screened earthquake epicentres on a global Robinson map."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from obspy import UTCDateTime
from obspy.clients.fdsn import Client

from config import ACTIVE_END, ACTIVE_START, MIN_MAGNITUDE, STATION_COORDINATES
from plot_epicentres_polar import fetch_global_catalog, read_unique_events


def load_1a_stations(path: Path, refresh: bool = False) -> list[dict[str, str]]:
    if path.exists() and not refresh:
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    inventory = Client("GEOFON").get_stations(
        network="1A",
        starttime=UTCDateTime(ACTIVE_START),
        endtime=UTCDateTime(ACTIVE_END),
        level="station",
    )
    rows = []
    seen = set()
    for network in inventory:
        for station in network:
            if station.code in seen:
                continue
            seen.add(station.code)
            rows.append({
                "network": network.code,
                "station": station.code,
                "latitude": f"{station.latitude:.6f}",
                "longitude": f"{station.longitude:.6f}",
                "elevation_m": f"{station.elevation:.1f}",
                "site_name": station.site.name or "",
            })
    rows.sort(key=lambda row: row["station"])
    if not rows:
        raise RuntimeError("GEOFON returned no 1A stations for the active period.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-input",
        type=Path,
        default=Path("results/event_station_candidates_1A_active.csv"),
        help="XKS event-station candidate CSV",
    )
    parser.add_argument(
        "--catalog-csv",
        type=Path,
        default=Path("results/global_catalog_active.csv"),
        help="Saved full global catalog; downloaded and cached if missing",
    )
    parser.add_argument("--refresh-catalog", action="store_true")
    parser.add_argument(
        "--highlight-event",
        help="Draw a labeled ring around one catalog event (for example us10006exl)",
    )
    parser.add_argument(
        "--station-csv",
        type=Path,
        default=Path("results/network_1A_stations.csv"),
        help="Saved GEOFON 1A station coordinates; fetched and cached if missing",
    )
    parser.add_argument("--refresh-stations", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/epicentres_global_map.png"),
        help="Output image path",
    )
    args = parser.parse_args()

    candidates = read_unique_events(args.candidate_input)
    if not candidates:
        raise RuntimeError(f"No candidate events found in {args.candidate_input}")

    if args.refresh_catalog or not args.catalog_csv.exists():
        all_events = fetch_global_catalog()
        args.catalog_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.catalog_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(all_events[0]))
            writer.writeheader()
            writer.writerows(all_events)
    else:
        all_events = read_unique_events(args.catalog_csv)

    candidate_ids = {event["event_id"] for event in candidates}
    catalog_ids = {event["event_id"] for event in all_events}
    missing = candidate_ids - catalog_ids
    if missing:
        raise RuntimeError("Candidates missing from the global catalog: " + ", ".join(sorted(missing)))
    candidates = [event for event in all_events if event["event_id"] in candidate_ids]
    highlighted = None
    if args.highlight_event:
        highlighted = next(
            (event for event in all_events if event["event_id"] == args.highlight_event),
            None,
        )
        if highlighted is None:
            raise RuntimeError(f"Event {args.highlight_event} not found in {args.catalog_csv}")
    array_stations = load_1a_stations(args.station_csv, refresh=args.refresh_stations)

    lonlat = ccrs.PlateCarree()
    fig = plt.figure(figsize=(14, 8.5), layout="constrained")
    # Center on the Pacific so the many ±180° events are not split at the map edge.
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson(central_longitude=180))
    ax.set_global()
    ax.add_feature(cfeature.OCEAN.with_scale("110m"), facecolor="#e8f2f8", zorder=0)
    ax.add_feature(cfeature.LAND.with_scale("110m"), facecolor="#f3f0e8", zorder=0)
    ax.coastlines(resolution="110m", linewidth=0.55, color="#454545", zorder=1)
    ax.add_feature(
        cfeature.BORDERS.with_scale("110m"),
        edgecolor="#888888",
        linewidth=0.3,
        zorder=1,
    )
    ax.gridlines(
        crs=lonlat,
        draw_labels=False,
        xlocs=np.arange(-180, 181, 30),
        ylocs=np.arange(-90, 91, 30),
        linewidth=0.45,
        color="#888888",
        alpha=0.45,
        linestyle=":",
        zorder=1,
    )

    all_lons = [float(event["event_longitude"]) for event in all_events]
    all_lats = [float(event["event_latitude"]) for event in all_events]
    ax.scatter(
        all_lons,
        all_lats,
        transform=lonlat,
        s=25,
        color="#89929a",
        alpha=0.62,
        edgecolors="white",
        linewidths=0.3,
        zorder=2,
    )

    candidate_lons = [float(event["event_longitude"]) for event in candidates]
    candidate_lats = [float(event["event_latitude"]) for event in candidates]
    candidate_mags = [float(event["magnitude"]) for event in candidates]
    points = ax.scatter(
        candidate_lons,
        candidate_lats,
        transform=lonlat,
        c=candidate_mags,
        cmap="plasma",
        s=82,
        marker="*",
        edgecolors="black",
        linewidths=0.55,
        zorder=3,
    )
    if highlighted is not None:
        event_lon = float(highlighted["event_longitude"])
        event_lat = float(highlighted["event_latitude"])
        ax.scatter(
            event_lon,
            event_lat,
            transform=lonlat,
            marker="o",
            s=230,
            facecolors="none",
            edgecolors="#00a6a6",
            linewidths=2.2,
            zorder=5,
        )
        ax.annotate(
            f"{args.highlight_event} (M{float(highlighted['magnitude']):.1f})",
            xy=(event_lon, event_lat),
            xycoords=lonlat._as_mpl_transform(ax),
            xytext=(9, -15),
            textcoords="offset points",
            fontsize=8,
            weight="bold",
            color="#006b6b",
            zorder=6,
        )

    station_lats = [coords[0] for coords in STATION_COORDINATES.values()]
    station_lons = [coords[1] for coords in STATION_COORDINATES.values()]
    station_center = (float(np.mean(station_lats)), float(np.mean(station_lons)))
    ax.scatter(
        station_center[1],
        station_center[0],
        transform=lonlat,
        marker="^",
        s=75,
        color="#168c55",
        edgecolors="black",
        linewidths=0.6,
        zorder=4,
    )
    ax.annotate(
        "PALK / MALK / HALK",
        xy=(station_center[1], station_center[0]),
        xycoords=lonlat._as_mpl_transform(ax),
        xytext=(9, 8),
        textcoords="offset points",
        fontsize=8,
        weight="bold",
        color="#145c3c",
        zorder=5,
    )

    array_lons = [float(station["longitude"]) for station in array_stations]
    array_lats = [float(station["latitude"]) for station in array_stations]
    ax.scatter(
        array_lons,
        array_lats,
        transform=lonlat,
        marker="D",
        s=25,
        color="#e67700",
        edgecolors="black",
        linewidths=0.3,
        alpha=0.9,
        zorder=4,
    )

    handles = [
        Line2D(
            [], [], marker="o", linestyle="None", markersize=6,
            markerfacecolor="#89929a", markeredgecolor="white",
            label=f"All global events M≥{MIN_MAGNITUDE:g} ({len(all_events)})",
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
        Line2D(
            [], [], marker="D", linestyle="None", markersize=6,
            markerfacecolor="#e67700", markeredgecolor="black",
            label=f"GEOFON 1A stations ({len(array_stations)})",
        ),
    ]
    if highlighted is not None:
        handles.append(
            Line2D(
                [], [], marker="o", linestyle="None", markersize=10,
                markerfacecolor="none", markeredgecolor="#00a6a6",
                markeredgewidth=2, label=f"Selected 10-hour event ({args.highlight_event})",
            )
        )
    ax.legend(handles=handles, loc="lower left", fontsize=9, framealpha=0.95)
    colorbar = fig.colorbar(points, ax=ax, orientation="horizontal", pad=0.055, shrink=0.55)
    colorbar.set_label("Magnitude of XKS candidates")
    ax.set_title(
        "Earthquake epicentres during the 1A network period\n"
        f"All M≥{MIN_MAGNITUDE:g} events; XKS candidates highlighted",
        fontsize=14,
        pad=12,
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
