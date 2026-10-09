"""Query USGS and screen events for XKS geometry at PALK, MALK, and HALK."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from obspy.geodetics import gps2dist_azimuth, locations2degrees
from obspy.taup import TauPyModel

from config import (
    ACTIVE_END,
    ACTIVE_START,
    MAX_DISTANCE_DEG,
    MIN_DISTANCE_DEG,
    MIN_MAGNITUDE,
    STATION_COORDINATES,
    STATIONS,
    TAUP_MODEL,
    XKS_PHASES,
)


def get_event_id(event) -> str:
    resource_id = str(event.resource_id)
    query = parse_qs(urlparse(resource_id).query)
    if query.get("eventid"):
        return query["eventid"][0]
    tail = urlparse(resource_id).path.rstrip("/").split("/")[-1]
    return tail or str(event.origins[0].time).replace(":", "").replace("-", "")


def origin_for(event):
    return event.preferred_origin() or (event.origins[0] if event.origins else None)


def magnitude_for(event):
    return event.preferred_magnitude() or (event.magnitudes[0] if event.magnitudes else None)


def screen(start: str, end: str, min_magnitude: float, min_distance: float, max_distance: float):
    client = Client("USGS")
    model = TauPyModel(TAUP_MODEL)
    catalog = client.get_events(
        starttime=UTCDateTime(start),
        endtime=UTCDateTime(end),
        minmagnitude=min_magnitude,
    )
    rows = []

    for event in catalog:
        origin = origin_for(event)
        magnitude = magnitude_for(event)
        if origin is None or magnitude is None:
            continue
        if origin.latitude is None or origin.longitude is None:
            continue

        event_id = get_event_id(event)
        depth_km = (origin.depth or 0.0) / 1000.0

        for station_spec in STATIONS:
            network = station_spec["network"]
            station = station_spec["station"]
            station_lat, station_lon = STATION_COORDINATES[(network, station)]
            distance = locations2degrees(
                station_lat, station_lon, origin.latitude, origin.longitude
            )
            if not min_distance <= distance <= max_distance:
                continue

            arrivals = model.get_travel_times(
                source_depth_in_km=depth_km,
                distance_in_degree=distance,
                phase_list=list(XKS_PHASES),
            )
            if not arrivals:
                continue
            arrival = min(arrivals, key=lambda item: item.time)

            # With event as point 1 and receiver as point 2, the returned
            # back-azimuth points from receiver back toward the event.
            _, _, back_azimuth = gps2dist_azimuth(
                origin.latitude, origin.longitude, station_lat, station_lon
            )
            rows.append({
                "event_id": event_id,
                "event_time_utc": str(origin.time),
                "magnitude": float(magnitude.mag),
                "event_latitude": float(origin.latitude),
                "event_longitude": float(origin.longitude),
                "depth_km": depth_km,
                "network": network,
                "station": station,
                "distance_deg": float(distance),
                "back_azimuth_deg": float(back_azimuth),
                "phase": arrival.name,
                "travel_time_s": float(arrival.time),
                "arrival_time_utc": str(origin.time + arrival.time),
            })

    columns = [
        "event_id", "event_time_utc", "magnitude", "event_latitude",
        "event_longitude", "depth_km", "network", "station", "distance_deg",
        "back_azimuth_deg", "phase", "travel_time_s", "arrival_time_utc",
    ]
    table = pd.DataFrame(rows, columns=columns)
    if not table.empty:
        table = table.sort_values(["event_time_utc", "station"]).reset_index(drop=True)
    return catalog, table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=ACTIVE_START, help="UTC start time, inclusive")
    parser.add_argument("--end", default=ACTIVE_END, help="UTC end time, exclusive")
    parser.add_argument("--min-magnitude", type=float, default=MIN_MAGNITUDE)
    parser.add_argument("--min-distance", type=float, default=MIN_DISTANCE_DEG)
    parser.add_argument("--max-distance", type=float, default=MAX_DISTANCE_DEG)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/event_station_candidates_1A_active.csv"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("results/event_screen_summary.json"),
    )
    args = parser.parse_args()

    catalog, table = screen(
        args.start, args.end, args.min_magnitude, args.min_distance, args.max_distance
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)

    unique_events = int(table["event_id"].nunique()) if not table.empty else 0
    summary = {
        "source": "USGS FDSN event service",
        "start_utc_inclusive": args.start,
        "end_utc_exclusive": args.end,
        "min_magnitude": args.min_magnitude,
        "distance_window_deg": [args.min_distance, args.max_distance],
        "phases": list(XKS_PHASES),
        "taup_model": TAUP_MODEL,
        "global_catalog_events": len(catalog),
        "unique_xks_geometry_events": unique_events,
        "event_station_pairs": len(table),
        "output_csv": str(args.output),
    }
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Global events M >= {args.min_magnitude}: {len(catalog)}")
    print(f"Unique XKS-geometry events: {unique_events}")
    print(f"Eligible event-station pairs: {len(table)}")
    print(f"Candidate table: {args.output}")
    print(f"Summary: {args.summary}")


if __name__ == "__main__":
    main()
