"""Download aligned two-hour waveform cuts and StationXML with ObsPy."""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import pandas as pd
from obspy import Stream, UTCDateTime
from obspy.clients.fdsn import Client

from config import ACTIVE_END, ACTIVE_START, CUT_SECONDS, PRE_EVENT_SECONDS, STATIONS

PROVIDERS = {"II": "EARTHSCOPE", "GE": "GEOFON"}


def safe_tag(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value))


def event_tag(row) -> str:
    origin = UTCDateTime(row["event_time_utc"])
    return f"{origin.strftime('%Y%m%dT%H%M%S')}_{safe_tag(row['event_id'])}"


def fetch_inventories(clients, outpath: Path):
    inventories = []
    for spec in STATIONS:
        inventory = clients[spec["network"]].get_stations(
            network=spec["network"],
            station=spec["station"],
            location=spec["location"],
            channel=spec["channel"],
            starttime=UTCDateTime(ACTIVE_START),
            endtime=UTCDateTime(ACTIVE_END),
            level="response",
        )
        inventories.append(inventory)
        print(
            f"Metadata {spec['network']}.{spec['station']}: "
            f"{sum(len(sta.channels) for net in inventory for sta in net)} channel epochs",
            flush=True,
        )

    combined = inventories[0]
    for inventory in inventories[1:]:
        combined += inventory
    outpath.parent.mkdir(parents=True, exist_ok=True)
    combined.write(str(outpath), format="STATIONXML")
    print(f"Saved response metadata: {outpath}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--events", type=Path,
        default=Path("results/event_station_candidates_1A_active.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/cuts_2h"))
    parser.add_argument(
        "--metadata", type=Path, default=Path("data/metadata/sri_lanka_response.xml")
    )
    parser.add_argument("--window-seconds", type=float, default=CUT_SECONDS)
    parser.add_argument("--pre-event-seconds", type=float, default=PRE_EVENT_SECONDS)
    parser.add_argument("--pause-seconds", type=float, default=0.25)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    events = pd.read_csv(args.events)
    required = {"event_id", "event_time_utc"}
    if not required.issubset(events.columns):
        parser.error(f"Event CSV must contain columns: {sorted(required)}")
    events = events.drop_duplicates("event_id").sort_values("event_time_utc")

    clients = {network: Client(provider) for network, provider in PROVIDERS.items()}
    fetch_inventories(clients, args.metadata)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    log_rows = []
    for index, (_, event) in enumerate(events.iterrows(), start=1):
        event_id = str(event["event_id"])
        origin = UTCDateTime(event["event_time_utc"])
        start = origin - args.pre_event_seconds
        end = start + args.window_seconds
        output = args.output_dir / f"{event_tag(event)}_2h.mseed"

        if output.exists() and output.stat().st_size and not args.overwrite:
            print(f"[{index}/{len(events)}] already present: {output.name}", flush=True)
            log_rows.append({
                "event_id": event_id, "station": "ALL", "status": "already_present",
                "trace_count": None, "start_utc": str(start), "end_utc": str(end),
                "path": str(output), "error": "",
            })
            continue

        combined = Stream()
        for spec in STATIONS:
            try:
                station_stream = clients[spec["network"]].get_waveforms(
                    network=spec["network"],
                    station=spec["station"],
                    location=spec["location"],
                    channel=spec["channel"],
                    starttime=start,
                    endtime=end,
                )
                station_stream.trim(starttime=start, endtime=end, pad=False)
                combined += station_stream
                status = "downloaded" if len(station_stream) else "empty"
                error = ""
                print(
                    f"  {spec['network']}.{spec['station']}: "
                    f"{len(station_stream)} traces",
                    flush=True,
                )
            except Exception as exc:
                station_stream = Stream()
                status = "unavailable_or_error"
                error = str(exc)
                print(f"  {spec['network']}.{spec['station']}: ERROR {exc}", flush=True)

            log_rows.append({
                "event_id": event_id,
                "station": spec["station"],
                "status": status,
                "trace_count": len(station_stream),
                "start_utc": str(start),
                "end_utc": str(end),
                "path": str(output) if len(station_stream) else "",
                "error": error,
            })
            time.sleep(args.pause_seconds)

        if combined:
            combined.sort()
            combined.write(str(output), format="MSEED")
            print(
                f"[{index}/{len(events)}] wrote {output.name}: "
                f"{len(combined)} traces, {output.stat().st_size:,} bytes",
                flush=True,
            )
        else:
            print(f"[{index}/{len(events)}] no traces for {event_id}", flush=True)

    log_path = Path("results/waveform_download_log.csv")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(log_rows).to_csv(log_path, index=False)
    print(f"Download log: {log_path}", flush=True)


if __name__ == "__main__":
    main()
