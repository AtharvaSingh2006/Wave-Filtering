"""Cut aligned two-hour event windows from locally supplied waveform files."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
from obspy import Stream, UTCDateTime, read

from config import CUT_SECONDS, PRE_EVENT_SECONDS, STATIONS, SUPPORTED_SUFFIXES


def input_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(
        item for item in path.rglob("*")
        if item.is_file() and item.suffix.lower() in SUPPORTED_SUFFIXES
    )


def safe_tag(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value))


def select_station(stream, spec):
    selected = stream.select(network=spec["network"], station=spec["station"])
    if spec["location"] != "*":
        selected = selected.select(location=spec["location"])
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=Path("data/incoming"),
        help="A waveform file or a folder containing MiniSEED/SAC files",
    )
    parser.add_argument(
        "--events", type=Path,
        default=Path("results/event_station_candidates_1A_active.csv"),
        help="CSV produced by screen_events.py",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/cuts_2h"))
    parser.add_argument("--window-seconds", type=float, default=CUT_SECONDS)
    parser.add_argument("--pre-event-seconds", type=float, default=PRE_EVENT_SECONDS)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    files = input_files(args.input)
    if not files:
        parser.error(
            f"No supported waveform files found under {args.input}. "
            "Place your MiniSEED or SAC files in data/incoming or pass --input."
        )
    events = pd.read_csv(args.events)
    required = {"event_id", "event_time_utc"}
    if not required.issubset(events.columns):
        parser.error(f"Event CSV must contain columns: {sorted(required)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    unique_events = events.drop_duplicates("event_id").sort_values("event_time_utc")
    log_rows = []

    for _, event in unique_events.iterrows():
        event_id = str(event["event_id"])
        origin = UTCDateTime(event["event_time_utc"])
        start = origin - args.pre_event_seconds
        end = start + args.window_seconds
        filename = (
            f"{origin.strftime('%Y%m%dT%H%M%S')}_{safe_tag(event_id)}_2h.mseed"
        )
        output_path = args.output_dir / filename

        if output_path.exists() and not args.overwrite:
            print(f"Skip existing: {output_path}")
            log_rows.append({
                "event_id": event_id, "station": "ALL", "status": "already_present",
                "trace_count": None, "start_utc": str(start), "end_utc": str(end),
                "path": str(output_path),
            })
            continue

        event_stream = Stream()
        station_counts = {spec["station"]: 0 for spec in STATIONS}
        read_errors = 0

        for source_file in files:
            try:
                partial = read(
                    str(source_file),
                    starttime=start,
                    endtime=end,
                )
            except Exception:
                read_errors += 1
                continue

            for spec in STATIONS:
                selected = select_station(partial, spec)
                if not selected:
                    continue
                selected.trim(starttime=start, endtime=end, pad=False)
                event_stream += selected
                station_counts[spec["station"]] += len(selected)

        event_stream.sort()
        for spec in STATIONS:
            station = spec["station"]
            status = "cut" if station_counts[station] else "no_overlap"
            log_rows.append({
                "event_id": event_id,
                "station": station,
                "status": status,
                "trace_count": station_counts[station],
                "start_utc": str(start),
                "end_utc": str(end),
                "path": str(output_path) if station_counts[station] else "",
            })

        if event_stream:
            event_stream.write(str(output_path), format="MSEED")
            print(
                f"{event_id}: {len(event_stream)} traces; "
                f"stations={station_counts}; saved {output_path}"
            )
        else:
            print(f"{event_id}: no matching traces (input read errors: {read_errors})")

    log_path = Path("results/cut_log.csv")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(log_rows).to_csv(log_path, index=False)
    print(f"Cut log: {log_path}")


if __name__ == "__main__":
    main()
