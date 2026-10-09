"""Download a continuous window from every active GEOFON 1A station."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from obspy import Stream, UTCDateTime
from obspy.clients.fdsn import Client


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--start",
        required=True,
        help="Window start in UTC, e.g. 2016-05-27T04:08:43",
    )
    parser.add_argument("--duration-hours", type=float, default=10.0)
    parser.add_argument("--output-dir", type=Path, default=Path("data/sl_1a_continuous"))
    parser.add_argument("--channel", default="HH?", help="GEOFON 1A channel pattern")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    start = UTCDateTime(args.start)
    if args.duration_hours <= 0:
        raise ValueError("duration-hours must be positive")
    end = start + args.duration_hours * 3600
    client = Client("GEOFON")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    inventory = client.get_stations(
        network="1A",
        station="*",
        location="*",
        channel=args.channel,
        starttime=start,
        endtime=end,
        level="response",
    )
    if not inventory or not len(inventory[0]):
        raise RuntimeError("GEOFON returned no 1A stations for this time window.")

    stamp = start.strftime("%Y%m%dT%H%M%S")
    stationxml_path = args.output_dir / f"1A_{stamp}_{args.duration_hours:g}h_stations.xml"
    inventory.write(str(stationxml_path), format="STATIONXML")
    print(f"Saved response metadata: {stationxml_path}")

    log_rows = []
    for network in inventory:
        for station in network:
            output_path = args.output_dir / (
                f"1A_{station.code}_{stamp}_{args.duration_hours:g}h.mseed"
            )
            if output_path.exists() and not args.overwrite:
                log_rows.append({
                    "network": network.code,
                    "station": station.code,
                    "status": "skipped_existing",
                    "trace_count": "",
                    "start_utc": str(start),
                    "end_utc": str(end),
                    "path": str(output_path),
                    "error": "",
                })
                print(f"{station.code}: already exists; skipped")
                continue

            try:
                stream: Stream = client.get_waveforms(
                    network=network.code,
                    station=station.code,
                    location="*",
                    channel=args.channel,
                    starttime=start,
                    endtime=end,
                )
                stream.merge(method=1, fill_value=None)
                # Keep gaps as separate unmasked segments; MiniSEED cannot
                # serialize NumPy masked arrays as one continuous trace.
                if any(np.ma.isMaskedArray(trace.data) for trace in stream):
                    stream = stream.split()
                stream.sort()
                if not stream:
                    raise RuntimeError("No samples remained after splitting waveform gaps.")
                stream.write(str(output_path), format="MSEED")
                by_component = {}
                for trace in stream:
                    component = trace.stats.channel[-1]
                    by_component[component] = by_component.get(component, 0) + trace.stats.npts
                coverage = []
                for component in "ENZ":
                    sample_rate = next(
                        (
                            channel.sample_rate
                            for channel in station.channels
                            if channel.code.endswith(component) and channel.sample_rate
                        ),
                        None,
                    )
                    if sample_rate:
                        expected = int(round((end - start) * sample_rate)) + 1
                        coverage.append(min(1.0, by_component.get(component, 0) / expected))
                status = "downloaded" if len(coverage) == 3 and min(coverage) >= 0.99 else "partial"
                log_rows.append({
                    "network": network.code,
                    "station": station.code,
                    "status": status,
                    "trace_count": str(len(stream)),
                    "start_utc": str(start),
                    "end_utc": str(end),
                    "path": str(output_path),
                    "error": "" if status == "downloaded" else f"component coverage fractions ENZ={coverage}",
                })
                print(
                    f"{station.code}: {status}, {len(stream)} traces "
                    f"({output_path.stat().st_size:,} bytes); ENZ coverage={coverage}"
                )
            except Exception as error:
                log_rows.append({
                    "network": network.code,
                    "station": station.code,
                    "status": "unavailable_or_error",
                    "trace_count": "0",
                    "start_utc": str(start),
                    "end_utc": str(end),
                    "path": "",
                    "error": str(error).replace("\n", " "),
                })
                print(f"{station.code}: unavailable ({error})")

    log_path = args.output_dir / f"1A_{stamp}_{args.duration_hours:g}h_download_log.csv"
    with log_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(log_rows[0]))
        writer.writeheader()
        writer.writerows(log_rows)
    downloaded = sum(row["status"] == "downloaded" for row in log_rows)
    print(f"Stations with data: {downloaded}/{len(log_rows)}")
    print(f"Station log: {log_path}")


if __name__ == "__main__":
    main()
