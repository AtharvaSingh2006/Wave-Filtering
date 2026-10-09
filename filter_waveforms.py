"""Detrend and bandpass-filter a local event cut; optionally correct/rotate with StationXML."""

from __future__ import annotations

import argparse
from pathlib import Path

from obspy import Stream, read, read_inventory

from config import FILTER_MAX_HZ, FILTER_MIN_HZ, STATIONS


def select_station(stream, spec):
    selected = stream.select(network=spec["network"], station=spec["station"])
    if spec["location"] != "*":
        selected = selected.select(location=spec["location"])
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--input", type=Path, help="One two-hour MiniSEED cut")
    source_group.add_argument(
        "--input-dir", type=Path,
        help="Process every MiniSEED cut in this directory",
    )
    parser.add_argument(
        "--stationxml", type=Path,
        help="Response-level StationXML covering the cut; enables response removal and ZNE rotation",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--freq-min", type=float, default=FILTER_MIN_HZ)
    parser.add_argument("--freq-max", type=float, default=FILTER_MAX_HZ)
    parser.add_argument("--water-level", type=float, default=60.0)
    args = parser.parse_args()

    inventory = read_inventory(str(args.stationxml)) if args.stationxml else None
    if args.input:
        inputs = [args.input]
    else:
        inputs = sorted(args.input_dir.glob("*.mseed"))
    if not inputs:
        raise FileNotFoundError("No .mseed files found to process.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for input_path in inputs:
        source = read(str(input_path))
        processed = Stream()

        for spec in STATIONS:
            station_stream = select_station(source, spec).copy()
            if not station_stream:
                continue

            station_stream.detrend("demean")
            station_stream.detrend("linear")
            station_stream.taper(max_percentage=0.02)

            if inventory is not None:
                station_stream.remove_response(
                    inventory=inventory,
                    output="VEL",
                    pre_filt=[
                        max(args.freq_min * 0.5, 0.001),
                        args.freq_min,
                        args.freq_max,
                        args.freq_max * 1.2,
                    ],
                    water_level=args.water_level,
                )
                station_stream.rotate(method="->ZNE", inventory=inventory)
            else:
                print(
                    f"Warning: {spec['station']} remains in counts and instrument axes. "
                    "Provide StationXML before interpreting amplitudes or running SWS."
                )

            nyquist = min(trace.stats.sampling_rate / 2 for trace in station_stream)
            if args.freq_max >= nyquist:
                raise ValueError(
                    f"freq-max {args.freq_max} Hz must be below {spec['station']} "
                    f"Nyquist frequency {nyquist} Hz"
                )
            station_stream.filter(
                "bandpass",
                freqmin=args.freq_min,
                freqmax=args.freq_max,
                corners=4,
                zerophase=True,
            )
            processed += station_stream

        if not processed:
            print(f"Skipping {input_path}: no target-station traces found.")
            continue

        processed.sort()
        output = args.output_dir / f"{input_path.stem}_filtered.mseed"
        processed.write(str(output), format="MSEED")
        print(f"Saved filtered stream: {output}")
        if inventory is None:
            print("This is a display/inspection filter only, not response-corrected SWS input.")


if __name__ == "__main__":
    main()
