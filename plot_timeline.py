"""Plot PALK, MALK, and HALK traces against a shared UTC time axis."""

from __future__ import annotations

import argparse
from datetime import timezone
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from obspy import UTCDateTime, read

from config import STATIONS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="One event MiniSEED cut")
    parser.add_argument("--origin", help="Optional event origin in UTC, for a marker line")
    parser.add_argument("--output", type=Path, help="PNG path; defaults to results/figures/")
    parser.add_argument(
        "--raw-scale", action="store_true",
        help="Plot counts directly instead of normalizing each trace for display",
    )
    args = parser.parse_args()

    stream = read(str(args.input))
    fig, axes = plt.subplots(len(STATIONS), 1, figsize=(15, 9), sharex=True)
    found = False

    for ax, spec in zip(axes, STATIONS):
        station_stream = stream.select(
            network=spec["network"], station=spec["station"]
        )
        if spec["location"] != "*":
            station_stream = station_stream.select(location=spec["location"])
        if not station_stream:
            ax.text(0.5, 0.5, "No matching trace", ha="center", va="center",
                    transform=ax.transAxes)
        for trace in station_stream:
            found = True
            values = trace.data.astype(float)
            if not args.raw_scale:
                scale = np.nanstd(values)
                if scale:
                    values = (values - np.nanmean(values)) / scale
            ax.plot(trace.times(type="matplotlib"), values, linewidth=0.55,
                    label=trace.id)
        ax.set_ylabel(spec["station"])
        ax.grid(True, alpha=0.25)
        if station_stream:
            ax.legend(loc="upper right", fontsize=8, ncol=3)

    if not found:
        parser.error("The input file contains no traces for II.PALK, GE.MALK, or GE.HALK.")

    if args.origin:
        origin = UTCDateTime(args.origin).datetime.replace(tzinfo=timezone.utc)
        origin_x = mdates.date2num(origin)
        for ax in axes:
            ax.axvline(origin_x, color="black", linestyle="--", linewidth=0.8)

    axes[-1].xaxis_date()
    axes[-1].xaxis.set_major_formatter(
        mdates.DateFormatter("%Y-%m-%d %H:%M", tz=timezone.utc)
    )
    axes[-1].set_xlabel("UTC")
    fig.suptitle(f"Aligned station traces: {args.input.stem}")
    fig.tight_layout()

    output = args.output or Path("results/figures") / f"{args.input.stem}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    print(f"Saved plot: {output}")


if __name__ == "__main__":
    main()
