"""Combine every available station/event trace into one phase-stack figure.

Uses the response-corrected, 20–100 s filtered waveforms and the same phase
windows/normalization as plot_all_station_phase_stacks.py. Station traces are
first stacked within each earthquake; the grand stack then gives each event
equal weight so events with more available stations do not dominate.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from obspy import read

from plot_all_station_phase_stacks import (
    COMPONENTS,
    DEFAULT_PROCESSED_DIR,
    DEFAULT_SOURCE_MANIFEST,
    PHASE_COLORS,
    PHASES,
    PRE_S,
    POST_S,
    SAMPLE_RATE_HZ,
    prepare_station_traces,
    read_rows,
)


DEFAULT_OUTPUT_DIR = Path("results/grand_phase_stacks_20_100s")


def stack_all_events(rows: list[dict[str, str]], processed_dir: Path):
    by_event: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_event.setdefault(row["event_id"], []).append(row)

    times = np.arange(-PRE_S, POST_S + 0.5 / SAMPLE_RATE_HZ, 1.0 / SAMPLE_RATE_HZ)
    event_stacks: dict[str, dict[str, list[tuple[str, np.ndarray]]]] = {
        phase: {component: [] for component in COMPONENTS} for phase in PHASES
    }
    station_traces: dict[str, dict[str, list[tuple[str, str, np.ndarray]]]] = {
        phase: {component: [] for component in COMPONENTS} for phase in PHASES
    }
    manifest_rows = []

    for event_id, event_rows in sorted(by_event.items()):
        processed_path = processed_dir / f"{event_id}_2h_vel_ZNE_20_100s.mseed"
        if not processed_path.is_file():
            raise FileNotFoundError(f"Processed response-corrected file missing: {processed_path}")
        stream = read(str(processed_path))

        for phase in PHASES:
            traces = prepare_station_traces(stream, event_rows, phase)
            for component in COMPONENTS:
                component_traces = traces[component]
                if not component_traces:
                    continue
                matrix = np.vstack([data for _station, data in component_traces])
                event_stack = np.mean(matrix, axis=0)
                event_stacks[phase][component].append((event_id, event_stack))
                for station_id, data in component_traces:
                    station_traces[phase][component].append((event_id, station_id, data))
                manifest_rows.append({
                    "event_id": event_id,
                    "phase": phase,
                    "component": component,
                    "n_stations": len(component_traces),
                    "stations": ";".join(station_id for station_id, _data in component_traces),
                    "event_weight_in_grand_stack": "equal",
                    "normalization": "per-station/component robust pre-arrival MAD scale from -30 to -5 s",
                    "processed_file": str(processed_path),
                })
        print(f"Added {event_id}: {len(event_rows)} available station record(s)")
    return times, event_stacks, station_traces, manifest_rows


def make_figure(times, event_stacks, station_traces, output: Path):
    phase_event_counts = {
        phase: max((len(event_stacks[phase][component]) for component in COMPONENTS), default=0)
        for phase in PHASES
    }
    station_ids = sorted({
        station_id
        for phase in PHASES
        for component in COMPONENTS
        for _event_id, station_id, _data in station_traces[phase][component]
    })
    station_cmap = plt.get_cmap("tab10")
    station_colors = {
        station_id: station_cmap(index % 10)
        for index, station_id in enumerate(station_ids)
    }
    fig, axes = plt.subplots(3, 3, figsize=(19, 12), sharex=True, layout="constrained")

    for row_index, phase in enumerate(PHASES):
        for column_index, component in enumerate(COMPONENTS):
            ax = axes[row_index, column_index]
            ax.axvspan(-2.0, 8.0, color="#d7eaf8", alpha=0.35, zorder=0)
            ax.axvline(0.0, color=PHASE_COLORS[phase], linestyle="--", linewidth=1.0, zorder=1)

            for _event_id, station_id, data in station_traces[phase][component]:
                ax.plot(times, data, color=station_colors[station_id], linewidth=0.35,
                        alpha=0.11, zorder=2)

            event_curves = event_stacks[phase][component]
            if event_curves:
                values = np.vstack([data for _event, data in event_curves])
                grand_stack = np.mean(values, axis=0)
                n_events = values.shape[0]
                if n_events > 1:
                    sem = np.std(values, axis=0, ddof=1) / np.sqrt(n_events)
                    ax.fill_between(times, grand_stack - sem, grand_stack + sem,
                                    color="#111827", alpha=0.13, linewidth=0, zorder=3)
                ax.plot(times, grand_stack, color="#111827", linewidth=2.0,
                        label="Equal-event grand stack", zorder=5)
                n_pairs = len(station_traces[phase][component])
                ax.text(0.98, 0.96, f"{n_events} events · {n_pairs} station/event traces",
                        transform=ax.transAxes, ha="right", va="top", fontsize=7.7,
                        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.76, "pad": 1.0})
            else:
                ax.text(0.5, 0.5, f"No complete {phase} windows",
                        transform=ax.transAxes, ha="center", va="center",
                        fontsize=9, color="#667085")

            ax.set_xlim(-PRE_S, POST_S)
            ax.set_xticks(np.arange(-30, 61, 10))
            ax.grid(True, axis="x", color="#b7c0ca", linewidth=0.45, alpha=0.65)
            ax.tick_params(axis="both", labelsize=8)
            if row_index == 0:
                ax.set_title({"Z": "Vertical (Z)", "R": "Radial (R)", "T": "Transverse (T)"}[component],
                             fontsize=11, pad=8)
            if column_index == 0:
                ax.set_ylabel(f"{phase}\nvelocity / pre-event noise", fontsize=9)
            if row_index == 2:
                ax.set_xlabel("Seconds from predicted arrival", fontsize=9)
            if row_index == 0 and column_index == 0:
                handles = [
                    Line2D([0], [0], color=station_colors[station_id], linewidth=1.2,
                           label=station_id.split(".")[-1])
                    for station_id in station_ids
                ]
                handles.append(Line2D([0], [0], color="#111827", linewidth=1.9,
                                      label="Equal-event grand stack"))
                ax.legend(handles=handles, loc="upper left", ncol=2, fontsize=7, framealpha=0.9)

    output.parent.mkdir(parents=True, exist_ok=True)
    n_events = len({event_id for phase in PHASES for component in COMPONENTS
                    for event_id, _curve in event_stacks[phase][component]})
    fig.suptitle(
        f"All screened earthquakes | {n_events} events | SKS / ScS / SKKS × Z / R / T\n"
        "All-station waveforms, response-corrected 20–100 s; equal event weighting",
        fontsize=13,
    )
    fig.text(
        0.5, -0.018,
        "Faint colored curves: individual station/event traces. Thick black: mean of event-level station stacks; "
        "each event has equal weight. Gray band: ±1 standard error across event stacks. "
        "Traces are normalized by pre-arrival noise (−30 to −5 s), so values are relative, not m/s. "
        "Blue shading: −2 to +8 s.",
        ha="center", va="top", fontsize=8, color="#4b5563",
    )
    fig.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    times, event_stacks, station_traces, manifest_rows = stack_all_events(
        read_rows(args.source_manifest), args.processed_dir
    )
    output = args.output_dir / "all_events_all_stations_SKSS_20_100s.png"
    make_figure(times, event_stacks, station_traces, output)
    manifest_path = args.output_dir / "grand_stack_manifest.csv"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    (args.output_dir / "README.txt").write_text(
        "Single combined figure containing all 23 screened event cuts and all available stations.\n"
        "Rows: SKS, ScS, SKKS. Columns: vertical Z, radial R, transverse T.\n"
        "Faint lines are every usable station/event waveform. Within each event, station traces are\n"
        "averaged; the grand black stack averages event stacks so each event has equal weight.\n"
        "Event-to-event source radiation and polarity differences can cancel in the grand mean; this\n"
        "summary does not replace inspection of the per-event stacks.\n"
        "Each station/component trace is divided by its robust pre-arrival noise scale (-30 to -5 s).\n"
        "The filter was applied after instrument response removal: 0.01-0.05 Hz (20-100 s).\n"
        "Values shown are normalized relative amplitudes, not physical m/s. See grand_stack_manifest.csv\n"
        "for station counts and included stations by event, phase, and component.\n",
        encoding="utf-8",
    )
    print(f"Combined figure: {output}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
