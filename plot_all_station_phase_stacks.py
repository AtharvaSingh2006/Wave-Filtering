"""Align and stack all available stations around predicted XKS phases.

Inputs are response-corrected ZNE velocity waveforms already filtered to
20–100 s. The output has SKS, ScS, and SKKS rows and Z, radial, and transverse
columns. Each station/component is divided by its pre-arrival noise scale so
each station contributes equally to the linear stack.
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
from obspy import Stream, UTCDateTime, read


DEFAULT_PROCESSED_DIR = Path("data/processed_20_100s")
DEFAULT_SOURCE_MANIFEST = Path(
    "results/record_sections_20_100s_all_events/all_events_station_phase_manifest.csv"
)
DEFAULT_OUTPUT_DIR = Path("results/all_station_phase_stacks_20_100s")
PHASES = ("SKS", "ScS", "SKKS")
COMPONENTS = ("Z", "R", "T")
PRE_S = 30.0
POST_S = 60.0
SAMPLE_RATE_HZ = 20.0
PHASE_COLORS = {"SKS": "#c53030", "ScS": "#7e22ce", "SKKS": "#a21caf"}


def read_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def component_trace(stream: Stream, network: str, station: str, component: str):
    candidates = [
        trace for trace in stream
        if trace.stats.network == network
        and trace.stats.station == station
        and trace.stats.channel.endswith(component)
    ]
    return max(candidates, key=lambda trace: trace.stats.npts) if candidates else None


def station_stream_for_components(stream: Stream, row: dict[str, str]):
    selected = Stream([
        trace.copy() for trace in stream
        if trace.stats.network == row["network"]
        and trace.stats.station == row["station"]
    ])
    if not selected:
        return selected
    baz = float(row["back_azimuth_deg"])
    selected.rotate(method="NE->RT", back_azimuth=baz)
    return selected


def aligned_window(trace, arrival: UTCDateTime):
    """Interpolate one complete 20 Hz window onto phase-relative sample times."""
    target = np.arange(-PRE_S, POST_S + 0.5 / SAMPLE_RATE_HZ, 1.0 / SAMPLE_RATE_HZ)
    relative_start = float(trace.stats.starttime - arrival)
    relative_trace_times = relative_start + np.arange(trace.stats.npts, dtype=float) / float(
        trace.stats.sampling_rate
    )
    if target[0] < relative_trace_times[0] or target[-1] > relative_trace_times[-1]:
        return target, None
    data = np.asarray(trace.data, dtype=float)
    if not np.all(np.isfinite(data)):
        return target, None
    return target, np.interp(target, relative_trace_times, data)


def noise_scale(times: np.ndarray, trace: np.ndarray) -> float:
    quiet = trace[(times >= -PRE_S) & (times <= -5.0)]
    if quiet.size < 20:
        return float("nan")
    center = float(np.median(quiet))
    sigma = 1.4826 * float(np.median(np.abs(quiet - center)))
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = float(np.sqrt(np.mean((quiet - center) ** 2)))
    return sigma


def prepare_station_traces(stream: Stream, station_rows: list[dict[str, str]], phase: str):
    traces_by_component = {component: [] for component in COMPONENTS}
    use_rows = [row for row in station_rows if row.get(f"{phase}_travel_time_s")]
    for row in use_rows:
        station_stream = station_stream_for_components(stream, row)
        if not station_stream:
            continue
        arrival = UTCDateTime(row[f"{phase}_arrival_utc"])
        station_id = f"{row['network']}.{row['station']}"
        for component in COMPONENTS:
            trace = component_trace(station_stream, row["network"], row["station"], component)
            if trace is None:
                continue
            times, data = aligned_window(trace, arrival)
            if data is None:
                continue
            scale = noise_scale(times, data)
            if not np.isfinite(scale) or scale <= 0:
                continue
            traces_by_component[component].append((station_id, data / scale))
    return traces_by_component


def plot_event(event_id: str, rows: list[dict[str, str]], processed_dir: Path, output_path: Path):
    processed_path = processed_dir / f"{event_id}_2h_vel_ZNE_20_100s.mseed"
    if not processed_path.is_file():
        raise FileNotFoundError(f"Response-corrected processed file missing: {processed_path}")
    stream = read(str(processed_path))
    first_row = rows[0]
    origin = UTCDateTime(first_row["origin_utc"])
    magnitude = float(first_row["magnitude"])
    station_names = sorted({f"{row['network']}.{row['station']}" for row in rows})
    nstations = len(station_names)
    target_times = np.arange(-PRE_S, POST_S + 0.5 / SAMPLE_RATE_HZ, 1.0 / SAMPLE_RATE_HZ)

    fig, axes = plt.subplots(
        nrows=3,
        ncols=3,
        figsize=(18, 10.5),
        sharex=True,
        layout="constrained",
    )
    manifest_rows = []
    station_cmap = plt.get_cmap("tab10")
    station_color_by_id = {
        station_id: station_cmap(index % 10)
        for index, station_id in enumerate(station_names)
    }

    for row_index, phase in enumerate(PHASES):
        stacked = prepare_station_traces(stream, rows, phase)
        for column_index, component in enumerate(COMPONENTS):
            ax = axes[row_index, column_index]
            ax.axvspan(-2.0, 8.0, color="#d7eaf8", alpha=0.35, zorder=0)
            ax.axvline(0.0, color=PHASE_COLORS[phase], linestyle="--", linewidth=1.0, zorder=1)
            component_traces = stacked[component]
            nstack = len(component_traces)

            if nstack == 0:
                ax.text(0.5, 0.5, f"No complete {phase} windows", transform=ax.transAxes,
                        ha="center", va="center", fontsize=9, color="#667085")
            else:
                matrix = np.vstack([data for _station, data in component_traces])
                mean_stack = np.mean(matrix, axis=0)
                if nstack > 1:
                    sem = np.std(matrix, axis=0, ddof=1) / np.sqrt(nstack)
                    ax.fill_between(target_times, mean_stack - sem, mean_stack + sem,
                                    color="#111827", alpha=0.12, linewidth=0, zorder=2)
                for station_id, data in component_traces:
                    ax.plot(target_times, data, color=station_color_by_id[station_id],
                            linewidth=0.65, alpha=0.55, zorder=3)
                ax.plot(target_times, mean_stack, color="#111827", linewidth=1.9,
                        zorder=4)
                ax.text(0.98, 0.96, f"n={nstack}", transform=ax.transAxes,
                        ha="right", va="top", fontsize=8,
                        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 1.0})
                station_list = ";".join(station_id for station_id, _data in component_traces)
                manifest_rows.append({
                    "event_id": event_id,
                    "phase": phase,
                    "component": component,
                    "n_stations": nstack,
                    "stations": station_list,
                    "window_start_s": -PRE_S,
                    "window_end_s": POST_S,
                    "normalization": "per-station/component robust pre-arrival MAD scale from -30 to -5 s",
                    "processed_file": str(processed_path),
                    "figure": str(output_path),
                })

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
                legend_handles = [
                    Line2D([0], [0], color=station_color_by_id[station_id], linewidth=1.2,
                           label=station_id.split(".")[-1])
                    for station_id in station_names
                ]
                legend_handles.append(Line2D([0], [0], color="#111827", linewidth=1.9,
                                             label="Mean stack"))
                ax.legend(handles=legend_handles, loc="upper left", ncol=2,
                          fontsize=7, framealpha=0.88)

    fig.suptitle(
        f"M{magnitude:.1f} | {origin.strftime('%Y-%m-%d %H:%M:%S UTC')} | {event_id}\n"
        f"All-station stacks: SKS / ScS / SKKS × Z / R / T | response-corrected, 20–100 s",
        fontsize=13,
    )
    fig.text(
        0.5, -0.018,
        "Thin colored lines are individual available stations; thick black is the equal-weight mean stack. "
        "Traces are scaled by robust pre-arrival noise (−30 to −5 s), so vertical units are relative, not m/s. "
        "Blue shading: −2 to +8 s window.",
        ha="center", va="top", fontsize=8, color="#4b5563",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=185, bbox_inches="tight")
    plt.close(fig)
    return manifest_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--event-id", help="Stack one event (default: all events in the manifest)")
    args = parser.parse_args()

    rows = read_rows(args.source_manifest)
    by_event: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_event.setdefault(row["event_id"], []).append(row)
    event_ids = sorted(by_event)
    if args.event_id:
        if args.event_id not in by_event:
            raise KeyError(f"No event {args.event_id} in {args.source_manifest}")
        event_ids = [args.event_id]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_manifest_rows = []
    for event_id in event_ids:
        output_path = args.output_dir / event_id / f"{event_id}_all_station_phase_stacks_20_100s.png"
        all_manifest_rows.extend(plot_event(event_id, by_event[event_id], args.processed_dir, output_path))
        available = sorted({row["station"] for row in by_event[event_id]})
        print(f"{event_id}: stacked available stations {', '.join(available)} -> {output_path}")

    manifest_path = args.output_dir / "phase_stack_manifest.csv"
    if all_manifest_rows:
        with manifest_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(all_manifest_rows[0]))
            writer.writeheader()
            writer.writerows(all_manifest_rows)
    (args.output_dir / "README.txt").write_text(
        "One stack figure per screened event. Rows are SKS, ScS, and SKKS; columns are Z, radial, and transverse.\n"
        "All available stations with a predicted arrival and a complete window are included in the panel's mean stack.\n"
        "Thin lines show each station; black line is the equal-weight arithmetic mean. Traces are normalized by\n"
        "robust pre-arrival noise from -30 to -5 s, so plotted values are relative, not physical velocity amplitudes.\n"
        "The plotted waveforms remain response-corrected velocity filtered to 20–100 s before normalization.\n"
        "ScS has no predicted arrival for every event; empty panels indicate no complete predicted windows.\n"
        "See phase_stack_manifest.csv for station inclusion by event, phase, and component.\n",
        encoding="utf-8",
    )
    print(f"Finished {len(event_ids)} event stack figure(s): {args.output_dir}")


if __name__ == "__main__":
    main()
