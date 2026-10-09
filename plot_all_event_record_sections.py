"""Create response-corrected 20–100 s record sections for every screened event.

The project has one two-hour MiniSEED cut per screened XKS candidate event.
This script response-corrects each available station to velocity, rotates its
components to ZNE, applies a 0.01–0.05 Hz zero-phase bandpass, and writes a full
event overview plus SKS, SKKS, and (where predicted) ScS-centered R/T sections.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from obspy import Stream, UTCDateTime, read, read_inventory
from obspy.geodetics import gps2dist_azimuth, locations2degrees
from obspy.taup import TauPyModel

from config import STATION_COORDINATES, STATIONS


DEFAULT_INPUT_DIR = Path("data/cuts_2h")
DEFAULT_STATIONXML = Path("data/metadata/sri_lanka_response.xml")
DEFAULT_CANDIDATES = Path("results/event_station_candidates_1A_active.csv")
DEFAULT_PROCESSED_DIR = Path("data/processed_20_100s")
DEFAULT_OUTPUT_DIR = Path("results/record_sections_20_100s_all_events")

FREQ_MIN_HZ = 0.01
FREQ_MAX_HZ = 0.05
FULL_END_SECONDS = 5400.0
PHASE_PRE_SECONDS = 30.0
PHASE_POST_SECONDS = 60.0
EARTH_RADIUS_KM = 6371.0

# Family names keep direct and diffracted arrivals together where TauP has no
# direct branch at the event's distance.
PHASE_GROUPS = {
    "P/Pdiff": ("P", "Pdiff"),
    "PP": ("PP",),
    "SKS": ("SKS",),
    "ScS": ("ScS",),
    "SKKS": ("SKKS",),
    "S/Sdiff": ("S", "Sdiff"),
    "SS": ("SS",),
}
PHASE_COLORS = {
    "P/Pdiff": "#1d4ed8",
    "PP": "#2563eb",
    "SKS": "#dc2626",
    "ScS": "#7e22ce",
    "SKKS": "#a21caf",
    "S/Sdiff": "#ea580c",
    "SS": "#9333ea",
    "Surface waves": "#c58b46",
}
ZOOM_PHASES = ("SKS", "SKKS", "ScS")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def candidate_events(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    events: dict[str, dict[str, str]] = {}
    for row in rows:
        events.setdefault(row["event_id"], row)
    return list(events.values())


def find_cut(input_dir: Path, event_id: str) -> Path:
    candidates = sorted(input_dir.glob(f"*_{event_id}_2h.mseed"))
    if not candidates:
        raise FileNotFoundError(f"No two-hour cut found for event {event_id} in {input_dir}")
    return candidates[0]


def station_spec(network: str, station: str) -> dict[str, str] | None:
    for spec in STATIONS:
        if spec["network"] == network and spec["station"] == station:
            return spec
    return None


def process_event(
    input_path: Path,
    output_path: Path,
    inventory,
    freq_min: float = FREQ_MIN_HZ,
    freq_max: float = FREQ_MAX_HZ,
) -> tuple[Stream, dict[str, str]]:
    """Remove response, rotate sensor axes to ZNE, and filter each station."""
    raw = read(str(input_path))
    corrected = Stream()
    status: dict[str, str] = {}
    present = sorted({(tr.stats.network, tr.stats.station) for tr in raw})

    for network, station in present:
        spec = station_spec(network, station)
        if spec is None:
            status[f"{network}.{station}"] = "not in configured station list"
            continue
        station_stream = raw.select(network=network, station=station).copy()
        location = spec["location"]
        if location != "*":
            station_stream = station_stream.select(location=location)
        if not station_stream:
            status[f"{network}.{station}"] = "no configured location data"
            continue

        try:
            for trace in station_stream:
                trace.detrend("demean")
                trace.detrend("linear")
                trace.taper(max_percentage=0.02, type="cosine")
                trace.remove_response(
                    inventory=inventory,
                    output="VEL",
                    pre_filt=(0.003, 0.006, 0.065, 0.09),
                    water_level=60.0,
                )

            # This step is essential for PALK's BH1/BH2 channels: it uses the
            # StationXML orientation to transform the sensor axes into N/E.
            station_stream.rotate(method="->ZNE", inventory=inventory)
            for trace in station_stream:
                nyquist = trace.stats.sampling_rate / 2.0
                if freq_max >= nyquist:
                    raise ValueError(
                        f"{trace.id} Nyquist {nyquist:g} Hz is below {freq_max:g} Hz"
                    )
                trace.filter(
                    "bandpass",
                    freqmin=freq_min,
                    freqmax=freq_max,
                    corners=4,
                    zerophase=True,
                )
                trace.data = np.asarray(trace.data, dtype=np.float32)
            corrected += station_stream
            status[f"{network}.{station}"] = "response-corrected velocity; rotated ZNE; filtered"
        except Exception as error:
            status[f"{network}.{station}"] = f"processing failed: {type(error).__name__}: {error}"

    corrected.sort()
    if not corrected:
        raise RuntimeError(f"No station traces could be processed from {input_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    corrected.write(str(output_path), format="MSEED")
    return corrected, status


def get_station_geometry(event: dict[str, str], network: str, station: str):
    lat, lon = STATION_COORDINATES[(network, station)]
    ev_lat, ev_lon = float(event["event_latitude"]), float(event["event_longitude"])
    distance = float(locations2degrees(ev_lat, ev_lon, lat, lon))
    _meters, azimuth, back_azimuth = gps2dist_azimuth(ev_lat, ev_lon, lat, lon)
    return lat, lon, distance, float(azimuth), float(back_azimuth)


def predict_phases(event: dict[str, str], stations: list[tuple[str, str]], model: TauPyModel):
    origin = UTCDateTime(event["event_time_utc"])
    phase_rows: dict[str, dict[str, dict[str, float | str]]] = defaultdict(dict)
    metadata: dict[str, dict[str, float]] = {}

    for network, station in stations:
        _lat, _lon, distance, azimuth, back_azimuth = get_station_geometry(event, network, station)
        metadata[station] = {
            "distance_deg": distance,
            "azimuth_deg": azimuth,
            "back_azimuth_deg": back_azimuth,
        }
        arrivals = model.get_travel_times(
            source_depth_in_km=float(event["depth_km"]),
            distance_in_degree=distance,
            phase_list=sorted({phase for group in PHASE_GROUPS.values() for phase in group}),
        )
        for family, family_phases in PHASE_GROUPS.items():
            options = [arrival for arrival in arrivals if arrival.name in family_phases]
            if not options:
                continue
            first = min(options, key=lambda item: item.time)
            phase_rows[family][station] = {
                "phase": str(first.name),
                "travel_time_s": float(first.time),
                "arrival_utc": str(origin + float(first.time)),
            }
    return phase_rows, metadata


def station_component(stream: Stream, network: str, station: str, component: str):
    matches = [
        tr for tr in stream
        if tr.stats.network == network
        and tr.stats.station == station
        and tr.stats.channel.endswith(component)
    ]
    if not matches:
        return None
    return max(matches, key=lambda tr: tr.stats.npts)


def relative_cut(trace, reference: UTCDateTime, start_s: float, end_s: float):
    sample_rate = float(trace.stats.sampling_rate)
    offset = float(trace.stats.starttime - reference)
    left = max(0, int(math.ceil((start_s - offset) * sample_rate)))
    right = min(trace.stats.npts, int(math.floor((end_s - offset) * sample_rate)) + 1)
    if right <= left:
        return np.array([]), np.array([])
    times = offset + np.arange(left, right, dtype=float) / sample_rate
    data = np.asarray(trace.data[left:right], dtype=float)
    finite = np.isfinite(data)
    return times[finite], data[finite]


def event_title(event: dict[str, str]) -> str:
    magnitude = float(event.get("magnitude", "nan"))
    origin = UTCDateTime(event["event_time_utc"])
    return f"M{magnitude:.1f} | {origin.strftime('%Y-%m-%d %H:%M:%S UTC')} | {event['event_id']}"


def first_prediction_time(phase_rows, family: str, stations: list[str]):
    return [
        float(phase_rows[family][station]["travel_time_s"])
        for station in stations
        if station in phase_rows.get(family, {})
    ]


def plot_full_record_section(
    event: dict[str, str],
    stream: Stream,
    phase_rows,
    station_meta: dict[str, dict[str, float]],
    stations: list[tuple[str, str]],
    output: Path,
    x_end_s: float = FULL_END_SECONDS,
):
    origin = UTCDateTime(event["event_time_utc"])
    stations = sorted(stations, key=lambda item: station_meta[item[1]]["distance_deg"])
    codes = [station for _network, station in stations]
    distances = np.array([station_meta[station]["distance_deg"] for station in codes])
    distance_span = float(np.ptp(distances)) if len(distances) > 1 else 0.5
    gap = float(np.median(np.diff(np.unique(distances)))) if len(np.unique(distances)) > 1 else 0.5
    wiggle_scale = max(min(gap * 0.27, 0.28), distance_span * 0.025, 0.06)

    fig, ax = plt.subplots(figsize=(18, 11), layout="constrained")
    surface_start, surface_end = [], []
    for distance in distances:
        arc_km = EARTH_RADIUS_KM * math.radians(float(distance))
        surface_start.append(arc_km / 4.2)
        surface_end.append(arc_km / 3.2)
    if surface_start:
        shade_start = max(0.0, min(surface_start))
        shade_end = min(x_end_s, max(surface_end))
        if shade_end > shade_start:
            ax.axvspan(
                shade_start, shade_end, color=PHASE_COLORS["Surface waves"],
                alpha=0.16, zorder=0,
            )
            ax.text(
                0.5 * (shade_start + shade_end), 0.97,
                "Surface waves (approx. 3.2–4.2 km/s)",
                transform=ax.get_xaxis_transform(), ha="center", va="top",
                color="#805b2d", fontsize=9,
            )

    for network, station in stations:
        distance = station_meta[station]["distance_deg"]
        trace = station_component(stream, network, station, "Z")
        if trace is None:
            continue
        times, data = relative_cut(trace, origin, 0.0, x_end_s)
        if data.size < 2:
            continue
        scale = float(np.nanpercentile(np.abs(data), 99.5))
        if not np.isfinite(scale) or scale <= 0:
            continue
        ax.plot(
            times, distance + wiggle_scale * data / scale,
            color="#252525", linewidth=0.55, zorder=2,
        )

    for family, color in PHASE_COLORS.items():
        if family == "Surface waves":
            continue
        picks = phase_rows.get(family, {})
        if not picks:
            continue
        pick_times = []
        for station in codes:
            if station not in picks:
                continue
            time = float(picks[station]["travel_time_s"])
            if 0 <= time <= x_end_s:
                ax.scatter(
                    [time], [station_meta[station]["distance_deg"]],
                    color=color, marker="o", s=24, edgecolor="white", linewidth=0.45,
                    zorder=5,
                )
                pick_times.append(time)
        if pick_times:
            center = float(np.median(pick_times))
            ax.axvline(center, color=color, linewidth=1.0, linestyle=":", alpha=0.9, zorder=1)

    ax.set_xlim(0, x_end_s)
    margin = max(gap * 0.45, 0.12)
    ax.set_ylim(float(np.min(distances) - margin), float(np.max(distances) + margin))
    ax.set_xlabel("Time since origin (s)", fontsize=11)
    ax.set_ylabel("Epicentral distance (degrees)", fontsize=11)
    ax.set_xticks(np.arange(0, x_end_s + 1, 300))
    ax.grid(True, axis="x", color="#aeb5bf", linewidth=0.45, alpha=0.65)
    ax.set_yticks(distances, [f"{distance:.2f}°" for distance in distances])
    ax.tick_params(axis="y", labelsize=8, labelleft=True, labelright=False)
    right = ax.twinx()
    right.set_ylim(ax.get_ylim())
    right.set_yticks(distances, codes)
    right.tick_params(axis="y", length=0, labelsize=9)
    right.grid(False)
    right.set_ylabel("Station", fontsize=10)
    for spine in right.spines.values():
        spine.set_visible(False)

    legend_handles = [
        Line2D([0], [0], color=PHASE_COLORS[key], linestyle=":", linewidth=1.4, label=key)
        for key in PHASE_GROUPS
    ]
    legend_handles.append(
        Line2D([0], [0], color=PHASE_COLORS["Surface waves"], linewidth=8, alpha=0.3,
               label="Surface-wave guide")
    )
    ax.legend(handles=legend_handles, loc="upper right", ncol=4, framealpha=0.93, fontsize=8)
    ax.set_title(event_title(event), fontsize=13, pad=19)
    fig.text(
        0.5, -0.035,
        "Response-corrected velocity, 20–100 s bandpass | normalized vertical wiggles. "
        "Colored lines and dots: iasp91 P/Pdiff, PP, SKS, ScS, SKKS, S/Sdiff, SS. "
        "P/Pdiff and S/Sdiff show the earliest branch; surface-wave shading is approximate.",
        ha="center", va="top", fontsize=8, color="#4b5563",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(fig)


def rotated_station_stream(stream: Stream, network: str, station: str, back_azimuth: float):
    selected = Stream([
        trace.copy() for trace in stream
        if trace.stats.network == network and trace.stats.station == station
    ])
    if not selected:
        return selected
    selected.rotate(method="NE->RT", back_azimuth=back_azimuth)
    return selected


def plot_phase_zoom(
    event: dict[str, str],
    stream: Stream,
    phase_rows,
    station_meta: dict[str, dict[str, float]],
    stations: list[tuple[str, str]],
    target_phase: str,
    output: Path,
    pre_s: float = PHASE_PRE_SECONDS,
    post_s: float = PHASE_POST_SECONDS,
):
    picks = phase_rows.get(target_phase, {})
    if not picks:
        return False
    origin = UTCDateTime(event["event_time_utc"])
    stations = sorted(stations, key=lambda item: station_meta[item[1]]["distance_deg"])
    codes = [station for _network, station in stations]
    ranks = {station: len(stations) - i - 1 for i, (_network, station) in enumerate(stations)}
    components = ("R", "T")
    colors = {"R": "#225ea8", "T": "#b23a48"}
    fig, axes = plt.subplots(2, 1, figsize=(15, max(6.5, 1.15 * len(stations) + 3)),
                             sharex=True, sharey=True, layout="constrained")

    for ax, component in zip(axes, components):
        ax.axvspan(-2.0, 8.0, color="#d7eaf8", alpha=0.45, zorder=0)
        for network, station in stations:
            rank = ranks[station]
            if station not in picks:
                continue
            target_pick = origin + float(picks[station]["travel_time_s"])
            baz = station_meta[station]["back_azimuth_deg"]
            rotated = rotated_station_stream(stream, network, station, baz)
            trace = station_component(rotated, network, station, component)
            if trace is None:
                continue
            times, data = relative_cut(trace, target_pick, -pre_s, post_s)
            if data.size < 2:
                continue
            scale = float(np.nanpercentile(np.abs(data), 99.5))
            if not np.isfinite(scale) or scale <= 0:
                continue
            ax.plot(times, rank + 0.34 * data / scale, color=colors[component],
                    linewidth=0.65, zorder=2)

        ax.axvline(0, color=PHASE_COLORS[target_phase], linewidth=1.2, linestyle="--", zorder=3)
        # Predicted SKS / ScS / SKKS arrivals are shown relative to this page's
        # target pick. This reveals close or overlapping phases on the same cut.
        for family in ZOOM_PHASES:
            family_picks = phase_rows.get(family, {})
            rel_offsets = []
            for network, station in stations:
                if station not in picks or station not in family_picks:
                    continue
                relative = float(family_picks[station]["travel_time_s"]) - float(picks[station]["travel_time_s"])
                if -pre_s <= relative <= post_s:
                    ax.scatter([relative], [ranks[station]], color=PHASE_COLORS[family], s=27,
                               edgecolor="white", linewidth=0.5, zorder=5)
                    rel_offsets.append(relative)
            if rel_offsets and family != target_phase:
                center = float(np.median(rel_offsets))
                ax.axvline(center, color=PHASE_COLORS[family], linewidth=0.85,
                           linestyle=":", alpha=0.9, zorder=1)

        ax.set_ylabel(f"{component} (normalized)", fontsize=10)
        ax.grid(True, axis="x", color="#b5bdc8", linewidth=0.5, alpha=0.65)
        ax.set_ylim(-0.65, len(stations) - 0.35)
        ax.set_yticks([ranks[station] for station in codes],
                      [f"{station}  {station_meta[station]['distance_deg']:.1f}°" for station in codes],
                      fontsize=8)
        ax.invert_yaxis()

    axes[-1].set_xlim(-pre_s, post_s)
    axes[-1].set_xticks(np.arange(-pre_s, post_s + 0.1, 10))
    axes[-1].set_xlabel(f"Seconds from predicted {target_phase} arrival", fontsize=10, labelpad=9)
    fig.suptitle(
        event_title(event) + f"\n{target_phase}-centered horizontal particle-motion traces | response-corrected velocity, 20–100 s",
        fontsize=12,
    )
    fig.text(
        0.5, -0.035,
        f"R (blue), T (red); dashed line = {target_phase} center; dots mark predicted SKS (red), ScS (purple), SKKS (magenta). "
        "Blue shading is the −2 to +8 s polarization window. Picks are iasp91 predictions; traces are normalized per station.",
        ha="center", va="top", fontsize=7.7, color="#4b5563",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--stationxml", type=Path, default=DEFAULT_STATIONXML)
    parser.add_argument("--candidate-csv", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--event-id", help="Process one candidate event (default: every candidate)")
    parser.add_argument("--reuse-processed", action="store_true",
                        help="Reuse response-corrected 20–100 s files already in --processed-dir")
    parser.add_argument("--freq-min", type=float, default=FREQ_MIN_HZ)
    parser.add_argument("--freq-max", type=float, default=FREQ_MAX_HZ)
    parser.add_argument("--full-end", type=float, default=FULL_END_SECONDS)
    parser.add_argument("--pre", type=float, default=PHASE_PRE_SECONDS)
    parser.add_argument("--post", type=float, default=PHASE_POST_SECONDS)
    args = parser.parse_args()

    if not (0 < args.freq_min < args.freq_max):
        parser.error("Require 0 < --freq-min < --freq-max in Hz")
    if args.full_end <= 0 or args.pre <= 0 or args.post <= 0:
        parser.error("Record-section and phase-zoom durations must be positive")
    if not args.stationxml.is_file():
        raise FileNotFoundError(f"StationXML not found: {args.stationxml}")

    events = candidate_events(read_rows(args.candidate_csv))
    if args.event_id:
        events = [event for event in events if event["event_id"] == args.event_id]
        if not events:
            raise KeyError(f"No candidate event {args.event_id} in {args.candidate_csv}")

    inventory = read_inventory(str(args.stationxml))
    model = TauPyModel("iasp91")
    args.processed_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    event_count = 0
    overview_count = 0
    zoom_count = 0
    for event in events:
        event_id = event["event_id"]
        input_path = find_cut(args.input_dir, event_id)
        processed_path = args.processed_dir / f"{event_id}_2h_vel_ZNE_20_100s.mseed"
        if args.reuse_processed and processed_path.exists():
            processed = read(str(processed_path))
            processing_status = "reused cached response-corrected 20–100 s stream"
        else:
            processed, status = process_event(
                input_path, processed_path, inventory,
                freq_min=args.freq_min, freq_max=args.freq_max,
            )
            processing_status = "; ".join(f"{key}: {value}" for key, value in status.items())

        stations = sorted({
            (trace.stats.network, trace.stats.station) for trace in processed
            if trace.stats.channel.endswith(("Z", "N", "E"))
        })
        if not stations:
            raise RuntimeError(f"No ZNE stations in processed event {event_id}")
        phase_rows, station_meta = predict_phases(event, stations, model)
        event_dir = args.output_dir / event_id
        event_dir.mkdir(parents=True, exist_ok=True)

        overview_path = event_dir / f"{event_id}_full_event_record_section_20_100s.png"
        plot_full_record_section(
            event, processed, phase_rows, station_meta, stations,
            overview_path, x_end_s=args.full_end,
        )
        overview_count += 1

        zoom_paths = []
        for phase in ZOOM_PHASES:
            if not phase_rows.get(phase):
                continue
            zoom_path = event_dir / f"{event_id}_{phase}_radial_transverse_20_100s.png"
            if plot_phase_zoom(
                event, processed, phase_rows, station_meta, stations,
                phase, zoom_path, pre_s=args.pre, post_s=args.post,
            ):
                zoom_paths.append(zoom_path)
                zoom_count += 1

        for network, station in stations:
            meta = station_meta[station]
            row = {
                "event_id": event_id,
                "origin_utc": event["event_time_utc"],
                "magnitude": event["magnitude"],
                "station": station,
                "network": network,
                "distance_deg": f"{meta['distance_deg']:.6f}",
                "back_azimuth_deg": f"{meta['back_azimuth_deg']:.4f}",
                "processed_velocity_file": str(processed_path),
                "processing": processing_status,
                "overview_png": str(overview_path),
                "zoom_pngs": ";".join(str(path) for path in zoom_paths),
            }
            for family in PHASE_GROUPS:
                phase = phase_rows.get(family, {}).get(station)
                row[f"{family}_branch"] = phase["phase"] if phase else ""
                row[f"{family}_travel_time_s"] = (
                    f"{float(phase['travel_time_s']):.3f}" if phase else ""
                )
                row[f"{family}_arrival_utc"] = phase["arrival_utc"] if phase else ""
            manifest_rows.append(row)

        event_count += 1
        print(
            f"{event_id}: {len(stations)} station(s), overview + {len(zoom_paths)} phase zoom(s); "
            f"processed file {processed_path}"
        )

    manifest_path = args.output_dir / "all_events_station_phase_manifest.csv"
    if manifest_rows:
        with manifest_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
            writer.writeheader()
            writer.writerows(manifest_rows)
    readme_path = args.output_dir / "README.txt"
    readme_path.write_text(
        "Record sections for every screened event with an available two-hour cut.\n"
        f"Events plotted: {event_count}\n"
        f"Response correction: ObsPy remove_response(output='VEL') using the supplied StationXML.\n"
        f"Bandpass: {args.freq_min:g}–{args.freq_max:g} Hz (20–100 s period), 4 corners, zero phase.\n"
        "Components are rotated to ZNE using StationXML before filtering. Horizontal focus plots use R/T.\n"
        "The overview uses normalized Z wiggles and TauP iasp91 predicted phase times.\n"
        "Each phase zoom is centered on that station's predicted arrival. ScS zooms are written only\n"
        "where TauP predicts ScS at one or more available stations.\n"
        "P/Pdiff and S/Sdiff columns show the earliest predicted direct/diffracted branch.\n"
        "The shaded surface-wave interval is a velocity-based guide, not a measured arrival pick.\n"
        "Waveforms in figures are normalized independently for visibility; use the processed MiniSEED\n"
        "in data/processed_20_100s for amplitude-based analysis.\n"
        "Earlier figures and reports remain archived in results/previous_outputs_20261005.\n",
        encoding="utf-8",
    )
    print(
        f"Finished {event_count} event(s): {overview_count} overview figure(s), "
        f"{zoom_count} phase zoom(s). Manifest: {manifest_path}"
    )


if __name__ == "__main__":
    main()
