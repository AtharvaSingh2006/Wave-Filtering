"""Run SWSPy-style single-event windows through Silver-Chan and CNN-SWS.

Inputs to the CNN follow the upstream four-trace contract (R, T, corrected R,
corrected T; 1,000 samples each at 20 Hz). The SC grid search is performed in
the pick-aware start/end windows used by this project's SWSPy event setup.
Inference uses the authors' supplied weights with a NumPy implementation of
their published Keras architecture, so TensorFlow is not required.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from obspy import Stream, Trace, UTCDateTime, read, read_inventory
from obspy.core.util.attribdict import AttribDict
from obspy.geodetics import gps2dist_azimuth, locations2degrees
from obspy.taup import TauPyModel

from audit_1a_geometry_and_polarization import (
    POLARIZATION_WINDOWS_S,
    PRIMARY_POLARIZATION_WINDOW,
    axial_difference_deg,
    axial_window_sigma_deg,
    numpy_horizontal_polarization,
    pca_metrics_by_window,
    polarization_window_mask,
    read_rows,
)
from process_1a_xks import cut_complete_window, response_correct_stream


PHASES = ("SKS", "SKKS", "PKS")
SAMPLE_RATE_HZ = 20.0
WINDOW_SECONDS = 50.0
FILTER_MIN_HZ = 0.04
FILTER_MAX_HZ = 0.5
ANALYSIS_START_S = np.linspace(-4.0, -1.0, 10)
ANALYSIS_END_S = np.linspace(20.0, 23.0, 10)
PHI_FAST_FROM_N_DEG = np.arange(0.0, 180.0, 2.0)
DELAY_GRID_S = np.arange(0.0, 4.0, 0.05)
ZERO_PAD_BEFORE_CONV = {1, 2, 4, 7, 8}


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def wrap_axial(angle: float) -> float:
    return float((angle + 90.0) % 180.0 - 90.0)


def verify_inventory_orientation(station: str, inventory) -> None:
    matches = [sta for net in inventory for sta in net if sta.code == station]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one StationXML record for {station}, found {len(matches)}")
    sta = matches[0]
    expected = {"N": (0.0, 0.0), "E": (90.0, 0.0), "Z": (0.0, -90.0)}
    for component, target in expected.items():
        channels = [channel for channel in sta.channels if channel.code.startswith("HH") and channel.code.endswith(component)]
        if len(channels) != 1:
            raise ValueError(f"{station}: expected one HH{component} channel")
        channel = channels[0]
        az_error = abs((float(channel.azimuth) - target[0] + 180.0) % 360.0 - 180.0)
        if az_error > 0.01 or abs(float(channel.dip) - target[1]) > 0.01:
            raise ValueError(
                f"{station} {channel.code}: unexpected orientation az={channel.azimuth}, dip={channel.dip}"
            )


def get_event(event_id: str, catalog_path: Path) -> dict[str, str]:
    events = [row for row in read_rows(catalog_path) if row.get("event_id") == event_id]
    if not events:
        raise KeyError(f"Event {event_id!r} is not in {catalog_path}")
    return events[0]


def station_geometry(event: dict[str, str], station: dict[str, str], taup: TauPyModel):
    ev_lat, ev_lon = float(event["event_latitude"]), float(event["event_longitude"])
    st_lat, st_lon = float(station["latitude"]), float(station["longitude"])
    distance_deg = float(locations2degrees(st_lat, st_lon, ev_lat, ev_lon))
    arrivals = taup.get_travel_times(
        source_depth_in_km=float(event["depth_km"]),
        distance_in_degree=distance_deg,
        phase_list=list(PHASES),
    )
    arrival = min(arrivals, key=lambda item: item.time) if arrivals else None
    _, _, baz = gps2dist_azimuth(ev_lat, ev_lon, st_lat, st_lon)
    origin = UTCDateTime(event["event_time_utc"])
    return distance_deg, arrival, origin + arrival.time if arrival else None, float(baz)


def _window_bounds(times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Locate all start/end combinations in sample times measured from the XKS pick."""
    starts = np.searchsorted(times, ANALYSIS_START_S, side="left")
    ends = np.searchsorted(times, ANALYSIS_END_S, side="right") - 1
    starts = np.clip(starts, 0, len(times) - 1)
    ends = np.clip(ends, 0, len(times) - 1)
    pairs = [(int(start), int(end)) for start in starts for end in ends if end - start >= 20]
    if len(pairs) != 100:
        raise ValueError(f"Only {len(pairs)}/100 pick-aware windows fit inside the 50-second input")
    return np.asarray([pair[0] for pair in pairs]), np.asarray([pair[1] for pair in pairs])


def silver_chan_grid_search(
    north: np.ndarray,
    east: np.ndarray,
    relative_times: np.ndarray,
    baz_deg: float,
) -> dict:
    """Minimize SC transverse energy over fast orientation and delay time.

    Ten admissible starts and ten admissible ends are evaluated for each
    parameter pair. Fast/slow projections are formed directly in geographic
    N/E coordinates; this avoids R/T handedness ambiguities. The reported
    objective is corrected-T energy divided by original-R energy.

    Positive ``dt`` means the slow component arrived later than the fast
    component. To undo splitting, the slow projection is advanced by ``dt``.
    """
    north = np.asarray(north, dtype=np.float64)
    east = np.asarray(east, dtype=np.float64)
    relative_times = np.asarray(relative_times, dtype=np.float64)
    finite = np.isfinite(north) & np.isfinite(east) & np.isfinite(relative_times)
    if finite.sum() != len(north) or len(north) != 1000 or len(east) != 1000:
        raise ValueError("SC search requires 1,000 finite aligned N/E samples")
    if not np.allclose(np.diff(relative_times), 1.0 / SAMPLE_RATE_HZ, rtol=0.0, atol=1e-8):
        raise ValueError("SC search requires uniformly sampled 20 Hz data")

    baz = np.deg2rad(baz_deg)
    radial = -east * np.sin(baz) - north * np.cos(baz)
    transverse = -east * np.cos(baz) + north * np.sin(baz)
    start_indices, end_indices = _window_bounds(relative_times)
    cumulative_r = np.r_[0.0, np.cumsum(radial**2)]
    energy_r = cumulative_r[end_indices + 1] - cumulative_r[start_indices]
    if np.any(energy_r <= np.finfo(float).tiny):
        raise ValueError("Original radial signal has zero energy in at least one search window")
    grid = np.full((len(PHI_FAST_FROM_N_DEG), len(DELAY_GRID_S)), np.nan, dtype=np.float64)
    times_for_interp = relative_times

    for phi_index, phi_deg in enumerate(PHI_FAST_FROM_N_DEG):
        phi = np.deg2rad(phi_deg)
        fast = north * np.cos(phi) + east * np.sin(phi)
        slow = -north * np.sin(phi) + east * np.cos(phi)
        for dt_index, delay_s in enumerate(DELAY_GRID_S):
            # Advancing delayed slow energy by dt removes receiver-side splitting.
            slow_advanced = np.interp(times_for_interp + delay_s, times_for_interp, slow)
            north_corrected = fast * np.cos(phi) - slow_advanced * np.sin(phi)
            east_corrected = fast * np.sin(phi) + slow_advanced * np.cos(phi)
            corrected_t = -east_corrected * np.cos(baz) + north_corrected * np.sin(baz)
            cumulative_t = np.r_[0.0, np.cumsum(corrected_t**2)]
            energy_t = cumulative_t[end_indices + 1] - cumulative_t[start_indices]
            grid[phi_index, dt_index] = float(np.mean(energy_t / energy_r))

    best_phi_index, best_dt_index = np.unravel_index(int(np.nanargmin(grid)), grid.shape)
    phi_n = float(PHI_FAST_FROM_N_DEG[best_phi_index])
    phi_rel = wrap_axial(phi_n - baz_deg)
    delay_s = float(DELAY_GRID_S[best_dt_index])
    phi = np.deg2rad(phi_n)
    fast = north * np.cos(phi) + east * np.sin(phi)
    slow = -north * np.sin(phi) + east * np.cos(phi)
    slow_advanced = np.interp(times_for_interp + delay_s, times_for_interp, slow)
    north_corrected = fast * np.cos(phi) - slow_advanced * np.sin(phi)
    east_corrected = fast * np.sin(phi) + slow_advanced * np.cos(phi)
    radial_corrected = -east_corrected * np.sin(baz) - north_corrected * np.cos(baz)
    transverse_corrected = -east_corrected * np.cos(baz) + north_corrected * np.sin(baz)
    near_null_sep = abs((phi_n - baz_deg + 45.0) % 90.0 - 45.0)
    return {
        "phi_from_N_deg": phi_n,
        "phi_relative_to_radial_deg": phi_rel,
        "dt_s": delay_s,
        "objective": float(grid[best_phi_index, best_dt_index]),
        "near_null_geometry_separation_deg": near_null_sep,
        "radial_corrected": radial_corrected,
        "transverse_corrected": transverse_corrected,
        "objective_grid": grid,
        "phi_grid_N_deg": PHI_FAST_FROM_N_DEG.copy(),
        "delay_grid_s": DELAY_GRID_S.copy(),
    }


def conv1d_relu(signal: np.ndarray, kernel: np.ndarray, bias: np.ndarray) -> np.ndarray:
    stride = 2
    kernel_size = kernel.shape[0]
    output_length = (signal.shape[1] - kernel_size) // stride + 1
    if output_length <= 0:
        raise ValueError("CNN Conv1D layer has no output samples")
    output = np.zeros((1, output_length, kernel.shape[2]), dtype=np.float32)
    for kernel_index in range(kernel_size):
        selected = signal[:, kernel_index : kernel_index + output_length * stride : stride, :]
        output += np.matmul(selected, kernel[kernel_index])
    output += bias.reshape(1, 1, -1)
    np.maximum(output, 0.0, out=output)
    return output


def cnn_probability(input_array: np.ndarray, weights_path: Path) -> tuple[float, float]:
    sample = np.asarray(input_array, dtype=np.float32)
    if sample.shape != (1000, 4) or not np.all(np.isfinite(sample)):
        raise ValueError(f"CNN input must be finite and shape (1000,4); got {sample.shape}")
    with np.load(weights_path) as weights:
        features = sample[None, :, :]
        for layer_index in range(9):
            if layer_index in ZERO_PAD_BEFORE_CONV:
                features = np.pad(features, ((0, 0), (1, 1), (0, 0)))
            features = conv1d_relu(
                features,
                weights[f"layer_{layer_index}_kernel"],
                weights[f"layer_{layer_index}_bias"],
            )
        features = features.reshape(1, -1)
        if features.shape[1] != 64:
            raise ValueError(f"Official CNN dense input must have 64 features; got {features.shape[1]}")
        logits = features @ weights["layer_9_kernel"] + weights["layer_9_bias"]
        logits = logits[0] - np.max(logits[0])
        probabilities = np.exp(logits) / np.sum(np.exp(logits))
    return float(probabilities[0]), float(probabilities[1])


def create_sac(trace: Trace, arrival: UTCDateTime, baz: float, component: str) -> Trace:
    result = trace.copy()
    result.stats.sac = AttribDict({
        "a": float(arrival - result.stats.starttime),
        "baz": float(baz),
        "kcmpnm": component.upper(),
    })
    return result


def write_four_sac_inputs(
    station: str,
    event_uid: str,
    arrival: UTCDateTime,
    baz: float,
    raw_r: Trace,
    raw_t: Trace,
    corrected_r: np.ndarray,
    corrected_t: np.ndarray,
    output_root: Path,
) -> list[Path]:
    station_folder = f"{station.ljust(6, 'x')}_1A"
    event_folder = output_root / "SKSOut" / station_folder / event_uid
    event_folder.mkdir(parents=True, exist_ok=True)
    out = []
    for suffix, source, data in (
        ("ro", raw_r, raw_r.data),
        ("to", raw_t, raw_t.data),
        ("rl", raw_r, corrected_r),
        ("tl", raw_t, corrected_t),
    ):
        component = source.copy()
        component.data = np.asarray(data, dtype=np.float32)
        channel_prefix = source.stats.channel[:-1]
        component.stats.channel = channel_prefix + ("R" if suffix in {"ro", "rl"} else "T")
        target = event_folder / f"{station_folder}.{suffix}"
        create_sac(component, arrival, baz, suffix.upper()).write(str(target), format="SAC")
        out.append(target)
    return out


def corrected_horizontal_particle_motion(
    window: Stream,
    pick: UTCDateTime,
    baz: float,
    corrected_r: np.ndarray,
    corrected_t: np.ndarray,
) -> dict:
    rt = window.copy()
    rt.rotate(method="NE->RT", back_azimuth=baz)
    r_trace = rt.select(channel="??R")[0]
    t_trace = rt.select(channel="??T")[0]
    r_trace.data = np.asarray(corrected_r, dtype=np.float64)
    t_trace.data = np.asarray(corrected_t, dtype=np.float64)
    ne = Stream([r_trace, t_trace])
    ne.rotate(method="RT->NE", back_azimuth=baz)
    east_matches = ne.select(channel="HHE")
    north_matches = ne.select(channel="HHN")
    if len(east_matches) != 1 or len(north_matches) != 1:
        raise ValueError("corrected waveform must contain exactly one HHE and one HHN component")
    east_trace = east_matches[0]
    north_trace = north_matches[0]
    if east_trace.stats.npts != north_trace.stats.npts or abs(east_trace.stats.starttime - north_trace.stats.starttime) > 0.5 / SAMPLE_RATE_HZ:
        raise ValueError("corrected HHE/HHN traces are not aligned")
    east_full = np.asarray(east_trace.data, dtype=np.float64)
    north_full = np.asarray(north_trace.data, dtype=np.float64)
    analysis_times = east_trace.times(reftime=pick)
    window_metrics = pca_metrics_by_window(east_full, north_full, analysis_times)
    primary_metrics = window_metrics[PRIMARY_POLARIZATION_WINDOW]
    mask = polarization_window_mask(analysis_times, PRIMARY_POLARIZATION_WINDOW)
    angle = primary_metrics["angle_deg"]
    ratio = primary_metrics["minor_major_ratio"]
    sigma = axial_window_sigma_deg([item["angle_deg"] for item in window_metrics.values()])
    return {
        "angle_deg": angle,
        "minor_major_ratio": ratio,
        "axis_stability_sigma_deg": sigma,
        "east": east_full[mask],
        "north": north_full[mask],
        "times": analysis_times[mask],
        "window_metrics": window_metrics,
    }


def plot_station_page(record: dict, output_pdf: PdfPages) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.3), layout="constrained")
    times = record["relative_times"]
    scale = float(np.nanpercentile(np.abs(np.r_[record["radial"], record["transverse"], record["radial_corrected"], record["transverse_corrected"]]), 99.0))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    axes[0, 0].plot(times, record["radial"] / scale, color="#246d9b", linewidth=0.8, label="R original")
    axes[0, 0].plot(times, record["transverse"] / scale, color="#c45d35", linewidth=0.8, label="T original")
    axes[0, 0].axvspan(-2, 8, color="#cfe5f5", alpha=0.38, zorder=0, label="Primary PCA window")
    axes[0, 0].axvline(0, color="#a82828", linestyle="--", linewidth=0.8)
    axes[0, 0].set_title("Original R and T")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].plot(times, record["radial_corrected"] / scale, color="#246d9b", linewidth=0.8, label="R corrected")
    axes[0, 1].plot(times, record["transverse_corrected"] / scale, color="#c45d35", linewidth=0.8, label="T corrected")
    axes[0, 1].axvspan(-2, 8, color="#cfe5f5", alpha=0.38, zorder=0, label="Primary PCA window")
    axes[0, 1].axvline(0, color="#a82828", linestyle="--", linewidth=0.8)
    axes[0, 1].set_title("Silver–Chan correction on full 50-second traces")
    axes[0, 1].legend(fontsize=8)
    for axis in axes[0]:
        axis.set_xlabel("Seconds from iasp91 theoretical XKS pick")
        axis.set_ylabel("Amplitude / joint 99th percentile")
        axis.grid(alpha=0.2)

    particle = axes[1, 0]
    e = record["corrected_e"]
    n = record["corrected_n"]
    motion_scale = float(np.nanpercentile(np.hypot(e, n), 98.0))
    if not np.isfinite(motion_scale) or motion_scale <= 0:
        motion_scale = 1.0
    particle.scatter(e / motion_scale, n / motion_scale, c=record["analysis_times"], cmap="viridis", s=8, alpha=0.72, linewidths=0)
    baz_rad = np.deg2rad(record["baz"])
    radial = np.array([np.sin(baz_rad), np.cos(baz_rad)])
    particle.plot([-radial[0], radial[0]], [-radial[1], radial[1]], color="#d77926", linestyle="--", linewidth=1.4, label="Expected radial BAZ")
    pca_rad = np.deg2rad(record["pca_deg"])
    pca_axis = np.array([np.sin(pca_rad), np.cos(pca_rad)])
    particle.plot([-pca_axis[0], pca_axis[0]], [-pca_axis[1], pca_axis[1]], color="#1479b8", linewidth=1.8, label="NumPy PCA axis")
    particle.set_aspect("equal", adjustable="datalim")
    particle.set_xlabel("East (normalized)")
    particle.set_ylabel("North (normalized)")
    particle.set_title(
        f"Corrected motion (−2 to +8 s): PCA {record['pca_deg']:.1f}°, "
        f"radial Δ {record['pca_delta']:.1f}°, λmin/λmax {record['pca_ratio']:.3f}"
    )
    particle.legend(fontsize=7)
    particle.grid(alpha=0.2)

    contour = axes[1, 1]
    scores = record["objective_grid"]
    log_scores = np.log10(np.maximum(scores, 1e-12))
    mesh = contour.contourf(record["delay_grid"], record["phi_grid"], log_scores, levels=18, cmap="magma_r")
    contour.scatter([record["dt"]], [record["phi"]], c="#20c997", edgecolors="black", s=50, marker="*", zorder=3, label="Minimum")
    contour.set_xlabel("Delay time δt (s)")
    contour.set_ylabel("Fast axis φ clockwise from north (°; axial)")
    contour.set_title("Silver–Chan corrected-T energy / original-R energy (log10)")
    contour.legend(fontsize=8)
    fig.colorbar(mesh, ax=contour, label="log10 energy ratio")

    fig.suptitle(
        f"{record['station']} — SWSPy tutorial event window + Silver–Chan CNN input\n"
        f"φ={record['phi']:.1f}°  δt={record['dt']:.2f}s  P(acceptable)={record['p_accept']:.3f}  "
        f"{record['status']}  |  0.04–0.5 Hz, 50 s, 20 Hz",
        fontsize=12,
    )
    fig.text(
        0.5, 0.005,
        f"PCA primary window −2 to +8 s; axial window-to-window σ={record['pca_sigma']:.1f}°. Angles: {record['pca_windows_text']}. "
        "CNN scores are uncalibrated for response-corrected SI amplitudes; boundary fits and null geometry remain unresolved.",
        ha="center", fontsize=8, color="#475569",
    )
    output_pdf.savefig(fig, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-id", default="us10006exl")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed_1a_10h"))
    parser.add_argument("--focused-dir", type=Path, default=Path("data/focused_xks_cnn_1a"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/sl_1a_continuous"))
    parser.add_argument("--stationxml", type=Path, default=Path("data/sl_1a_continuous/1A_20160819T000000_10h_stations.xml"))
    parser.add_argument("--stations-csv", type=Path, default=Path("results/network_1A_stations.csv"))
    parser.add_argument("--catalog-csv", type=Path, default=Path("results/global_catalog_active.csv"))
    parser.add_argument("--weights", type=Path, default=Path("models/CNN_XKS_numpy_weights.npz"))
    parser.add_argument("--sac-dir", type=Path, default=Path("data/cnn_sws_input"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/cnn_sws_1a_10h"))
    args = parser.parse_args()

    event = get_event(args.event_id, args.catalog_csv)
    station_rows = read_rows(args.stations_csv)
    station_by_code = {row["station"]: row for row in station_rows}
    inventory = read_inventory(str(args.stationxml))
    taup = TauPyModel("iasp91")
    event_uid = "EQ" + UTCDateTime(event["event_time_utc"]).strftime("%y%j%H%M")
    model_sha256 = hashlib.sha256(args.weights.read_bytes()).hexdigest()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.focused_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    result_rows: list[dict[str, str]] = []
    inference_seconds = 0.0
    total_start = time.perf_counter()
    for station in station_rows:
        code = station["station"]
        record_row: dict[str, str] = {
            "event_id": args.event_id,
            "station": code,
            "phase": "",
            "status": "no_input",
            "probability_acceptable": "",
            "probability_unacceptable": "",
            "cnn_class_at_0p5": "not_run",
            "final_screen": "not_assessed",
            "error": "",
        }
        try:
            verify_inventory_orientation(code, inventory)
            distance, arrival, pick, baz = station_geometry(event, station, taup)
            if arrival is None:
                record_row.update({"status": "no_xks_arrival", "distance_deg": f"{distance:.5f}"})
                result_rows.append(record_row)
                continue
            record_row.update({
                "phase": arrival.name,
                "arrival_utc": str(pick),
                "distance_deg": f"{distance:.8f}",
                "back_azimuth_deg": f"{baz:.8f}",
            })
            input_path = args.raw_dir / f"1A_{code}_20160819T000000_10h.mseed"
            if not input_path.exists() or input_path.stat().st_size < 100_000:
                record_row.update({"status": "no_complete_window", "error": "No raw continuous waveform file."})
                result_rows.append(record_row)
                continue
            continuous, response_errors = response_correct_stream(
                input_path,
                inventory,
                FILTER_MIN_HZ,
                FILTER_MAX_HZ,
            )
            if not continuous:
                record_row.update({
                    "status": "response_correction_failed",
                    "error": "; ".join(response_errors[:3]) or "No usable response-corrected traces.",
                })
                result_rows.append(record_row)
                continue
            window, window_error = cut_complete_window(continuous, code, pick, WINDOW_SECONDS)
            if window is None:
                detail = f"{window_error}; response errors: {'; '.join(response_errors[:2])}" if response_errors else window_error
                record_row.update({"status": "no_complete_window", "error": detail})
                result_rows.append(record_row)
                continue
            if any(not np.all(np.isfinite(trace.data)) for trace in window):
                record_row.update({"status": "invalid_waveform", "error": "Non-finite sample in the 50-second input."})
                result_rows.append(record_row)
                continue
            if any(abs(trace.stats.sampling_rate - SAMPLE_RATE_HZ) > 1e-6 for trace in window):
                record_row.update({"status": "invalid_waveform", "error": "Input is not sampled at exactly 20 Hz."})
                result_rows.append(record_row)
                continue

            focused_path = args.focused_dir / args.event_id / f"{code}_{arrival.name}_50s.mseed"
            focused_path.parent.mkdir(parents=True, exist_ok=True)
            for trace in window:
                trace.data = np.asarray(trace.data, dtype=np.float32)
                if hasattr(trace.stats, "mseed"):
                    trace.stats.mseed.encoding = "FLOAT32"
            window.write(str(focused_path), format="MSEED")

            rt = window.copy()
            rt.rotate(method="NE->RT", back_azimuth=baz)
            radial_trace = rt.select(channel="??R")[0]
            transverse_trace = rt.select(channel="??T")[0]
            north_trace = window.select(channel="??N")[0]
            east_trace = window.select(channel="??E")[0]
            relative_times = radial_trace.times(reftime=pick)
            result = silver_chan_grid_search(
                north_trace.data,
                east_trace.data,
                relative_times,
                baz,
            )
            input_array = np.column_stack((
                radial_trace.data,
                transverse_trace.data,
                result["radial_corrected"],
                result["transverse_corrected"],
            )).astype(np.float32)
            t0 = time.perf_counter()
            p_accept, p_reject = cnn_probability(input_array, args.weights)
            inference_seconds += time.perf_counter() - t0

            polarization = corrected_horizontal_particle_motion(
                window, pick, baz, result["radial_corrected"], result["transverse_corrected"]
            )
            pca_deg = polarization["angle_deg"]
            pca_ratio = polarization["minor_major_ratio"]
            pca_sigma = polarization["axis_stability_sigma_deg"]
            corrected_e = polarization["east"]
            corrected_n = polarization["north"]
            analysis_times = polarization["times"]
            pca_windows_text = " | ".join(
                f"{start:+g}..{end:+g}s {polarization['window_metrics'][name]['angle_deg']:.1f}°"
                for name, start, end in POLARIZATION_WINDOWS_S
            )
            pca_delta = axial_difference_deg(pca_deg, baz)
            fast_geom_sep = result["near_null_geometry_separation_deg"]
            cnn_label = "acceptable" if p_accept >= 0.5 else "unacceptable"
            dt_at_upper_bound = result["dt_s"] >= float(DELAY_GRID_S[-1] - 1e-9)
            if fast_geom_sep < 15.0:
                final_screen = "near_null_geometry_excluded"
                status = "near-null geometry; not a Quality-N rank"
            elif dt_at_upper_bound:
                final_screen = "delay_search_upper_bound_unresolved"
                status = "delay search reached upper bound; pick unresolved"
            elif p_accept >= 0.5:
                final_screen = "cnn_candidate_needs_manual_qc"
                status = "CNN accepted candidate; score calibration and manual QC required"
            else:
                final_screen = "cnn_below_threshold_unvalidated_scale"
                status = "CNN score below threshold; amplitude scale unvalidated"

            filtered_dir = args.sac_dir / "SKSOut"
            sac_files = write_four_sac_inputs(
                code,
                event_uid,
                pick,
                baz,
                radial_trace,
                transverse_trace,
                result["radial_corrected"],
                result["transverse_corrected"],
                args.sac_dir,
            )
            rec = {
                "station": code,
                "phase": arrival.name,
                "pick": pick,
                "distance": distance,
                "baz": baz,
                "phi": result["phi_from_N_deg"],
                "phi_rel": result["phi_relative_to_radial_deg"],
                "dt": result["dt_s"],
                "objective": result["objective"],
                "null_separation": fast_geom_sep,
                "p_accept": p_accept,
                "p_reject": p_reject,
                "cnn_label": cnn_label,
                "final_screen": final_screen,
                "status": status,
                "pca_deg": pca_deg,
                "pca_ratio": pca_ratio,
                "pca_delta": pca_delta,
                "pca_sigma": pca_sigma,
                "pca_windows": polarization["window_metrics"],
                "pca_windows_text": pca_windows_text,
                "corrected_e": corrected_e,
                "corrected_n": corrected_n,
                "radial": np.asarray(radial_trace.data),
                "transverse": np.asarray(transverse_trace.data),
                "radial_corrected": result["radial_corrected"],
                "transverse_corrected": result["transverse_corrected"],
                "relative_times": relative_times,
                "analysis_times": analysis_times,
                "objective_grid": result["objective_grid"],
                "phi_grid": result["phi_grid_N_deg"],
                "delay_grid": result["delay_grid_s"],
            }
            records.append(rec)
            record_row.update({
                "status": status,
                "distance_deg": f"{distance:.8f}",
                "back_azimuth_deg": f"{baz:.8f}",
                "phi_fast_from_N_deg": f"{rec['phi']:.4f}",
                "phi_fast_relative_to_radial_deg": f"{rec['phi_rel']:.4f}",
                "delay_time_s": f"{rec['dt']:.4f}",
                "delay_search_upper_bound_hit": str(dt_at_upper_bound),
                "sc_corrected_transverse_energy_ratio": f"{rec['objective']:.8g}",
                "nearest_null_geometry_separation_deg": f"{fast_geom_sep:.4f}",
                "probability_acceptable": f"{p_accept:.8f}",
                "probability_unacceptable": f"{p_reject:.8f}",
                "cnn_class_at_0p5": cnn_label,
                "final_screen": final_screen,
                "corrected_polarization_numpy_deg": f"{pca_deg:.4f}",
                "corrected_polarization_primary_window_s": "-2,+8",
                "corrected_polarization_window_stability_axial_sigma_deg": f"{pca_sigma:.4f}",
                "corrected_polarization_radial_difference_deg": f"{pca_delta:.4f}",
                "corrected_horizontal_minor_major_variance_ratio": f"{pca_ratio:.8f}",
                "near_linear_screen_ratio_le_0p10": str(pca_ratio <= 0.10),
                "input_format": "1000 samples at 20 Hz; R,T,Rcorr,Tcorr; no amplitude normalization",
                "filter_hz": f"{FILTER_MIN_HZ:.2f}-{FILTER_MAX_HZ:.1f}",
                "model_sha256": model_sha256,
                "sac_input_count": str(len(sac_files)),
                "error": "",
            })
            for window_name, metrics in polarization["window_metrics"].items():
                record_row[f"corrected_pca_{window_name}_angle_deg"] = f"{metrics['angle_deg']:.4f}"
                record_row[f"corrected_pca_{window_name}_minor_major_ratio"] = f"{metrics['minor_major_ratio']:.8f}"
                record_row[f"corrected_pca_{window_name}_sample_count"] = str(metrics["n_samples"])
            result_rows.append(record_row)
            print(f"{code}: phi={rec['phi']:.1f} deg, dt={rec['dt']:.2f}s, P(A)={p_accept:.3f}, {status}")
        except Exception as error:
            record_row.update({"status": "processing_error", "error": f"{type(error).__name__}: {error}"})
            result_rows.append(record_row)
            print(f"{code}: {type(error).__name__}: {error}")

    total_seconds = time.perf_counter() - total_start
    result_csv = args.output_dir / f"1A_{args.event_id}_cnn_sws_predictions.csv"
    write_rows(result_csv, result_rows)
    pdf_path = args.output_dir / f"1A_{args.event_id}_cnn_sws_single_event.pdf"
    with PdfPages(pdf_path) as pdf:
        fig, ax = plt.subplots(figsize=(12, 8.5), layout="constrained")
        ax.axis("off")
        ax.text(0.0, 1.0, "Single-event shear-wave splitting: SWSPy windowing + CNN-SWS", fontsize=18, weight="bold", va="top")
        summary = (
            f"Event: {args.event_id} | M{float(event['magnitude']):.1f} | origin {event['event_time_utc']} UTC\n"
            f"Stations in 1A inventory: {len(station_rows)} | valid 50-s windows classified: {len(records)}\n"
            f"Preprocessing: response-corrected continuous data; 0.04–0.5 Hz; 20 Hz; 50-s XKS windows centered on iasp91 picks\n"
            "Polarization: corrected E/N PCA in −2..+8 s (primary); compared against −1..+10, 0..+10 and 0..+15 s. σ is axial circular spread across these four windows.\n"
            f"Splitting parameters: Silver–Chan transverse-energy grid search over 100 SWSPy-style pick windows; φ grid 2°, δt grid 0.05 s, 0–3.95 s\n"
            f"CNN: authors’ CNN_XKS weights; official 9-Conv1D topology; NumPy inference; model SHA-256 {model_sha256}\n"
            f"Inference time: {inference_seconds:.3f} s for {len(records)} windows ({len(records)/inference_seconds:.1f} windows/s) | total run {total_seconds:.1f} s\n"
            "CNN probabilities are not calibrated for these response-corrected SI-amplitude inputs. This event has no hand labels, so accuracy is not measurable.\n"
            "All CNN-accepted picks still need visual/manual rank; near-null geometry is excluded before interpretation and is not a Quality-N rank.\n"
            "Near-linear PCA screen is λmin/λmax ≤0.10 for display only; no A/B/C/N grades are assigned automatically."
        )
        ax.text(0.0, 0.91, summary, fontsize=9.5, va="top", linespacing=1.6, wrap=True)
        display_rows = []
        for row in result_rows:
            if row.get("probability_acceptable"):
                phi = row.get("phi_fast_from_N_deg", "")
                dt = row.get("delay_time_s", "")
                prob = row.get("probability_acceptable", "")
                sep = row.get("nearest_null_geometry_separation_deg", "")
                pol = row.get("corrected_polarization_numpy_deg", "")
                ratio = row.get("corrected_horizontal_minor_major_variance_ratio", "")
                pca_sigma = row.get("corrected_polarization_window_stability_axial_sigma_deg", "")
                display_rows.append([
                    row["station"], row.get("phase", ""), phi, dt, prob, sep, pol, ratio, pca_sigma,
                    "LIN" if row.get("near_linear_screen_ratio_le_0p10") == "True" else "ELLIPTIC",
                    row.get("final_screen", ""),
                ])
            else:
                display_rows.append([row["station"], row.get("phase", ""), "—", "—", "—", "—", "—", "—", "—", "—", row.get("status", "")])
        columns = ["Station", "Phase", "φN°", "δt s", "P(A)", "Null Δ°", "Pol°", "λmin/max", "PCA σ°", "PCA", "Screen"]
        table = ax.table(cellText=display_rows, colLabels=columns, cellLoc="center", loc="lower center", bbox=[0.0, 0.02, 1.0, 0.58])
        table.auto_set_font_size(False)
        table.set_fontsize(5.8)
        table.scale(1.0, 1.15)
        ax.set_title("Station-by-station classification and polarization linearity", loc="left", fontsize=11, pad=4)
        pdf.savefig(fig, dpi=160)
        plt.close(fig)
        for record in records:
            plot_station_page(record, pdf)

    accepted_count = sum(float(row.get("probability_acceptable") or 0.0) >= 0.5 for row in result_rows)
    near_null_count = sum(row.get("final_screen") == "near_null_geometry_excluded" for row in result_rows)
    delay_boundary_count = sum(row.get("delay_search_upper_bound_hit") == "True" for row in result_rows)
    linear_count = sum(row.get("near_linear_screen_ratio_le_0p10") == "True" for row in result_rows)
    print(f"Predictions CSV: {result_csv}")
    print(f"Marked CNN report PDF: {pdf_path} ({len(records) + 1} pages)")
    print(f"CNN P(acceptable) >= 0.5: {accepted_count}/{len(records)}; near-null-geometry exclusions: {near_null_count}; near-linear corrected E/N PCA: {linear_count}/{len(records)}")
    print(f"Silver-Chan delay fits at 3.95 s upper bound: {delay_boundary_count}/{len(records)}; these are unresolved.")
    print(f"Model inference efficiency (this run only): {len(records)/inference_seconds:.1f} windows/s; accuracy is unknown without independent labels.")


if __name__ == "__main__":
    main()
