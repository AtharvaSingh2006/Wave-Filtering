"""Response-correct a 1A continuous window and measure its SKS splitting."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import swspy
from swspy.splitting import split as sws_split
from obspy import Stream, UTCDateTime, read, read_inventory
from obspy.core.util.attribdict import AttribDict
from obspy.geodetics import gps2dist_azimuth, locations2degrees
from obspy.taup import TauPyModel
from swspy.splitting.split import remove_splitting


PHASES = ("SKS", "SKKS", "PKS")
CHANNEL_COMPONENTS = ("Z", "N", "E")
FILTER_MIN_HZ = 0.01
FILTER_MAX_HZ = 0.5


def _find_pol_numpy_compatible(x, y):
    """Return the dominant polarization using real symmetric eigensolving.

    SWSPy 1.0.2 uses ``numpy.linalg.eig`` on a real covariance matrix and can
    receive complex eigenvectors with current NumPy builds. The covariance is
    real and symmetric, so ``eigh`` is the appropriate real-valued solver.
    """
    covariance = np.cov(np.vstack((np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))))
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    vector = eigenvectors[:, int(np.argmax(eigenvalues))]
    angle = float(np.rad2deg(np.arctan2(vector[0], vector[1])))
    if angle < 0:
        angle += 180.0
    largest = float(np.max(eigenvalues))
    error = 180.0 * float(np.min(eigenvalues)) / largest if largest > 0 else float("nan")
    return angle, error


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def get_event(path: Path, event_id: str) -> dict[str, str]:
    matches = [row for row in read_csv_rows(path) if row["event_id"] == event_id]
    if not matches:
        raise KeyError(f"Event {event_id} not found in {path}")
    return matches[0]


def predict_arrival(event: dict[str, str], station: dict[str, str], model: TauPyModel):
    distance_deg = locations2degrees(
        float(station["latitude"]),
        float(station["longitude"]),
        float(event["event_latitude"]),
        float(event["event_longitude"]),
    )
    arrivals = model.get_travel_times(
        source_depth_in_km=float(event["depth_km"]),
        distance_in_degree=distance_deg,
        phase_list=list(PHASES),
    )
    if not arrivals:
        return distance_deg, None, None, None
    arrival = min(arrivals, key=lambda item: item.time)
    origin_time = UTCDateTime(event["event_time_utc"])
    _, _, back_azimuth = gps2dist_azimuth(
        float(event["event_latitude"]),
        float(event["event_longitude"]),
        float(station["latitude"]),
        float(station["longitude"]),
    )
    return distance_deg, arrival, origin_time + arrival.time, back_azimuth


def response_correct_stream(raw_path: Path, inventory, freq_min: float, freq_max: float):
    raw = read(str(raw_path))
    if any(np.ma.isMaskedArray(trace.data) for trace in raw):
        raw = raw.split()
    processed = Stream()
    errors = []
    for trace in raw:
        if trace.stats.npts < 100:
            continue
        try:
            trace = trace.copy()
            trace.detrend("demean")
            trace.detrend("linear")
            trace.remove_response(
                inventory=inventory,
                output="VEL",
                pre_filt=[max(freq_min * 0.5, 0.001), freq_min, freq_max, freq_max * 1.5],
                water_level=60.0,
            )
            trace.filter(
                "bandpass",
                freqmin=freq_min,
                freqmax=freq_max,
                corners=4,
                zerophase=True,
            )
            if trace.stats.sampling_rate > 40.0:
                factor = int(round(trace.stats.sampling_rate / 20.0))
                if not math.isclose(trace.stats.sampling_rate / factor, 20.0, abs_tol=1e-6):
                    raise ValueError(f"Cannot decimate {trace.stats.sampling_rate:g} Hz to 20 Hz")
                trace.decimate(factor, no_filter=False, strict_length=False)
            trace.data = np.asarray(trace.data, dtype=np.float32)
            processed.append(trace)
        except Exception as error:
            errors.append(f"{trace.id}: {type(error).__name__}: {error}")
    processed.sort()
    return processed, errors


def cut_complete_window(stream: Stream, station: str, arrival_time: UTCDateTime, seconds: float):
    sample_rate = min(trace.stats.sampling_rate for trace in stream)
    start = arrival_time - seconds / 2.0
    end = start + seconds - 1.0 / sample_rate
    cut = Stream()
    for component in CHANNEL_COMPONENTS:
        options = [
            trace for trace in stream
            if trace.stats.channel.endswith(component)
            and trace.stats.starttime <= start
            and trace.stats.endtime >= end
        ]
        if not options:
            return None, f"no continuous {component} trace across predicted XKS window"
        trace = max(options, key=lambda item: item.stats.npts).copy()
        trace.trim(starttime=start, endtime=end, nearest_sample=True, pad=False)
        expected_npts = int(round(seconds * sample_rate))
        if trace.stats.npts != expected_npts:
            return None, f"{component} window has {trace.stats.npts}, expected {expected_npts} samples"
        trace.stats.station = station
        cut.append(trace)
    if len({round(trace.stats.sampling_rate, 8) for trace in cut}) != 1:
        return None, "components have different sampling rates"
    if max(abs(float(trace.stats.starttime - cut[0].stats.starttime)) for trace in cut) > 1.0 / sample_rate:
        return None, "components are not aligned in time"
    return cut, ""


def make_record_section(windows: dict[str, Stream], arrival_times: dict[str, UTCDateTime], output: Path):
    stations = sorted(windows)
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True, sharey=True, layout="constrained")
    colors = {"Z": "#667085", "N": "#1877b8", "E": "#d16d15"}
    for axis, component in zip(axes, CHANNEL_COMPONENTS):
        for index, station in enumerate(stations):
            trace = next(tr for tr in windows[station] if tr.stats.channel.endswith(component))
            times = trace.times(reftime=trace.stats.starttime)
            arrival_offset = float(arrival_times[station] - trace.stats.starttime)
            relative_times = times - arrival_offset
            data = np.asarray(trace.data, dtype=float)
            amplitude = np.nanpercentile(np.abs(data), 99.5)
            if not np.isfinite(amplitude) or amplitude == 0:
                continue
            axis.plot(relative_times, index + 0.36 * data / amplitude, color=colors[component], linewidth=0.55)
        axis.axvline(0, color="#be123c", linewidth=0.8, linestyle="--")
        axis.set_ylabel(f"{component}\nstation")
        axis.set_yticks(range(len(stations)), stations, fontsize=7)
        axis.grid(True, axis="x", alpha=0.25)
    axes[-1].set_xlabel("Seconds from predicted XKS arrival")
    axes[0].set_title("Response-corrected XKS windows (normalized wiggle record section)")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_cnn_sws_inputs(
    window: Stream,
    station: str,
    event_uid: str,
    arrival_time: UTCDateTime,
    back_azimuth: float,
    phi_from_q: float,
    delay_s: float,
    output_root: Path,
):
    station_folder = f"{station.ljust(6, 'x')}_1A"
    event_folder = output_root / "SKSOut" / station_folder / event_uid
    event_folder.mkdir(parents=True, exist_ok=True)

    raw_rt = window.copy()
    raw_rt.rotate(method="NE->RT", back_azimuth=back_azimuth)
    corrected = remove_splitting(
        window.copy(),
        phi=float(phi_from_q),
        dt=float(delay_s),
        back_azi=float(back_azimuth),
        event_inclin_angle_at_station=0.0,
        return_BPA=False,
        return_FS=False,
    )
    corrected_rt = corrected.copy()
    corrected_rt.rotate(method="NE->RT", back_azimuth=back_azimuth)

    component_map = {
        "ro": raw_rt.select(channel="??R"),
        "to": raw_rt.select(channel="??T"),
        "rl": corrected_rt.select(channel="??R"),
        "tl": corrected_rt.select(channel="??T"),
    }
    written = []
    for suffix, component_stream in component_map.items():
        if len(component_stream) != 1:
            raise RuntimeError(f"Expected one {suffix} component, found {len(component_stream)}")
        trace = component_stream[0].copy()
        trace.stats.sac = AttribDict({
            "a": float(arrival_time - trace.stats.starttime),
            "baz": float(back_azimuth),
            "kcmpnm": suffix.upper(),
        })
        output_path = event_folder / f"{station_folder}.{suffix}"
        trace.write(str(output_path), format="SAC")
        written.append(str(output_path))
    return written


def make_event_sws_object(
    stations: list[str],
    windows: dict[str, Stream],
    metadata: dict[str, dict],
    event_uid: str,
    origin_time: UTCDateTime,
):
    """Build one SWSPy object for this event's multiple receiver stations."""
    combined = Stream()
    for station in stations:
        for trace in windows[station].copy():
            trace.data = np.asarray(trace.data, dtype=np.float64)
            combined.append(trace)
    combined.sort()
    obj = swspy.splitting.create_splitting_object(
        combined,
        event_uid=event_uid,
        stations_in=stations,
        S_phase_arrival_times=[metadata[station]["arrival_time"] for station in stations],
        back_azis_all_stations=[metadata[station]["back_azimuth"] for station in stations],
        receiver_inc_angles_all_stations=[0.0 for _ in stations],
    )
    obj.origin_time = origin_time
    obj.overall_win_start_pre_fast_S_pick = 4.0
    obj.win_S_pick_tolerance = 1.0
    obj.overall_win_start_post_fast_S_pick = 20.0
    obj.rotate_step_deg = 2.0
    obj.max_t_shift_s = 2.0
    obj.n_win = 10
    return obj


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-id", default="us10006exl")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/sl_1a_continuous"))
    parser.add_argument(
        "--raw-pattern",
        default="1A_*_20160819T000000_10h.mseed",
        help="Glob pattern for the downloaded 10-hour station files",
    )
    parser.add_argument(
        "--stationxml",
        type=Path,
        default=Path("data/sl_1a_continuous/1A_20160819T000000_10h_stations.xml"),
    )
    parser.add_argument(
        "--stations-csv",
        type=Path,
        default=Path("results/network_1A_stations.csv"),
    )
    parser.add_argument(
        "--catalog-csv",
        type=Path,
        default=Path("results/global_catalog_active.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed_1a_10h"))
    parser.add_argument("--focused-dir", type=Path, default=Path("data/focused_xks_1a"))
    parser.add_argument("--cnn-input-dir", type=Path, default=Path("data/cnn_sws_input"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/sws_1a_10h"))
    parser.add_argument("--freq-min", type=float, default=FILTER_MIN_HZ)
    parser.add_argument("--freq-max", type=float, default=FILTER_MAX_HZ)
    parser.add_argument("--window-seconds", type=float, default=50.0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--reuse-processed",
        action="store_true",
        help="Reuse response-corrected files already in --output-dir",
    )
    args = parser.parse_args()

    event = get_event(args.catalog_csv, args.event_id)
    station_rows = read_csv_rows(args.stations_csv)
    stations_by_code = {row["station"]: row for row in station_rows}
    inventory = read_inventory(str(args.stationxml))
    model = TauPyModel("iasp91")
    sws_split._find_pol = _find_pol_numpy_compatible
    event_uid = "EQ" + UTCDateTime(event["event_time_utc"]).strftime("%y%j%H%M")
    output_stem = f"1A_{args.event_id}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.focused_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    measurement_rows = []
    coverage_rows = []
    valid_windows: dict[str, Stream] = {}
    arrivals_by_station: dict[str, UTCDateTime] = {}
    splitting_metadata: dict[str, dict] = {}
    input_pattern = args.raw_dir / args.raw_pattern
    raw_files = sorted(args.raw_dir.glob(args.raw_pattern))
    if not raw_files:
        raise FileNotFoundError(f"No raw waveforms match {input_pattern}")
    raw_station_codes = {
        pieces[1]
        for raw_path in raw_files
        if len(pieces := raw_path.stem.split("_")) > 2
    }

    print(
        f"Event {args.event_id}: M{float(event['magnitude']):.1f}, "
        f"origin {event['event_time_utc']}; processing {len(raw_files)} station files"
    )
    for raw_path in raw_files:
        try:
            raw_head = read(str(raw_path), headonly=True)
            station_code = raw_head[0].stats.station
        except Exception as error:
            coverage_rows.append({"file": raw_path.name, "station": "", "status": "read_error", "error": str(error)})
            continue
        station_info = stations_by_code.get(station_code)
        if station_info is None:
            coverage_rows.append({"file": raw_path.name, "station": station_code, "status": "no_coordinates", "error": ""})
            continue

        distance_deg, phase_arrival, arrival_time, back_azimuth = predict_arrival(event, station_info, model)
        if phase_arrival is None:
            coverage_rows.append({"file": raw_path.name, "station": station_code, "status": "no_XKS_arrival", "error": ""})
            continue

        processed_path = args.output_dir / f"{raw_path.stem}_vel_{int(round(20))}hz.mseed"
        if args.reuse_processed and processed_path.exists():
            processed_stream = read(str(processed_path))
            processing_errors = []
        else:
            processed_stream, processing_errors = response_correct_stream(
                raw_path, inventory, args.freq_min, args.freq_max
            )
            if processed_stream:
                processed_stream.write(str(processed_path), format="MSEED")
        event_window, window_error = cut_complete_window(
            processed_stream, station_code, arrival_time, args.window_seconds
        ) if processed_stream else (None, "no processed traces")

        coverage_rows.append({
            "file": raw_path.name,
            "station": station_code,
            "phase": phase_arrival.name,
            "arrival_utc": str(arrival_time),
            "distance_deg": f"{distance_deg:.3f}",
            "processed_trace_count": len(processed_stream),
            "processed_path": str(processed_path) if processed_stream else "",
            "status": "window_ready" if event_window else "no_complete_XKS_window",
            "error": "; ".join(processing_errors + ([window_error] if window_error else [])),
        })
        if not event_window:
            print(f"{station_code}: skipped SWS window ({window_error})")
            continue

        focus_path = args.focused_dir / args.event_id / f"{station_code}_{phase_arrival.name}_{int(args.window_seconds)}s.mseed"
        focus_path.parent.mkdir(parents=True, exist_ok=True)
        event_window.write(str(focus_path), format="MSEED")
        valid_windows[station_code] = event_window
        arrivals_by_station[station_code] = arrival_time
        splitting_metadata[station_code] = {
            "distance_deg": distance_deg,
            "phase": phase_arrival.name,
            "arrival_time": arrival_time,
            "back_azimuth": back_azimuth,
        }

    # Treat these windows as multiple receiver observations of one event,
    # matching the SWSPy tutorial's single-event analysis structure.
    station_codes = sorted(valid_windows)
    fatal_station_errors: dict[str, str] = {}

    def diagnose_fatal_station(codes: list[str]) -> list[str]:
        """Isolate station-specific exceptions before the shared analysis run."""
        if not codes:
            return []
        trial = make_event_sws_object(codes, valid_windows, splitting_metadata, output_stem,
                                      UTCDateTime(event["event_time_utc"]))
        try:
            trial.perform_sws_analysis(
                coord_system="ZNE", sws_method="EV", return_clusters_data=True,
                num_threads=args.threads,
            )
            return []
        except Exception as error:
            if len(codes) == 1:
                fatal_station_errors[codes[0]] = f"{type(error).__name__}: {error}"
                return codes
            middle = len(codes) // 2
            return diagnose_fatal_station(codes[:middle]) + diagnose_fatal_station(codes[middle:])

    sws_object = make_event_sws_object(
        station_codes, valid_windows, splitting_metadata, output_stem,
        UTCDateTime(event["event_time_utc"]),
    )
    try:
        result = sws_object.perform_sws_analysis(
            coord_system="ZNE", sws_method="EV", return_clusters_data=True,
            num_threads=args.threads,
        )
    except Exception:
        failed_codes = set(diagnose_fatal_station(station_codes))
        if not failed_codes:
            raise
        for station_code in sorted(failed_codes):
            print(f"{station_code}: excluded from shared SWSPy run ({fatal_station_errors[station_code]})")
        station_codes = [code for code in station_codes if code not in failed_codes]
        sws_object = make_event_sws_object(
            station_codes, valid_windows, splitting_metadata, output_stem,
            UTCDateTime(event["event_time_utc"]),
        )
        result = sws_object.perform_sws_analysis(
            coord_system="ZNE", sws_method="EV", return_clusters_data=True,
            num_threads=args.threads,
        )

    result_by_station = {}
    if result is not None and not result.empty:
        result_by_station = {row["station"]: row.to_dict() for _, row in result.iterrows()}

    for station_code in sorted(valid_windows):
        station_meta = splitting_metadata[station_code]
        event_window = valid_windows[station_code]
        phase_name = station_meta["phase"]
        arrival_time = station_meta["arrival_time"]
        back_azimuth = station_meta["back_azimuth"]
        if station_code in fatal_station_errors:
            measurement_rows.append({
                "event_id": args.event_id,
                "event_uid": event_uid,
                "event_time_utc": event["event_time_utc"],
                "magnitude": event["magnitude"],
                "station": station_code,
                "phase": phase_name,
                "arrival_utc": str(arrival_time),
                "distance_deg": station_meta["distance_deg"],
                "back_azimuth_deg": back_azimuth,
                "measurement_status": "failed",
                "error": fatal_station_errors[station_code],
            })
            continue
        result_row = result_by_station.get(station_code)
        if result_row is None:
            measurement_rows.append({
                "event_id": args.event_id,
                "event_uid": event_uid,
                "event_time_utc": event["event_time_utc"],
                "magnitude": event["magnitude"],
                "station": station_code,
                "phase": phase_name,
                "arrival_utc": str(arrival_time),
                "distance_deg": station_meta["distance_deg"],
                "back_azimuth_deg": back_azimuth,
                "measurement_status": "failed",
                "error": "SWSPy returned no clustered measurement for this station.",
            })
            print(f"{station_code}: no clustered SWSPy solution")
            continue

        measurement = {
            "event_id": args.event_id,
            "event_uid": event_uid,
            "event_time_utc": event["event_time_utc"],
            "magnitude": event["magnitude"],
            "station": station_code,
            "phase": phase_name,
            "arrival_utc": str(arrival_time),
            "distance_deg": station_meta["distance_deg"],
            "back_azimuth_deg": back_azimuth,
            "window_start_utc": str(event_window[0].stats.starttime),
            "window_seconds": args.window_seconds,
            "sampling_rate_hz": event_window[0].stats.sampling_rate,
            "measurement_status": "computed",
            **result_row,
            "cnn_sws_sac_inputs": "EV preview only; not valid CNN-SWS inputs",
        }
        measurement_rows.append(measurement)
        phi = float(result_row.get("phi_from_Q", float("nan")))
        delay = float(result_row.get("dt", float("nan")))
        if math.isfinite(phi) and math.isfinite(delay):
            try:
                write_cnn_sws_inputs(
                    event_window.copy(), station_code, event_uid, arrival_time,
                    back_azimuth, phi, delay, args.cnn_input_dir,
                )
            except Exception as error:
                measurement["cnn_sws_sac_inputs"] = f"EV preview write error: {error}"
        print(
            f"{station_code}: {phase_name} | "
            f"phi_N={result_row.get('phi_from_N')}° | dt={result_row.get('dt')} s"
        )

    try:
        # The shared object's station list also contains receivers skipped by
        # SWSPy because they lack a cluster. Plot only rows that have results.
        sws_object.stations_list = sorted(result_by_station)
        sws_object.plot(
            outdir=str(args.results_dir / "plots"),
            suppress_direct_plotting=True,
        )
    except Exception as error:
        print(f"SWSPy station-plot generation warning: {type(error).__name__}: {error}")

    for station_code in sorted(set(stations_by_code) - raw_station_codes):
        station_info = stations_by_code[station_code]
        distance_deg, phase_arrival, arrival_time, _ = predict_arrival(event, station_info, model)
        coverage_rows.append({
            "file": "",
            "station": station_code,
            "phase": phase_arrival.name if phase_arrival else "",
            "arrival_utc": str(arrival_time) if arrival_time else "",
            "distance_deg": f"{distance_deg:.3f}",
            "processed_trace_count": 0,
            "processed_path": "",
            "status": "no_raw_waveform",
            "error": "No 10-hour MiniSEED file was downloaded for this station.",
        })

    coverage_path = args.results_dir / f"{output_stem}_processing_coverage.csv"
    pd.DataFrame(coverage_rows).to_csv(coverage_path, index=False)
    result_path = args.results_dir / f"{output_stem}_sws_measurements.csv"
    pd.DataFrame(measurement_rows).to_csv(result_path, index=False)

    if valid_windows:
        combined = Stream()
        for station in sorted(valid_windows):
            combined += valid_windows[station]
        combined.sort()
        focus_mseed = args.focused_dir / args.event_id / f"{output_stem}_all_stations_{int(args.window_seconds)}s.mseed"
        combined.write(str(focus_mseed), format="MSEED")
        make_record_section(
            valid_windows,
            arrivals_by_station,
            args.results_dir / f"{output_stem}_xks_record_section.png",
        )
    print(f"Processed waveform folder: {args.output_dir}")
    print(f"XKS event windows ({len(valid_windows)} stations): {args.focused_dir / args.event_id}")
    print(f"SWS results: {result_path}")
    print(f"Processing availability: {coverage_path}")
    print(f"EV preview SAC windows (not CNN-valid): {args.cnn_input_dir / 'SKSOut'}")


if __name__ == "__main__":
    main()
