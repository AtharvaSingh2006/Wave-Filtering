"""Audit station/event geometry and remeasure corrected horizontal polarization.

The NumPy polarization angle is an unoriented line measured clockwise from
north: atan2(E, N) modulo 180 degrees. ObsPy's WGS84 back-azimuth is checked
against an independent vectorized spherical solution before it is used.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
from obspy import UTCDateTime, read, read_inventory
from obspy.geodetics import gps2dist_azimuth, locations2degrees
from swspy.splitting.split import remove_splitting


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def number(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return math.nan


def numpy_station_to_event_geometry(
    station_lat_deg: np.ndarray,
    station_lon_deg: np.ndarray,
    event_lat_deg: float,
    event_lon_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return spherical central angle and station-to-event azimuth in degrees."""
    station_lat = np.deg2rad(np.asarray(station_lat_deg, dtype=np.float64))
    station_lon = np.deg2rad(np.asarray(station_lon_deg, dtype=np.float64))
    event_lat = np.deg2rad(np.float64(event_lat_deg))
    event_lon = np.deg2rad(np.float64(event_lon_deg))
    delta_lat = event_lat - station_lat
    delta_lon = event_lon - station_lon

    haversine = (
        np.sin(delta_lat / 2.0) ** 2
        + np.cos(station_lat) * np.cos(event_lat) * np.sin(delta_lon / 2.0) ** 2
    )
    haversine = np.clip(haversine, 0.0, 1.0)
    central_angle_deg = np.rad2deg(
        2.0 * np.arctan2(np.sqrt(haversine), np.sqrt(np.maximum(0.0, 1.0 - haversine)))
    )
    azimuth_deg = np.mod(
        np.rad2deg(
            np.arctan2(
                np.sin(delta_lon) * np.cos(event_lat),
                np.cos(station_lat) * np.sin(event_lat)
                - np.sin(station_lat) * np.cos(event_lat) * np.cos(delta_lon),
            )
        ),
        360.0,
    )
    return central_angle_deg, azimuth_deg


def numpy_horizontal_polarization(east: np.ndarray, north: np.ndarray) -> tuple[float, float]:
    """Return PCA line angle clockwise from north and minor/major eigenvalue ratio."""
    east = np.asarray(east, dtype=np.float64)
    north = np.asarray(north, dtype=np.float64)
    valid = np.isfinite(east) & np.isfinite(north)
    if valid.sum() < 3:
        return math.nan, math.nan
    horizontal = np.vstack((east[valid], north[valid]))
    covariance = np.cov(horizontal, ddof=1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    major_index = int(np.argmax(eigenvalues))
    major = float(eigenvalues[major_index])
    minor = float(eigenvalues[1 - major_index])
    if not math.isfinite(major) or major <= np.finfo(np.float64).tiny:
        return math.nan, math.nan
    vector_east, vector_north = eigenvectors[:, major_index]
    angle_deg = float(np.mod(np.rad2deg(np.arctan2(vector_east, vector_north)), 180.0))
    return angle_deg, minor / major


def axial_difference_deg(first: float, second: float) -> float:
    if not (math.isfinite(first) and math.isfinite(second)):
        return math.nan
    return float(abs((first - second + 90.0) % 180.0 - 90.0))


POLARIZATION_WINDOWS_S = (
    ("m2_p8", -2.0, 8.0),
    ("m1_p10", -1.0, 10.0),
    ("0_p10", 0.0, 10.0),
    ("0_p15", 0.0, 15.0),
)
PRIMARY_POLARIZATION_WINDOW = POLARIZATION_WINDOWS_S[0][0]


def polarization_window_mask(relative_times: np.ndarray, window_name: str) -> np.ndarray:
    """Select samples nearest the configured time-window boundaries."""
    relative_times = np.asarray(relative_times, dtype=np.float64)
    bounds = {name: (start_s, end_s) for name, start_s, end_s in POLARIZATION_WINDOWS_S}
    if window_name not in bounds:
        raise KeyError(f"Unknown polarization window {window_name!r}")
    time_steps = np.diff(relative_times)
    time_steps = time_steps[np.isfinite(time_steps) & (time_steps > 0.0)]
    if time_steps.size == 0:
        raise ValueError("Cannot infer the sample interval for polarization windows")
    half_sample = float(np.median(time_steps)) * 0.5
    start_s, end_s = bounds[window_name]
    return (
        (relative_times >= start_s - half_sample)
        & (relative_times <= end_s + half_sample)
        & np.isfinite(relative_times)
    )


def pca_metrics_by_window(
    east: np.ndarray,
    north: np.ndarray,
    relative_times: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Measure corrected horizontal PCA in fixed XKS signal windows.

    Boundaries are assigned to the nearest sample, within half a sample, to
    match an arrival-centered ObsPy trim. Axes are axial (0 and 180 degrees
    are equivalent).
    """
    east = np.asarray(east, dtype=np.float64)
    north = np.asarray(north, dtype=np.float64)
    relative_times = np.asarray(relative_times, dtype=np.float64)
    if east.shape != north.shape or east.shape != relative_times.shape or east.ndim != 1:
        raise ValueError("E, N, and relative-time arrays must be aligned one-dimensional arrays")
    measurements: dict[str, dict[str, float]] = {}
    for name, start_s, end_s in POLARIZATION_WINDOWS_S:
        use = polarization_window_mask(relative_times, name)
        angle, ratio = numpy_horizontal_polarization(east[use], north[use])
        measurements[name] = {
            "start_s": start_s,
            "end_s": end_s,
            "angle_deg": angle,
            "minor_major_ratio": ratio,
            "n_samples": int(use.sum()),
        }
    return measurements


def axial_window_sigma_deg(angles_deg: list[float] | np.ndarray) -> float:
    """Return the circular standard deviation for unoriented PCA axes."""
    angles = np.asarray(angles_deg, dtype=np.float64)
    angles = angles[np.isfinite(angles)]
    if angles.size < 2:
        return math.nan
    doubled = np.deg2rad(2.0 * angles)
    resultant = float(np.hypot(np.mean(np.cos(doubled)), np.mean(np.sin(doubled))))
    if resultant <= np.finfo(float).tiny:
        return 90.0
    sigma = math.degrees(math.sqrt(-2.0 * math.log(min(resultant, 1.0))) * 0.5)
    return max(0.0, min(90.0, sigma))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-id", default="us10006exl")
    parser.add_argument("--station-csv", type=Path, default=Path("results/network_1A_stations.csv"))
    parser.add_argument("--catalog-csv", type=Path, default=Path("results/global_catalog_active.csv"))
    parser.add_argument("--stationxml", type=Path, default=Path("data/sl_1a_continuous/1A_20160819T000000_10h_stations.xml"))
    parser.add_argument("--focused-dir", type=Path, default=Path("data/focused_xks_1a"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/sws_1a_10h"))
    args = parser.parse_args()

    event_matches = [row for row in read_rows(args.catalog_csv) if row.get("event_id") == args.event_id]
    if not event_matches:
        raise KeyError(f"Event {args.event_id!r} is not in {args.catalog_csv}")
    event = event_matches[0]
    event_lat = float(event["event_latitude"])
    event_lon = float(event["event_longitude"])
    stations = read_rows(args.station_csv)
    station_by_code = {row["station"]: row for row in stations}
    inventory = read_inventory(str(args.stationxml))
    xml_by_code = {station.code: station for network in inventory for station in network}

    station_lats = np.asarray([float(row["latitude"]) for row in stations], dtype=np.float64)
    station_lons = np.asarray([float(row["longitude"]) for row in stations], dtype=np.float64)
    np_distance, np_baz = numpy_station_to_event_geometry(station_lats, station_lons, event_lat, event_lon)
    geometry_rows: list[dict[str, str]] = []
    for index, row in enumerate(stations):
        code = row["station"]
        sta_xml = xml_by_code.get(code)
        if sta_xml is None:
            raise ValueError(f"Station {code} is missing from StationXML")
        lat_diff = float(sta_xml.latitude) - station_lats[index]
        lon_diff = float(sta_xml.longitude) - station_lons[index]
        channels = [channel for channel in sta_xml.channels if channel.code.startswith("HH")]
        orientation_ok = (
            len([channel for channel in channels if channel.code.endswith("N")]) == 1
            and len([channel for channel in channels if channel.code.endswith("E")]) == 1
            and len([channel for channel in channels if channel.code.endswith("Z")]) == 1
        )
        for channel in channels:
            component = channel.code[-1]
            az = float(channel.azimuth) % 360.0
            dip = float(channel.dip)
            if component == "N":
                orientation_ok &= min(abs(az), abs(az - 360.0)) < 0.01 and abs(dip) < 0.01
            elif component == "E":
                orientation_ok &= abs(az - 90.0) < 0.01 and abs(dip) < 0.01
            elif component == "Z":
                orientation_ok &= abs(az) < 0.01 and abs(abs(dip) - 90.0) < 0.01

        obspy_distance_deg = float(locations2degrees(station_lats[index], station_lons[index], event_lat, event_lon))
        _, az_event_to_station, obspy_baz = gps2dist_azimuth(
            event_lat, event_lon, station_lats[index], station_lons[index]
        )
        geometry_rows.append({
            "station": code,
            "station_latitude_csv": f"{station_lats[index]:.8f}",
            "station_longitude_csv": f"{station_lons[index]:.8f}",
            "station_latitude_stationxml": f"{sta_xml.latitude:.8f}",
            "station_longitude_stationxml": f"{sta_xml.longitude:.8f}",
            "coordinate_match": str(abs(lat_diff) < 1e-8 and abs(lon_diff) < 1e-8),
            "HHZ_HHN_HHE_orientation_ok": str(bool(orientation_ok)),
            "event_latitude": f"{event_lat:.8f}",
            "event_longitude": f"{event_lon:.8f}",
            "distance_numpy_spherical_deg": f"{np_distance[index]:.8f}",
            "distance_obspy_deg": f"{obspy_distance_deg:.8f}",
            "distance_difference_deg": f"{np_distance[index] - obspy_distance_deg:.8f}",
            "baz_numpy_spherical_deg": f"{np_baz[index]:.8f}",
            "baz_obspy_wgs84_deg": f"{obspy_baz:.8f}",
            "baz_difference_axial_deg": f"{axial_difference_deg(np_baz[index], obspy_baz):.8f}",
            "azimuth_event_to_station_obspy_deg": f"{az_event_to_station:.8f}",
        })

    measurement_path = args.results_dir / f"1A_{args.event_id}_sws_measurements.csv"
    measurements = read_rows(measurement_path)
    polarization_by_station: dict[str, dict[str, str]] = {}
    for measurement in measurements:
        if measurement.get("measurement_status") != "computed":
            continue
        station = measurement["station"]
        phase = measurement.get("phase", "SKS")
        trace_path = args.focused_dir / args.event_id / f"{station}_{phase}_50s.mseed"
        if not trace_path.exists():
            continue
        raw = read(str(trace_path))
        corrected = remove_splitting(
            raw.copy(),
            phi=float(measurement["phi_from_Q"]),
            dt=float(measurement["dt"]),
            back_azi=float(measurement["back_azimuth_deg"]),
            event_inclin_angle_at_station=0.0,
            return_BPA=False,
            return_FS=False,
        )
        pick = UTCDateTime(measurement["arrival_utc"])
        east = corrected.select(channel="HHE")
        north = corrected.select(channel="HHN")
        if len(east) != 1 or len(north) != 1:
            continue
        e_trace, n_trace = east[0], north[0]
        if (
            e_trace.stats.npts != n_trace.stats.npts
            or abs(e_trace.stats.starttime - n_trace.stats.starttime) > 0.5 / e_trace.stats.sampling_rate
            or abs(e_trace.stats.sampling_rate - n_trace.stats.sampling_rate) > 1e-8
        ):
            continue
        relative_times = e_trace.times(reftime=pick)
        window_metrics = pca_metrics_by_window(e_trace.data, n_trace.data, relative_times)
        primary_metrics = window_metrics[PRIMARY_POLARIZATION_WINDOW]
        pca_deg = primary_metrics["angle_deg"]
        eigen_ratio = primary_metrics["minor_major_ratio"]
        pca_sigma = axial_window_sigma_deg([
            values["angle_deg"] for values in window_metrics.values()
        ])
        baz_deg = float(measurement["back_azimuth_deg"])
        old_pol = number(measurement, "src_pol_from_N")
        fields = {
            "src_pol_numpy_from_N": f"{pca_deg:.6f}" if math.isfinite(pca_deg) else "",
            "src_pol_eigenvalue_minor_major_ratio": f"{eigen_ratio:.8f}" if math.isfinite(eigen_ratio) else "",
            "src_pol_linearity_1_minus_ratio": f"{1.0 - eigen_ratio:.8f}" if math.isfinite(eigen_ratio) else "",
            "src_pol_near_linear_screen_ratio_le_0p10": str(eigen_ratio <= 0.10) if math.isfinite(eigen_ratio) else "",
            "src_pol_deviation_from_radial_BAZ_deg": f"{axial_difference_deg(pca_deg, baz_deg):.6f}" if math.isfinite(pca_deg) else "",
            "src_pol_numpy_vs_SWSPy_axial_difference_deg": f"{axial_difference_deg(pca_deg, old_pol):.6f}" if math.isfinite(pca_deg) and math.isfinite(old_pol) else "",
            "src_pol_primary_window_s": "-2,+8",
            "src_pol_axis_stability_axial_sigma_deg": f"{pca_sigma:.6f}" if math.isfinite(pca_sigma) else "",
        }
        for name, values in window_metrics.items():
            fields[f"src_pol_pca_{name}_deg"] = f"{values['angle_deg']:.6f}" if math.isfinite(values["angle_deg"]) else ""
            fields[f"src_pol_ratio_{name}"] = f"{values['minor_major_ratio']:.8f}" if math.isfinite(values["minor_major_ratio"]) else ""
            fields[f"src_pol_samples_{name}"] = str(values["n_samples"])
        polarization_by_station[station] = fields

    empty_polarization_fields = {
        "src_pol_numpy_from_N": "",
        "src_pol_eigenvalue_minor_major_ratio": "",
        "src_pol_linearity_1_minus_ratio": "",
        "src_pol_near_linear_screen_ratio_le_0p10": "",
        "src_pol_deviation_from_radial_BAZ_deg": "",
        "src_pol_numpy_vs_SWSPy_axial_difference_deg": "",
        "src_pol_primary_window_s": "-2,+8",
        "src_pol_axis_stability_axial_sigma_deg": "",
    }
    for name, _, _ in POLARIZATION_WINDOWS_S:
        empty_polarization_fields.update({
            f"src_pol_pca_{name}_deg": "",
            f"src_pol_ratio_{name}": "",
            f"src_pol_samples_{name}": "",
        })

    for row in measurements:
        row.update(polarization_by_station.get(row.get("station", ""), empty_polarization_fields))
    write_rows(measurement_path, measurements)

    screen_path = args.results_dir / f"1A_{args.event_id}_paper_quality_screen.csv"
    if screen_path.exists():
        screen_rows = read_rows(screen_path)
        for row in screen_rows:
            row.update(polarization_by_station.get(row.get("station", ""), empty_polarization_fields))
        write_rows(screen_path, screen_rows)

    audit_path = args.results_dir / f"1A_{args.event_id}_coordinate_polarization_audit.csv"
    write_rows(audit_path, geometry_rows)
    max_distance_diff = max(abs(float(row["distance_difference_deg"])) for row in geometry_rows)
    max_baz_diff = max(float(row["baz_difference_axial_deg"]) for row in geometry_rows)
    coord_match_count = sum(row["coordinate_match"] == "True" for row in geometry_rows)
    orientation_ok_count = sum(row["HHZ_HHN_HHE_orientation_ok"] == "True" for row in geometry_rows)
    print(f"Wrote {audit_path}")
    print(f"Station coordinate matches: {coord_match_count}/{len(stations)}; canonical HHZ/HHN/HHE orientation: {orientation_ok_count}/{len(stations)}")
    print(f"NumPy spherical vs ObsPy max distance difference: {max_distance_diff:.6g} deg")
    print(f"NumPy spherical vs ObsPy max axial BAZ difference: {max_baz_diff:.6g} deg")
    print(
        f"Recomputed corrected E/N PCA in the primary -2 to +8 s window and compared four windows "
        f"for {len(polarization_by_station)} stations."
    )


if __name__ == "__main__":
    main()
