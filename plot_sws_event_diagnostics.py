"""Make one combined, pick-centered SWS view and a station-diagnostics PDF."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.backends.backend_pdf import PdfPages
from obspy import read, UTCDateTime
from swspy.splitting.split import remove_splitting
from audit_1a_geometry_and_polarization import (
    POLARIZATION_WINDOWS_S,
    PRIMARY_POLARIZATION_WINDOW,
    axial_window_sigma_deg,
    pca_metrics_by_window,
    polarization_window_mask,
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def number(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def make_combined_figure(
    event_id: str,
    measurement_rows: list[dict[str, str]],
    coverage_rows: list[dict[str, str]],
    station_rows: list[dict[str, str]],
    focused_dir: Path,
    output: Path,
) -> None:
    measurements = {row.get("station", ""): row for row in measurement_rows}
    coverage = {row.get("station", ""): row for row in coverage_rows}
    stations = sorted(row["station"] for row in station_rows)
    count = len(stations)
    fig, ax = plt.subplots(figsize=(21, 13), layout="constrained")
    ax.set_xlim(-26.5, 98)
    ax.set_ylim(-0.9, count + 1.1)
    ax.set_xticks(np.arange(-25, 26, 5))
    ax.grid(True, axis="x", color="#cbd5e1", linewidth=0.5, zorder=0)
    ax.axvspan(-2, 8, color="#d7eaf8", alpha=0.75, zorder=0)
    for boundary in (-1, 0, 10, 15):
        ax.axvline(boundary, color="#80a9c7", linewidth=0.55, linestyle=":", zorder=1)
    ax.axvspan(-1, 1, color="#ffeb99", alpha=0.45, zorder=0)
    ax.axvline(0, color="#c2413a", linewidth=1.1, linestyle="--", zorder=3)

    y_positions = {station: count - index - 1 for index, station in enumerate(stations)}
    for station in stations:
        y = y_positions[station]
        measurement = measurements.get(station, {})
        coverage_row = coverage.get(station, {})
        is_computed = measurement.get("measurement_status") == "computed"
        phase = coverage_row.get("phase") or measurement.get("phase") or "SKS"
        pick_text = "—"
        pick_iso = coverage_row.get("arrival_utc") or measurement.get("arrival_utc")
        if pick_iso:
            try:
                pick_text = UTCDateTime(pick_iso).strftime("%H:%M:%S") + f".{int(UTCDateTime(pick_iso).microsecond / 1000):03d}"
            except Exception:
                pick_text = pick_iso

        trace_file = focused_dir / event_id / f"{station}_{phase}_50s.mseed"
        plotted_raw = False
        if trace_file.exists():
            try:
                raw_zne = read(str(trace_file))
                pick = UTCDateTime(pick_iso or measurement["arrival_utc"])
                baz = number(measurement, "back_azimuth_deg")
                if not np.isfinite(baz):
                    baz = number(coverage_row, "back_azimuth_deg")
                raw_rt = raw_zne.copy()
                raw_rt.rotate(method="NE->RT", back_azimuth=baz)
                raw_t = raw_rt.select(channel="??T")[0]
                raw_y = np.asarray(raw_t.data, dtype=float)
                relative_times = raw_t.times(reftime=pick)
                corrected_y = None
                if is_computed:
                    corrected = remove_splitting(
                        raw_zne.copy(),
                        phi=number(measurement, "phi_from_Q"),
                        dt=number(measurement, "dt"),
                        back_azi=baz,
                        event_inclin_angle_at_station=0.0,
                        return_BPA=False,
                        return_FS=False,
                    )
                    corrected.rotate(method="NE->RT", back_azimuth=baz)
                    corrected_trace = corrected.select(channel="??T")[0]
                    corrected_y = np.asarray(corrected_trace.data, dtype=float)
                all_values = np.abs(raw_y) if corrected_y is None else np.abs(np.r_[raw_y, corrected_y])
                scale = float(np.nanpercentile(all_values, 99.5))
                if np.isfinite(scale) and scale > 0:
                    ax.plot(relative_times, y + 0.34 * raw_y / scale, color="#2878a5", linewidth=0.65, zorder=2)
                    if corrected_y is not None:
                        ax.plot(relative_times, y + 0.34 * corrected_y / scale, color="#d7643b", linewidth=0.65, linestyle="--", zorder=2)
                    plotted_raw = True
            except Exception as error:
                coverage_row = {**coverage_row, "_plot_error": f"{type(error).__name__}: {error}"}

        if not plotted_raw:
            ax.plot([-25, 25], [y, y], color="#c9cdd2", linewidth=0.45, zorder=1)

        if is_computed:
            phi = number(measurement, "phi_from_N")
            phi_err = number(measurement, "phi_err")
            delay = number(measurement, "dt")
            delay_err = number(measurement, "dt_err")
            pol = number(measurement, "src_pol_numpy_from_N")
            if not np.isfinite(pol):
                pol = number(measurement, "src_pol_from_N")
            pol_radial_delta = number(measurement, "src_pol_deviation_from_radial_BAZ_deg")
            pol_eigen_ratio = number(measurement, "src_pol_eigenvalue_minor_major_ratio")
            pol_sigma = number(measurement, "src_pol_axis_stability_axial_sigma_deg")
            screen = measurement.get("interpretation_status", "")
            if screen == "near_null_geometry_excluded":
                row_color = "#b42318"
                status = "near-null: exclude"
            elif screen == "unresolved_high_uncertainty":
                row_color = "#a15c00"
                status = "unresolved"
            elif screen == "manual_cnn_review_required":
                row_color = "#175cd3"
                status = "review needed"
            else:
                row_color = "#667085"
                status = "not screened"
            phi_text = f"{phi:.1f} ± {phi_err:.1f}°"
            delay_text = f"{delay:.2f} ± {delay_err:.2f} s"
            pol_text = (
                f"{pol:.1f}/{pol_radial_delta:.1f}/{pol_eigen_ratio:.2f}/{pol_sigma:.1f}"
                if np.isfinite(pol_radial_delta) and np.isfinite(pol_eigen_ratio) and np.isfinite(pol_sigma)
                else f"{pol:.1f}/—/—/—"
            )
            pol_color = (
                "#b42318"
                if (np.isfinite(pol_radial_delta) and pol_radial_delta > 15.0)
                or (np.isfinite(pol_eigen_ratio) and pol_eigen_ratio > 0.10)
                else "#334155"
            )
            null_sep_text = f"{number(measurement, 'nearest_null_axis_separation_deg'):.1f}°"
        else:
            phi_text = delay_text = pol_text = "—"
            null_sep_text = "—"
            error = measurement.get("error", "")
            coverage_status = coverage_row.get("status", "")
            if coverage_status == "no_complete_XKS_window":
                status = "no 50s window"
            elif coverage_status == "no_raw_waveform":
                status = "no raw data"
            elif "Degrees of freedom" in error:
                status = "short window"
            elif "no measurement" in error.lower():
                status = "no cluster"
            else:
                status = "no estimate"
            row_color = "#7b8188"

        ax.text(27.2, y, pick_text, ha="left", va="center", fontsize=6.7, color="#334155")
        ax.text(39.5, y, phi_text, ha="left", va="center", fontsize=6.7, color=row_color)
        ax.text(52.5, y, delay_text, ha="left", va="center", fontsize=6.7, color="#334155")
        ax.text(66.2, y, pol_text, ha="left", va="center", fontsize=6.2, color=pol_color if is_computed else row_color)
        ax.text(77.0, y, null_sep_text, ha="left", va="center", fontsize=6.7, color="#334155")
        ax.text(85.0, y, status, ha="left", va="center", fontsize=6.2, color=row_color)

    ax.set_yticks([y_positions[station] for station in stations], stations, fontsize=7)
    ax.tick_params(axis="y", length=0, pad=4)
    ax.set_xlabel("Seconds relative to predicted iasp91 SKS pick (0 s)")
    ax.set_title(
        f"Single-event SKS splitting diagnostics — {event_id}\n"
        "All 1A stations; radial-transverse transverse component, before (blue) and after SWSPy correction (orange dashed)",
        fontsize=13,
        pad=10,
    )
    headers = [
        (27.2, "Pick UTC"),
        (39.5, "Fast axis φN ± σφ"),
        (52.5, "Delay δt ± σδt"),
        (66.2, "PCA° / Δrad° / λ / σax°"),
        (77.0, "BAZ to fast/slow (°)"),
        (85.0, "Paper-informed triage"),
    ]
    for x, label in headers:
        ax.text(x, count + 0.38, label, ha="left", va="bottom", fontsize=7, weight="bold", color="#172b3a")
    ax.text(
        -25,
        -0.65,
        "Blue shading: primary corrected E/N PCA window (−2 to +8 s); dotted marks show the other PCA-window boundaries, and yellow is ±1 s pick tolerance. "
        "Each row reports primary PCA angle/Δ radial/λmin-to-max/axial σ across the four windows (−2,+8; −1,+10; 0,+10; 0,+15 s). "
        "Δ radial is axial distance to WGS84 BAZ; λmin/max ≤0.10 is a near-linear screen. "
        "Near-null geometry (<15° from fast or slow polarization) is an event-geometry screen, not a Quality-N null label. "
        "Other finite EV estimates are unresolved; this is not a CNN A/B/C/N classification.",
        fontsize=7,
        color="#475569",
        ha="left",
        va="center",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_diagnostic_pdf(
    combined_png: Path,
    plots_dir: Path,
    output_pdf: Path,
    supplemental_pages: list[Path] | None = None,
) -> int:
    plot_files = sorted(plots_dir.glob("*.png"))
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output_pdf) as pdf:
        for path in [combined_png, *plot_files, *(supplemental_pages or [])]:
            if not path.exists():
                continue
            with Image.open(path) as source:
                source.thumbnail((1800, 1440), resample=Image.Resampling.LANCZOS)
                image = np.asarray(source.convert("RGB")).copy()
            fig = plt.figure(figsize=(12, 9), layout="constrained")
            ax = fig.add_subplot(1, 1, 1)
            ax.imshow(image)
            ax.axis("off")
            pdf.savefig(fig, dpi=150)
            plt.close(fig)
    return len(plot_files)


def make_polarization_figures(
    event_id: str,
    measurements: list[dict[str, str]],
    focused_dir: Path,
    output_dir: Path,
) -> list[Path]:
    """Plot particle-motion ellipses to show whether corrected motion is linear."""
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for row in measurements:
        if row.get("measurement_status") != "computed":
            continue
        station = row["station"]
        phase = row.get("phase", "SKS")
        trace_file = focused_dir / event_id / f"{station}_{phase}_50s.mseed"
        if not trace_file.exists():
            continue
        try:
            pick = UTCDateTime(row["arrival_utc"])
            baz = number(row, "back_azimuth_deg")
            raw = read(str(trace_file))
            corrected = remove_splitting(
                raw.copy(),
                phi=number(row, "phi_from_Q"),
                dt=number(row, "dt"),
                back_azi=baz,
                event_inclin_angle_at_station=0.0,
                return_BPA=False,
                return_FS=False,
            )
            traces = {}
            for label, stream in (("raw", raw), ("corrected", corrected)):
                east = stream.select(channel="HHE")
                north = stream.select(channel="HHN")
                if len(east) != 1 or len(north) != 1:
                    raise ValueError("expected exactly one HHE and one HHN component")
                if (
                    east[0].stats.npts != north[0].stats.npts
                    or abs(east[0].stats.starttime - north[0].stats.starttime) > 0.5 / east[0].stats.sampling_rate
                ):
                    raise ValueError("HHE/HHN component sample counts or start times differ")
                relative_times = east[0].times(reftime=pick)
                traces[label] = (
                    np.asarray(east[0].data, dtype=np.float64),
                    np.asarray(north[0].data, dtype=np.float64),
                    relative_times,
                    polarization_window_mask(relative_times, PRIMARY_POLARIZATION_WINDOW),
                )
            east_full, north_full, corrected_times, primary_mask = traces["corrected"]
            window_metrics = pca_metrics_by_window(east_full, north_full, corrected_times)
            primary_metrics = window_metrics[PRIMARY_POLARIZATION_WINDOW]
            east = east_full[primary_mask]
            north = north_full[primary_mask]
            times_primary = corrected_times[primary_mask]
            covariance = np.cov(np.vstack((east, north)))
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            major_index = int(np.argmax(eigenvalues))
            major_vector = eigenvectors[:, major_index]
            pca_deg = primary_metrics["angle_deg"]
            pca_delta = number(row, "src_pol_deviation_from_radial_BAZ_deg")
            ratio = primary_metrics["minor_major_ratio"]
            pca_sigma = axial_window_sigma_deg([metric["angle_deg"] for metric in window_metrics.values()])
            axis_length = float(np.nanpercentile(np.hypot(east, north), 95.0))
            if not np.isfinite(axis_length) or axis_length <= 0:
                raise ValueError("zero or non-finite corrected horizontal signal")
            radial = np.array([np.sin(np.deg2rad(baz)), np.cos(np.deg2rad(baz))])

            fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2), layout="constrained")
            for column, label in enumerate(("raw", "corrected")):
                e, n, times, motion_mask = traces[label]
                scale = axis_length
                axes[0, column].axvspan(-2, 8, color="#cfe5f5", alpha=0.42, zorder=0, label="Primary PCA window")
                for boundary in (-1, 0, 10, 15):
                    axes[0, column].axvline(boundary, color="#80a9c7", linewidth=0.55, linestyle=":", zorder=1)
                axes[0, column].plot(times, n / scale, color="#1b75a8", linewidth=0.9, label="N")
                axes[0, column].plot(times, e / scale, color="#d77926", linewidth=0.9, label="E")
                axes[0, column].axvline(0, color="#a82828", linestyle="--", linewidth=0.8)
                axes[0, column].axvspan(-1, 1, color="#ffeb99", alpha=0.25)
                axes[0, column].set_title(f"{label.title()} N/E traces")
                axes[0, column].set_xlabel("Seconds from theoretical XKS arrival")
                axes[0, column].set_ylabel("Amplitude / corrected 95th-percentile scale")
                axes[0, column].grid(alpha=0.22)
                axes[0, column].legend(loc="upper right", fontsize=8, ncol=2)

                particle = axes[1, column]
                particle.scatter(e[motion_mask] / scale, n[motion_mask] / scale, c=times[motion_mask], cmap="viridis", s=8, alpha=0.68, linewidths=0)
                particle.plot(
                    [-axis_length * radial[0] / scale, axis_length * radial[0] / scale],
                    [-axis_length * radial[1] / scale, axis_length * radial[1] / scale],
                    color="#d77926", linestyle="--", linewidth=1.6, label="Expected radial BAZ",
                )
                if label == "corrected":
                    particle.plot(
                        [-axis_length * major_vector[0] / scale, axis_length * major_vector[0] / scale],
                        [-axis_length * major_vector[1] / scale, axis_length * major_vector[1] / scale],
                        color="#1b75a8", linewidth=2.0, label="NumPy PCA major axis",
                    )
                    particle.set_title(
                        f"Corrected motion (−2 to +8 s) — PCA {pca_deg:.1f}°; "
                        f"Δradial {pca_delta:.1f}°; λmin/λmax {ratio:.3f}; σax {pca_sigma:.1f}°"
                    )
                else:
                    particle.set_title("Original motion (−2 to +8 s)")
                particle.set_xlabel("East (normalized)")
                particle.set_ylabel("North (normalized)")
                particle.set_aspect("equal", adjustable="datalim")
                particle.grid(alpha=0.22)
                particle.legend(loc="best", fontsize=7)

            near_linear = ratio <= 0.10
            status = "near-linear screen (variance ratio ≤ 0.10)" if near_linear else "elliptical / low-linearity screen"
            fig.suptitle(
                f"{station} — SWSPy single-event polarization check | {status}\n"
                "PCA axis is axial and clockwise from north; orange is WGS84 event direction; blue marks the primary PCA window",
                fontsize=12,
            )
            window_angle_text = "; ".join(
                f"{start:+g} to {end:+g} s: {metrics['angle_deg']:.1f}°"
                for name, start, end in POLARIZATION_WINDOWS_S
                for metrics in [window_metrics[name]]
            )
            fig.text(
                0.5, 0.005,
                f"PCA axes by window: {window_angle_text}. Axial circular σ across windows = {pca_sigma:.1f}°.",
                ha="center", fontsize=8, color="#475569",
            )
            output = output_dir / f"{station}_{phase}_particle_motion.png"
            fig.savefig(output, dpi=180, bbox_inches="tight")
            plt.close(fig)
            outputs.append(output)
        except Exception as error:
            print(f"{station}: could not create polarization figure: {type(error).__name__}: {error}")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-id", default="us10006exl")
    parser.add_argument("--measurement-dir", type=Path, default=Path("results/sws_1a_10h"))
    parser.add_argument("--focused-dir", type=Path, default=Path("data/focused_xks_1a"))
    parser.add_argument("--station-csv", type=Path, default=Path("results/network_1A_stations.csv"))
    parser.add_argument("--output")
    args = parser.parse_args()

    output = Path(args.output) if args.output else args.measurement_dir / f"1A_{args.event_id}_combined_sws_diagnostics.png"
    measurement_rows = read_rows(args.measurement_dir / f"1A_{args.event_id}_sws_measurements.csv")
    coverage_rows = read_rows(args.measurement_dir / f"1A_{args.event_id}_processing_coverage.csv")
    station_rows = read_rows(args.station_csv)
    quality_path = args.measurement_dir / f"1A_{args.event_id}_paper_quality_screen.csv"
    quality_rows = read_rows(quality_path) if quality_path.exists() else measurement_rows
    make_combined_figure(args.event_id, quality_rows, coverage_rows, station_rows, args.focused_dir, output)
    map_path = args.measurement_dir / f"1A_{args.event_id}_sws_station_map.png"
    particle_plots = make_polarization_figures(
        args.event_id, measurement_rows, args.focused_dir, args.measurement_dir / "particle_motion"
    )
    pdf_path = output.with_suffix(".pdf")
    supplemental = ([map_path] if map_path.exists() else []) + particle_plots
    individual_count = make_diagnostic_pdf(
        output, args.measurement_dir / "plots", pdf_path, supplemental_pages=supplemental
    )
    print(f"Combined all-station pick/SWS figure: {output}")
    print(
        f"Combined event diagnostics PDF: {pdf_path} ({individual_count} SWSPy tutorial plots, "
        f"{len(particle_plots)} NumPy polarization pages, summary and map)"
    )


if __name__ == "__main__":
    main()
