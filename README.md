# Wave Filtering: Sri Lanka Teleseismic Waveforms

## Project summary

This project builds a reproducible Python workflow for screening teleseismic earthquakes, downloading and processing three-component seismic waveforms, and reviewing core-phase records from Sri Lankan stations. It uses ObsPy for event and waveform handling, response correction, component rotation, travel-time predictions, and filtering; NumPy and Matplotlib support the analysis and figures.

The main station set is PALK, MALK, and HALK. The workflow also includes a separate example using the temporary GEOFON 1A Sri Lanka network to explore XKS shear-wave splitting. Record-section plots provide event and phase context; splitting measurements are exploratory and require waveform-level quality review.

## Stations and networks

The three permanent stations are requested using these verified network codes:

| Station | Network | Horizontal components in the target data |
| --- | --- | --- |
| PALK | II | BH1 and BH2; use StationXML orientation metadata to rotate to geographic axes |
| MALK | GE | BHE and BHN |
| HALK | GE | BHE and BHN |

GEOFON network 1A is a different, temporary Sri Lanka deployment with SL station codes. It is used by the separate 10-hour XKS example and should not be confused with II.PALK, GE.MALK, or GE.HALK.

## Event screening

The active interval is 2016-05-25 through 2017-07-04 inclusive. The default screen queries the USGS earthquake catalog for magnitude 6.3 or greater, then uses iasp91 travel times and an 85°–140° distance range to identify SKS, SKKS, or PKS geometry candidates.

The current active-period screen contains 75 catalog events, 23 candidate XKS events, and 67 event–station geometry pairs. These are screening candidates, not confirmed arrivals or accepted shear-wave splitting results. Four of the 23 available two-hour event cuts currently lack PALK data; the remaining cuts contain the three target stations where data are available.

Refresh the candidate list with:

    .\.venv\Scripts\python.exe screen_events.py

The candidate table and screen summary are saved under results.

## Main waveform workflow

### 1. Download two-hour event cuts

The downloader requests the screened station/event data, writes raw MiniSEED files to data/cuts_2h, saves response metadata as data/metadata/sri_lanka_response.xml, and records download coverage in results/waveform_download_log.csv. The default two-hour window begins 10 minutes before the event origin and ends 110 minutes after it.

    .\.venv\Scripts\python.exe download_waveforms.py

### 2. Filter response-corrected waveforms

For general event inspection, the filter script removes the instrument response when matching StationXML is available, rotates components using orientation metadata, and applies a 0.01–0.5 Hz bandpass. Processed files are kept separate from raw downloads in data/processed.

    .\.venv\Scripts\python.exe filter_waveforms.py --input-dir data\cuts_2h --stationxml data\metadata\sri_lanka_response.xml

### 3. Create all-event phase record sections

The all-event record-section workflow response-corrects to velocity, rotates to ZNE, and applies a zero-phase 0.01–0.05 Hz bandpass (20–100 s periods). It writes processed MiniSEED files to data/processed_20_100s and event plots and a station/phase manifest to results/record_sections_20_100s_all_events.

    .\.venv\Scripts\python.exe plot_all_event_record_sections.py

The full-event views show predicted phase context, including P-family phases, SKS, ScS where predicted, SKKS, S-family phases, and an approximate surface-wave interval. The focused views show SKS, SKKS, and available ScS windows. Phase markers are iasp91 predictions, not manually picked arrivals. Full-event traces are normalized for display.

### 4. Stack stations and events

Create one station stack figure for each event:

    .\.venv\Scripts\python.exe plot_all_station_phase_stacks.py

Combine the event and station stacks into one overview:

    .\.venv\Scripts\python.exe plot_grand_all_event_stack.py

Per-event stacks are saved in results/all_station_phase_stacks_20_100s. The combined SKS/ScS/SKKS overview is saved in results/grand_phase_stacks_20_100s/all_events_all_stations_SKSS_20_100s.png. These figures use station-level noise normalization, so displayed amplitudes are relative and are not physical velocity amplitudes. The grand stack gives each event equal weight after stacking its available stations. Differences in source radiation and back-azimuth can cause traces to cancel; inspect the individual event plots alongside the combined overview.

## Separate 1A XKS splitting example

The 1A example uses a 10-hour continuous recording around the M7.4 event us10006exl (2016-08-19 07:32 UTC). It is separate from the three-station II/GE workflow.

To download a 10-hour recording, provide its UTC start time:

    .\.venv\Scripts\python.exe download_1a_waveforms.py --start 2016-08-19T00:00:00

Process the current example event and run the SWSPy eigenvalue method:

    .\.venv\Scripts\python.exe process_1a_xks.py --event-id us10006exl

The pipeline response-corrects waveforms to velocity, applies a 0.01–0.5 Hz bandpass, resamples to 20 Hz, predicts XKS arrivals with iasp91, and extracts phase windows. Processed continuous traces, focused windows, measurements, coverage tables, and diagnostic plots are stored separately under data/processed_1a_10h, data/focused_xks_1a, and results/sws_1a_10h.

For the combined diagnostic figure and preliminary quality screen:

    .\.venv\Scripts\python.exe plot_sws_event_diagnostics.py --event-id us10006exl
    .\.venv\Scripts\python.exe classify_sws_quality.py --event-id us10006exl
    .\.venv\Scripts\python.exe plot_sws_station_map.py --event-id us10006exl

The event example currently has 28 complete XKS windows and 25 finite SWSPy estimates. The preliminary screen flags 7 near-null geometry candidates, 18 high-uncertainty estimates, 3 windows without a solution, 1 incomplete phase window, and 1 missing waveform. No station currently passes the preliminary acceptance screen. These results should not be presented as validated anisotropy or reliable fast-axis directions.

The optional CNN output is supporting information only. Its scores are not calibrated for the current response-corrected SI-amplitude inputs, and this event has no hand-labelled examples for an accuracy estimate. A zero score is not, by itself, evidence that a waveform is unusable.

## Setup

The repository includes a requirements.txt for a separate project environment. From PowerShell in the project folder:

    uv venv .venv
    uv pip install --python .venv\Scripts\python.exe -r requirements.txt

Run scripts using the environment’s Python executable:

    .\.venv\Scripts\python.exe script_name.py

If the environment was created without pip, use uv as shown above to install the requirements.

## Output layout

- data/cuts_2h — raw two-hour event waveform cuts
- data/metadata — StationXML response and orientation metadata
- data/processed — general response-corrected and filtered cuts
- data/processed_20_100s — velocity waveforms for 20–100 s record-section analysis
- data/sl_1a_continuous — separate GEOFON 1A continuous downloads
- data/processed_1a_10h and data/focused_xks_1a — processed 1A data and XKS windows
- results — event lists, coverage manifests, maps, record sections, stacks, and splitting diagnostics
- results/previous_outputs_20261005 — archived earlier plots and reports

Waveform downloads and generated products may be large; their presence depends on which pipeline steps have been run.

## Scientific notes

- Catalog magnitude and predicted distance/arrival screens identify candidates only. They do not establish waveform quality or a usable splitting measurement.
- Predicted phase times depend on the iasp91 model and should be checked against the records.
- The 20–100 s plots are normalized for display or stack balance; use response-corrected data and the manifests for quantitative analysis.
- Splitting parameters need clear XKS arrivals, stable window choices, constrained delay and fast-axis estimates, and independent quality control. Particle-motion linearity is one useful diagnostic, but does not by itself validate splitting.
- CNN probabilities are not treated as definitive quality labels until calibrated on comparable data with suitable labels.

## References

- Zhang & Gao (2022), [CNN quality control for shear-wave splitting measurements](https://doi.org/10.1029/2021GL097101).
- Liu & Gao (2013), [A review of shear-wave splitting analyses](https://doi.org/10.1785/0120120355).
- [SWSPy single-event tutorial](https://swspy.readthedocs.io/en/latest/tutorials/single_event_example/single_event_example.html).
- [CNN-SWS source repository](https://github.com/YW-Zhang94/CNN-SWS).
- [ObsPy basic processing exercise](https://krischer.github.io/seismo_live_build/html/ObsPy/07_Basic_Processing_Exercise_solution_wrapper.html).

