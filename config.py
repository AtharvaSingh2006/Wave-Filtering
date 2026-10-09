"""Shared configuration for the Sri Lanka waveform workflow."""

ACTIVE_START = "2016-05-25T00:00:00"
ACTIVE_END = "2017-07-05T00:00:00"  # exclusive; includes 2017-07-04

SAMPLE_START = "2016-05-01T00:00:00"
SAMPLE_END = "2017-08-01T00:00:00"

MIN_MAGNITUDE = 6.3
MIN_DISTANCE_DEG = 85.0
MAX_DISTANCE_DEG = 140.0
XKS_PHASES = ("SKS", "SKKS", "PKS")
TAUP_MODEL = "iasp91"

# Verified FDSN codes and channels in the target period.
# PALK horizontals are BH1/BH2; use StationXML orientation before SWS.
STATIONS = [
    {"network": "II", "station": "PALK", "location": "00", "channel": "BH?"},
    {"network": "GE", "station": "MALK", "location": "*", "channel": "BH?"},
    {"network": "GE", "station": "HALK", "location": "*", "channel": "BH?"},
]

STATION_COORDINATES = {
    ("II", "PALK"): (7.2728, 80.7022),
    ("GE", "MALK"): (8.39682, 80.54254),
    ("GE", "HALK"): (6.0877, 80.6806),
}

CUT_SECONDS = 2 * 60 * 60
PRE_EVENT_SECONDS = 10 * 60
FILTER_MIN_HZ = 0.01
FILTER_MAX_HZ = 0.5

SUPPORTED_SUFFIXES = {".mseed", ".msd", ".miniseed", ".sac", ".seed"}
