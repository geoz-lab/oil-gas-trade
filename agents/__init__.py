"""
Load all numbered agent modules by file path so Python can import them
despite the numeric prefix in their filenames.

Each module is registered in sys.modules under both "agents.<alias>" and
"<alias>" so that unittest.mock.patch("agents.<alias>.<name>") works correctly.
"""
import importlib.util
import sys
from pathlib import Path

_DIR = Path(__file__).parent


def _load(alias: str, filename: str):
    full_name = f"agents.{alias}"
    path = _DIR / filename
    spec = importlib.util.spec_from_file_location(full_name, path)
    mod = importlib.util.module_from_spec(spec)   # type: ignore[arg-type]
    mod.__name__ = full_name
    sys.modules[full_name] = mod
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)                   # type: ignore[union-attr]
    return mod


task_planner         = _load("task_planner",         "01_task_planner.py")
baseline_forecast    = _load("baseline_forecast",    "02_baseline_forecast.py")
signal_retrieval     = _load("signal_retrieval",     "03_signal_retrieval.py")
signal_judge         = _load("signal_judge",         "03b_signal_judge.py")
event_detection      = _load("event_detection",      "04_event_detection.py")
evidence_structuring = _load("evidence_structuring", "05_evidence_structuring.py")
temporal_reasoning   = _load("temporal_reasoning",   "06_temporal_reasoning.py")
debate_system        = _load("debate_system",        "07_debate_system.py")
risk_calibration     = _load("risk_calibration",     "08_risk_calibration.py")
forecast_adjustment  = _load("forecast_adjustment",  "09_forecast_adjustment.py")
report_generator     = _load("report_generator",     "10_report_generator.py")

__all__ = [
    "task_planner", "baseline_forecast", "signal_retrieval", "signal_judge",
    "event_detection", "evidence_structuring", "temporal_reasoning",
    "debate_system", "risk_calibration", "forecast_adjustment",
    "report_generator",
]
