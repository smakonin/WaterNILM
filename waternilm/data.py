"""Explicit one-minute CSV contract and timestamp-based AMPds preparation."""

import csv
from dataclasses import dataclass
import math
from bisect import bisect_right


@dataclass
class Series:
    timestamps: list[int]
    observed: list[int]
    water: list[float]
    active: list[bool]
    truth: list[float] | None = None

    def caps(self, step):
        """Validate the pulse grid instead of silently truncating water use."""
        if not math.isfinite(step) or step <= 0:
            raise ValueError("Water step must be finite and positive")
        result = []
        for value in self.water:
            state = round(value / step)
            if not math.isclose(state * step, value, rel_tol=0, abs_tol=1e-7):
                raise ValueError(f"Water value {value} is not a multiple of water step {step}")
            result.append(state)
        return result


def number(value, name):
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {name}: {value!r}") from exc
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def integer(value, name):
    value = number(value, name)
    if value != int(value):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def read_csv(path):
    """Read normalized lowercase headers; preserve row ordering."""
    with open(path, newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError(f"{path}: CSV has no header")
        fields = [f.strip().lower() for f in reader.fieldnames]
        if len(set(fields)) != len(fields):
            raise ValueError(f"{path}: duplicate column names")
        reader.fieldnames = fields
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path}: CSV has no data")
    if any(None in r or any(v is None for v in r.values()) for r in rows):
        raise ValueError(f"{path}: inconsistent CSV row width")
    return fields, rows


def timestamps(rows):
    key = next((key for key in ("timestamp", "unix_ts") if key in rows[0]), None)
    if key is None:
        raise ValueError("CSV requires timestamp or unix_ts")
    result = [integer(row[key], "timestamp") for row in rows]
    if any(b - a != 60 for a, b in zip(result, result[1:])):
        raise ValueError("Timestamps must be increasing, unique and exactly 60 seconds apart")
    return result


def load_series(path):
    fields, rows = read_csv(path)
    required = {"electrical_state", "water_lpm", "active"}
    if not required <= set(fields):
        raise ValueError(f"Missing columns: {', '.join(sorted(required - set(fields)))}")
    times = timestamps(rows)
    observed = [integer(r["electrical_state"], "electrical_state") for r in rows]
    water = [number(r["water_lpm"], "water_lpm") for r in rows]
    active = [integer(r["active"], "active") for r in rows]
    if any(x not in (0, 1) for x in active):
        raise ValueError("active must be 0 or 1")
    if any(state and not on for state, on in zip(observed, active)):
        raise ValueError("Inactive samples must have electrical_state 0")
    truth = ([number(r["truth_lpm"], "truth_lpm") for r in rows]
             if "truth_lpm" in fields else None)
    return Series(times, observed, water, [bool(x) for x in active], truth)


def prepare_ampds(electricity_path, water_path, *, boundaries, start=None, end=None, labels_path=None):
    """Join named AMPds columns by exact timestamp, then quantize current.

    Boundaries are explicit positive current thresholds in amperes. Values
    equal to a boundary enter the higher state (np.digitize's default rule).
    Obtain these from a documented training-only PMF procedure; this adapter
    deliberately does not substitute another algorithm for Library_PMF.
    """
    boundaries = list(boundaries)
    if not boundaries or any(not math.isfinite(x) or x <= 0 for x in boundaries) or any(b <= a for a, b in zip(boundaries, boundaries[1:])):
        raise ValueError("Current boundaries must be positive, finite and strictly increasing")
    e_fields, electricity = read_csv(electricity_path)
    w_fields, water = read_csv(water_path)
    if not {"i", "s"} <= set(e_fields) or "avg_rate" not in w_fields:
        raise ValueError("AMPds inputs require electricity I and S, and water AVG_RATE columns")
    e_times, w_times = timestamps(electricity), timestamps(water)
    water_by_time = dict(zip(w_times, water))
    selected = [(t, r) for t, r in zip(e_times, electricity)
                if (start is None or t >= start) and (end is None or t < end)]
    if not selected:
        raise ValueError("Selected date range is empty")
    if any(t not in water_by_time for t, _ in selected):
        raise ValueError("Water readings missing at selected electricity timestamps")
    selected_water_times = [t for t in w_times
                            if selected[0][0] <= t <= selected[-1][0]]
    if selected_water_times != [t for t, _ in selected]:
        raise ValueError("Electricity and water timestamp grids differ")
    labels_by_time = None
    if labels_path:
        fields, labels = read_csv(labels_path)
        if "avg_rate" not in fields:
            raise ValueError("Labels require avg_rate")
        labels_by_time = dict(zip(timestamps(labels), labels))
        if any(t not in labels_by_time for t, _ in selected):
            raise ValueError("Labels missing at selected timestamps")
    output = []
    for timestamp, electricity_row in selected:
        state = bisect_right(boundaries, number(electricity_row["i"], "I"))
        active = int(number(electricity_row["s"], "S") != 0)
        if state and not active:
            raise ValueError("Quantized current is nonzero while apparent power S is zero")
        row = {"timestamp": timestamp, "electrical_state": state,
               "water_lpm": number(water_by_time[timestamp]["avg_rate"], "AVG_RATE"), "active": active}
        if labels_by_time is not None:
            row["truth_lpm"] = number(labels_by_time[timestamp]["avg_rate"], "label AVG_RATE")
        output.append(row)
    return output


def write_rows(path, rows):
    if not rows:
        raise ValueError("No output rows")
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
