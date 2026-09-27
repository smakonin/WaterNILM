"""Readable port of Untitled6.ipynb, cells 5, 7 and 8.

The training counts and unusual singleton smoothing follow the notebook.
The exact decoder keeps one path per full hidden history. ``legacy`` mode
instead keeps one history per current water state, as the notebook does.
Both modes use log probabilities and backpointers to avoid underflow and
repeated copying of entire paths. These are explicitly documented changes.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import product
import math
from numbers import Integral

History = tuple[int, ...]
Table = dict[History, dict]


def _states(values, name):
    values = list(values)
    if any(not isinstance(x, Integral) or x < 0 for x in values):
        raise ValueError(f"{name} must contain nonnegative integers")
    return [int(x) for x in values]


def activity_ranges(active, max_gap=0):
    """Return half-open activity intervals, joining gaps <= max_gap samples.

    Unlike the notebook's samples(), empty/all-off input returns an empty
    list. Activities touching either boundary are retained.
    """
    if not isinstance(max_gap, Integral) or max_gap < 0:
        raise ValueError("max_gap must be a nonnegative integer")
    indices = [i for i, value in enumerate(active) if value]
    if not indices:
        return []
    result = []
    start = previous = indices[0]
    for index in indices[1:]:
        if index - previous > max_gap + 1:
            result.append((start, previous + 1))
            start = index
        previous = index
    result.append((start, previous + 1))
    return result


@dataclass
class Model:
    order: int
    num_observed: int
    water_step: float
    transition: Table
    emission: Table

    def to_dict(self):
        """Portable JSON representation; no pickle or executable payloads."""
        return {
            "format": "waternilm-1", "order": self.order,
            "num_observed": self.num_observed, "water_step": self.water_step,
            "transition": [[list(h), [[j, p] for j, p in row.items()]]
                           for h, row in self.transition.items()],
            "emission": [[list(h), [[list(o), p] for o, p in row.items()]]
                         for h, row in self.emission.items()],
        }

    @classmethod
    def from_dict(cls, data):
        if data.get("format") != "waternilm-1":
            raise ValueError("Unsupported model format")
        order, num = data["order"], data["num_observed"]
        _configuration(order, num, data["water_step"])
        tables = []
        for name in ("transition", "emission"):
            table = {}
            for history, entries in data[name]:
                history = tuple(_states(history, "history"))
                if len(history) != order or history in table:
                    raise ValueError("Invalid or duplicate model history")
                row = {}
                for target, probability in entries:
                    if name == "emission":
                        target = tuple(_states(target, "emission"))
                        if len(target) != order or any(x >= num for x in target):
                            raise ValueError("Invalid emission history")
                    else:
                        target = _states([target], "transition")[0]
                    if target in row or not isinstance(probability, (int, float)) or not math.isfinite(probability) or not 0 < probability <= 1:
                        raise ValueError("Invalid or duplicate model probability")
                    row[target] = probability
                if not row or not math.isclose(sum(row.values()), 1.0, rel_tol=1e-9):
                    raise ValueError("Model probability rows must sum to one")
                table[history] = row
            tables.append(table)
        if not all(tables):
            raise ValueError("Empty model")
        return cls(order, num, data["water_step"], *tables)


def _configuration(order, num_observed, water_step):
    if not isinstance(order, Integral) or not 1 <= order <= 4:
        raise ValueError("order must be an integer from 1 through 4")
    if not isinstance(num_observed, Integral) or num_observed < 1:
        raise ValueError("num_observed must be a positive integer")
    if not math.isfinite(water_step) or water_step <= 0:
        raise ValueError("water_step must be finite and positive")
    if num_observed ** order > 1_000_000:
        raise ValueError("Observed-state vocabulary is too large for smoothing")


def train(hidden, observed, intervals, *, num_observed, order=2, water_step=0.5):
    """Fit from aggregate water states and appliance electrical states.

    ``hidden`` here is aggregate training water, NOT hand-labelled appliance
    water. Ground-truth appliance labels belong only in evaluation.
    Each interval starts with ``order`` dummy zero states, as in the notebook.
    """
    _configuration(order, num_observed, water_step)
    hidden = _states(hidden, "hidden")
    observed = _states(observed, "observed")
    if len(hidden) != len(observed) or not hidden:
        raise ValueError("Training sequences must be nonempty and aligned")
    if any(x >= num_observed for x in observed):
        raise ValueError("Observed state exceeds num_observed")
    intervals = list(intervals)
    if not intervals:
        raise ValueError("No training activities")
    previous_end = 0
    for start, end in intervals:
        if not isinstance(start, Integral) or not isinstance(end, Integral) or not previous_end <= start < end <= len(hidden):
            raise ValueError("Training intervals must be ordered, disjoint and in bounds")
        previous_end = end

    transition = defaultdict(Counter)
    emission = defaultdict(Counter)
    for start, end in intervals:
        water = [0] * order + hidden[start:end]
        electricity = [0] * order + observed[start:end]
        for t in range(order, len(water)):
            transition[tuple(water[t-order:t])][water[t]] += 1
            emission[tuple(water[t-order+1:t+1])][tuple(electricity[t-order+1:t+1])] += 1
            # Register histories needed to return to zero after water use.
            for length in range(1, order):
                history = tuple(water[t-length:t]) + (0,) * (order-length)
                transition[history]
                emission[history]

    for row in transition.values():
        if 0 not in row:
            row[0] = 1

    if len(emission) * num_observed ** order > 2_000_000:
        raise ValueError("Smoothing would exceed 2 million entries; reduce order or vocabulary")
    observation_histories = list(product(range(num_observed), repeat=order))
    for row in emission.values():
        rare = [o for o in observation_histories if row[o] <= 1]
        if rare:
            # Historical rule: replace counts of zero AND one by their mean;
            # if all are unseen, use one. This is not Laplace smoothing.
            replacement = sum(row[o] for o in rare) / len(rare) or 1.0
            for observations in rare:
                row[observations] = replacement

    def normalize(table):
        result = {}
        for history, row in table.items():
            total = sum(row.values())
            result[history] = {j: count / total for j, count in row.items()}
        return result

    return Model(order, num_observed, water_step, normalize(transition), normalize(emission))


def capped_viterbi(observed, caps, model, *, mode="exact"):
    """Infer water states <= caps, or raise if no feasible path exists.

    ``exact`` retains every reachable length-order history; ``legacy`` uses
    the notebook's pruning by current water state. Equal scores prefer the
    lexicographically larger predecessor, matching its tuple max convention.
    Complexity depends on the number of reachable histories and transitions;
    higher orders can still be expensive even with sparse tables.
    """
    observed = _states(observed, "observed")
    caps = _states(caps, "caps")
    if len(observed) != len(caps):
        raise ValueError("Observed states and caps must have equal lengths")
    if any(x >= model.num_observed for x in observed):
        raise ValueError("Observed state exceeds model vocabulary")
    if mode not in ("exact", "legacy"):
        raise ValueError("mode must be exact or legacy")
    if not observed:
        return []
    order = model.order
    padded = [0] * order + observed
    scores = {(0,) * order: 0.0}
    backpointers = []
    for t, cap in enumerate(caps, start=order):
        observation = tuple(padded[t-order+1:t+1])
        candidates = {}
        for predecessor, score in scores.items():
            for state, transition_probability in model.transition.get(predecessor, {}).items():
                if state > cap or transition_probability <= 0:
                    continue
                history = predecessor[1:] + (state,)
                emission_probability = model.emission.get(history, {}).get(observation, 0)
                if emission_probability <= 0:
                    continue
                candidate = (score + math.log(transition_probability) + math.log(emission_probability), predecessor, history)
                key = history if mode == "exact" else state
                if key not in candidates or candidate > candidates[key]:
                    candidates[key] = candidate
        if not candidates:
            raise ValueError(f"No feasible water-state path at sample {t-order}; check training coverage and caps")
        scores = {history: score for score, _, history in candidates.values()}
        backpointers.append({history: previous for _, previous, history in candidates.values()})
    history = max(scores, key=lambda h: (scores[h], h))
    result = []
    for pointers in reversed(backpointers):
        result.append(history[-1])
        history = pointers[history]
    return result[::-1]
