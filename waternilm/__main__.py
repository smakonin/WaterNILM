"""Command-line entry point: python -m waternilm --help."""

import argparse
import json
from pathlib import Path
import sys

from .data import load_series, prepare_ampds, write_rows
from .model import Model, activity_ranges, capped_viterbi, train


def parser():
    root = argparse.ArgumentParser(description="Research water disaggregation from appliance electrical states")
    commands = root.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Align AMPds meter files and apply explicit current boundaries")
    prepare.add_argument("--electricity", required=True)
    prepare.add_argument("--water", required=True)
    prepare.add_argument("--labels", help="Optional DWW.csv evaluation labels; never used for training")
    prepare.add_argument("--boundaries", required=True, type=float, nargs="+", help="Positive current thresholds in amperes")
    prepare.add_argument("--start", type=int, help="Inclusive Unix timestamp")
    prepare.add_argument("--end", type=int, help="Exclusive Unix timestamp")
    prepare.add_argument("--output", required=True)
    fit = commands.add_parser("train", help="Fit a model from aggregate water and electrical states")
    fit.add_argument("input")
    fit.add_argument("--model", required=True)
    fit.add_argument("--num-observed", required=True, type=int, help="Vocabulary size, including off state zero")
    fit.add_argument("--order", type=int, choices=range(1, 5), default=2)
    fit.add_argument("--water-step", type=float, default=0.5, help="Water-state increment in L/min (default 0.5)")
    fit.add_argument("--max-gap", type=int, default=0, help="Bridge up to this many inactive minutes")
    predict = commands.add_parser("predict", help="Decode held-out data and optionally report evaluation metrics")
    predict.add_argument("input")
    predict.add_argument("--model", required=True)
    predict.add_argument("--output", required=True)
    predict.add_argument("--decoder", choices=("exact", "legacy"), default="exact")
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        # Protect input files from accidental output-path collisions.
        destination = Path(args.model if args.command == "train" else args.output).resolve()
        sources = ([args.electricity, args.water, args.labels] if args.command == "prepare"
                   else [args.input] + ([args.model] if args.command == "predict" else []))
        if destination in [Path(p).resolve() for p in sources if p]:
            raise ValueError("Output path must differ from every input path")
        if destination.exists():
            raise ValueError(f"Output already exists: {destination}; choose a new path")
        if args.command == "prepare":
            rows = prepare_ampds(args.electricity, args.water, boundaries=args.boundaries,
                                 start=args.start, end=args.end, labels_path=args.labels)
            write_rows(destination, rows)
            print(json.dumps({"rows": len(rows), "output": str(destination)}))
            return 0
        series = load_series(args.input)
        if args.command == "train":
            intervals = activity_ranges(series.active, args.max_gap)
            model = train(series.caps(args.water_step), series.observed, intervals,
                          num_observed=args.num_observed, order=args.order, water_step=args.water_step)
            document = model.to_dict()
            document["max_gap"] = args.max_gap
            document["training_range"] = [series.timestamps[0], series.timestamps[-1]]
            destination.write_text(json.dumps(document, indent=2) + "\n")
            print(json.dumps({"activities": len(intervals), "model": str(destination)}))
            return 0
        document = json.loads(Path(args.model).read_text())
        model = Model.from_dict(document)
        training_range = document.get("training_range")
        if training_range and series.timestamps[0] <= training_range[1] and series.timestamps[-1] >= training_range[0]:
            raise ValueError("Prediction timestamps overlap the training range; use held-out data")
        caps = series.caps(model.water_step)
        intervals = activity_ranges(series.active, document.get("max_gap", 0))
        prediction = [0.0] * len(caps)
        for start, end in intervals:
            prediction[start:end] = [state * model.water_step for state in capped_viterbi(
                series.observed[start:end], caps[start:end], model, mode=args.decoder)]
        rows = [{"timestamp": t, "predicted_lpm": p, "water_lpm": w}
                for t, p, w in zip(series.timestamps, prediction, series.water)]
        report = {"decoder": args.decoder, "rows": len(rows), "activities": len(intervals)}
        if series.truth is not None:
            for row, truth in zip(rows, series.truth):
                row["truth_lpm"] = truth
            # Evaluate all input minutes and activity minutes separately to expose
            # the effect of long inactive/zero-water periods on aggregate metrics.
            for name, indices in (("all_minutes", range(len(rows))),
                                  ("activity_minutes", [i for a, b in intervals for i in range(a, b)])):
                errors = [prediction[i] - series.truth[i] for i in indices]
                if errors:
                    mse = sum(e*e for e in errors) / len(errors)
                    report[name] = {"n": len(errors), "mae_lpm": sum(abs(e) for e in errors)/len(errors),
                                    "mse_lpm2": mse, "rmse_lpm": mse**0.5}
                else:
                    report[name] = {"n": 0}
        write_rows(destination, rows)
        print(json.dumps(report, indent=2))
        return 0
    except (ValueError, OSError, KeyError, TypeError, OverflowError) as exc:
        print(f"waternilm: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
