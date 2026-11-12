"""Command line entry point: ``python -m ssio_eval.cli <command>``.

Commands
--------
eval      evaluate one estimate against one ground-truth file
batch     evaluate a whole config (this is how you fill in the paper table)
synth     generate a synthetic pair with a known answer
selftest  run the built-in numeric checks
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import metrics
from .io.datasets import load_trajectory
from .traj import Trajectory


# --------------------------------------------------------------------- output
def _fmt(value) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _print_block(title: str, d: dict) -> None:
    print(f"\n== {title} ==")
    keys = [
        "mean",
        "median",
        "rmse",
        "std",
        "p95",
        "max",
        "align",
        "frame",
        "scale",
        "yaw_deg",
        "n",
        "duration_s",
        "path_length_m",
        "rmse_pct_of_path",
        "gt_speed_mean",
        "est_speed_mean",
        "rot_rmse_deg",
        "trans_error_per_meter",
        "n_segments",
    ]
    for k in keys:
        if k in d and d[k] is not None:
            print(f"  {k:>20s}: {_fmt(d[k])}")


def _plot(est: Trajectory, gt: Trajectory, ate_rmse: float, out: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    e, g = metrics.associate(est, gt)
    R, t, s, _ = metrics.estimate_alignment(e.p, g.p, mode="se3")
    p_al = s * (e.p @ R.T) + t
    err = np.linalg.norm(p_al - g.p, axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(g.p[:, 0], g.p[:, 1], "k-", lw=1.2, label="gt")
    axes[0].plot(e.p[:, 0], e.p[:, 1], "r-", lw=0.9, alpha=0.8, label="est (raw)")
    axes[0].plot(p_al[:, 0], p_al[:, 1], "b--", lw=0.9, alpha=0.8, label="est (aligned)")
    axes[0].set_aspect("equal")
    axes[0].set_xlabel("x [m]")
    axes[0].set_ylabel("y [m]")
    axes[0].legend()
    axes[0].set_title(title)

    axes[1].plot(g.t - g.t[0], err)
    axes[1].axhline(ate_rmse, color="r", ls="--", lw=1, label=f"RMSE {ate_rmse:.3f} m")
    axes[1].set_xlabel("t [s]")
    axes[1].set_ylabel("position error [m]")
    axes[1].legend()
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)


# ------------------------------------------------------------------- commands
def cmd_eval(args: argparse.Namespace) -> int:
    est = load_trajectory(args.est, time_unit=args.time_unit, quat_order=args.quat_order)
    gt = load_trajectory(args.gt, time_unit="auto", quat_order=args.gt_quat_order)

    print(f"est: {est.summary()}")
    print(f"gt : {gt.summary()}")

    result = metrics.evaluate(
        est,
        gt,
        align=args.align,
        with_scale=args.scale,
        ave_frame=args.ave_frame,
        ave_align=args.ave_align,
        rpe_delta=args.rpe_delta,
        include_rpe=args.rpe,
    )
    _print_block("ATE (position)", result["ate"])
    _print_block(f"AVE (velocity, frame={args.ave_frame})", result["ave"])
    _print_block("AVE (complementary view)", result["ave_alt"])
    if "rpe" in result:
        _print_block("RPE", result["rpe"])

    if args.offset_scan:
        scan = metrics.best_time_offset(est, gt, search=args.offset_scan, mode=args.align)
        print(f"\n== time offset scan (diagnostic) ==\n  {scan}")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "est": str(args.est),
            "gt": str(args.gt),
            "est_summary": est.summary(),
            "gt_summary": gt.summary(),
            "results": result,
            "args": {k: v for k, v in vars(args).items() if not callable(v)},
        }
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\n[json] {out}")

    if args.plot:
        _plot(est, gt, result["ate"]["rmse"], Path(args.plot), title=Path(args.est).name)
        print(f"[plot] {args.plot}")
    return 0


def cmd_batch(args: argparse.Namespace) -> int:
    import csv

    import yaml

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    defaults = cfg.get("defaults", {})
    align = args.align or defaults.get("align", "se3")
    ave_frame = args.ave_frame or defaults.get("ave_frame", "world")
    rpe_delta = defaults.get("rpe_delta", 1.0)

    runs = list(cfg.get("runs") or [])
    if args.pred_root:
        runs = [dict(r, est=str(args.pred_root / f"{r.get('seq', r.get('name'))}.csv")) for r in runs]
    if not runs:
        print("config has no runs; nothing to do", file=sys.stderr)
        return 2

    rows = []
    pending = 0
    for r in runs:
        seq = r.get("seq") or r.get("name") or Path(r["gt"]).parent.name
        if not Path(r["est"]).exists():
            print(f"[todo] {seq:<28s} no prediction yet at {r['est']}")
            pending += 1
            continue
        try:
            est = load_trajectory(r["est"], time_unit=args.time_unit or "auto")
            gt = load_trajectory(r["gt"])
            res = metrics.evaluate(est, gt, align=align, ave_frame=ave_frame, rpe_delta=rpe_delta)
            rows.append(
                {
                    "dataset": r.get("dataset", "-"),
                    "seq": seq,
                    "ate_rmse": round(res["ate"]["rmse"], 4),
                    "ate_mean": round(res["ate"]["mean"], 4),
                    "ate_pct_path": round(res["ate"]["rmse_pct_of_path"], 3),
                    "ave_mean": round(res["ave"]["mean"], 4),
                    "ave_median": round(res["ave"]["median"], 4),
                    "ave_p95": round(res["ave"]["p95"], 4),
                    "n": res["ave"]["n"],
                    "duration_s": round(res["ave"]["duration_s"], 2),
                }
            )
            print(f"[ok]   {seq:<28s} ATE {rows[-1]['ate_rmse']:.3f} m  AVE {rows[-1]['ave_mean']:.3f} m/s")
        except Exception as exc:  # noqa: BLE001 - batch must not die on one sequence
            print(f"[fail] {seq:<28s} {exc}", file=sys.stderr)
            rows.append({"dataset": r.get("dataset", "-"), "seq": seq, "error": str(exc)[:200]})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for row in rows for k in row})
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[csv] {out}  ({len(rows)} rows)")

    ok = [r for r in rows if "error" not in r]
    if ok:
        print(f"mean AVE over {len(ok)} sequences: {np.mean([r['ave_mean'] for r in ok]):.4f} m/s")
    if pending:
        print(f"{pending} sequence(s) awaiting predictions (not an error)")
    return 0 if len(ok) == len(rows) else 1


def cmd_synth(args: argparse.Namespace) -> int:
    from .synth import make_pair

    gt_path, est_path = make_pair(
        args.out,
        seed=args.seed,
        vel_bias_world=tuple(args.vel_bias),
        pos_noise=args.pos_noise,
        yaw_offset_deg=args.yaw_offset,
        scale=args.scale_est,
    )
    print(f"[gt ] {gt_path}")
    print(f"[est] {est_path}")
    print("\nNow run:\n  python -m ssio_eval.cli eval --est "
          f"{est_path} --gt {gt_path} --align se3")
    print(f"Expected AVE ~= |vel_bias| = {np.linalg.norm(args.vel_bias):.4f} m/s")
    print(f"Expected ATE ~= drift over {args.duration:.0f} s")
    return 0


def cmd_selftest(args: argparse.Namespace) -> int:
    from .synth import figure_eight

    ok = True
    gt = figure_eight(duration=30.0)
    res = metrics.evaluate(gt, gt, align="none")
    print(f"identity ATE  {res['ate']['rmse']:.3e} m   (expect ~0)")
    print(f"identity AVE  {res['ave']['mean']:.3e} m/s (expect ~0)")
    ok &= res["ate"]["rmse"] < 1e-9 and res["ave"]["mean"] < 1e-9

    bias = np.array([0.05, 0.02, 0.0])
    from .synth import corrupt

    est = corrupt(gt, vel_bias_world=bias, pos_noise=0.0)
    r = metrics.ave(est, gt, frame="world", mode="none")
    print(f"bias AVE      {r['mean']:.4f} m/s (expect {np.linalg.norm(bias):.4f})")
    ok &= abs(r["mean"] - np.linalg.norm(bias)) < 1e-3

    from . import frames

    Rz = frames.rotz(np.deg2rad(37.0))
    est2 = gt.transform(Rz, np.array([4.0, -3.0, 0.5]))
    r2 = metrics.ate(est2, gt, mode="se3")
    print(f"rigid-align ATE {r2['rmse']:.3e} m (expect ~0)")
    ok &= r2["rmse"] < 1e-9

    print("\nSELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


# ---------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ssio_eval", description="Odometry metrics for the SSIO reproduction")
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--align", choices=["none", "yaw", "se3"], default="se3")
        sp.add_argument("--scale", action="store_true", help="estimate a scale factor (Sim3)")
        sp.add_argument("--ave-frame", choices=["world", "body"], default="world")
        sp.add_argument(
            "--ave-align",
            choices=["none", "yaw", "se3"],
            default=None,
            help="alignment used for world-frame AVE (default: yaw)",
        )

    e = sub.add_parser("eval", help="evaluate one estimate")
    e.add_argument("--est", required=True)
    e.add_argument("--gt", required=True)
    e.add_argument("--time-unit", default="auto", choices=["auto", "s", "ms", "us", "ns"])
    e.add_argument("--quat-order", default="wxyz", choices=["wxyz", "xyzw"])
    e.add_argument("--gt-quat-order", default="wxyz", choices=["wxyz", "xyzw"])
    e.add_argument("--rpe", action="store_true")
    e.add_argument("--rpe-delta", type=float, default=1.0)
    e.add_argument("--offset-scan", type=float, default=0.0, help="scan +-X seconds (diagnostic)")
    e.add_argument("--out", help="write a JSON report here")
    e.add_argument("--plot", help="write a PNG here")
    common(e)
    e.set_defaults(func=cmd_eval)

    b = sub.add_parser("batch", help="evaluate a config of runs")
    b.add_argument("--config", required=True)
    b.add_argument("--out", default="runs/batch_results.csv")
    b.add_argument("--pred-root", help="directory holding <seq>.csv predictions")
    b.add_argument("--time-unit", default="auto", choices=["auto", "s", "ms", "us", "ns"])
    b.add_argument("--align", choices=["none", "yaw", "se3"])
    b.add_argument("--ave-frame", choices=["world", "body"])
    b.set_defaults(func=cmd_batch)

    s = sub.add_parser("synth", help="generate a synthetic pair with a known answer")
    s.add_argument("--out", default="runs/synth")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--duration", type=float, default=90.0)
    s.add_argument("--vel-bias", type=float, nargs=3, default=[0.05, 0.02, 0.0])
    s.add_argument("--pos-noise", type=float, default=0.01)
    s.add_argument("--yaw-offset", type=float, default=0.0)
    s.add_argument("--scale-est", type=float, default=1.0)
    s.set_defaults(func=cmd_synth)

    t = sub.add_parser("selftest", help="built-in numeric checks")
    t.set_defaults(func=cmd_selftest)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "offset_scan", 0.0) < 0:
        raise SystemExit("--offset-scan must be positive")
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())