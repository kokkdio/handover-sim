import argparse
import csv
import html
from pathlib import Path

import numpy as np

from handover.gym_compat import gym
from handover.config import get_cfg
from handover.benchmark_wrapper import HandoverBenchmarkWrapper


def _as_numpy(value):
    if value is None:
        return np.full(3, np.nan, dtype=np.float32)
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=np.float32)


def _get_human_hand_pos(env):
    mano = env.env.mano
    if hasattr(mano, "_pose") and hasattr(mano, "_frame"):
        return _as_numpy(mano._pose[mano._frame, 0:3])
    return np.full(3, np.nan, dtype=np.float32)


def _constant_velocity_predict(history, horizon, dt):
    history = np.asarray(history, dtype=np.float32)
    if len(history) == 0 or not np.all(np.isfinite(history[-1])):
        return np.full((horizon, 3), np.nan, dtype=np.float32)
    if len(history) < 2:
        return np.repeat(history[-1][None], horizon, axis=0)

    valid = history[np.all(np.isfinite(history), axis=1)]
    if len(valid) < 2:
        return np.repeat(history[-1][None], horizon, axis=0)

    velocity = (valid[-1] - valid[0]) / max((len(valid) - 1) * dt, 1e-6)
    future_times = np.arange(1, horizon + 1, dtype=np.float32)[:, None] * dt
    return valid[-1][None] + future_times * velocity[None]


def _write_csv(path, records):
    columns = [
        "frame",
        "human_x",
        "human_y",
        "human_z",
        "prediction_step",
        "predicted_x",
        "predicted_y",
        "predicted_z",
    ]
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for record in records:
            frame = record["frame"]
            human = record["human"]
            for i, point in enumerate(record["prediction"], start=1):
                writer.writerow([frame] + human.tolist() + [i] + point.tolist())


def _project(points, x_index, y_index, width, height, padding, bounds):
    x_min, x_max, y_min, y_max = bounds
    x = points[:, x_index]
    y = points[:, y_index]
    sx = padding + (x - x_min) / max(x_max - x_min, 1e-6) * (width - 2 * padding)
    sy = height - padding - (y - y_min) / max(y_max - y_min, 1e-6) * (height - 2 * padding)
    return np.stack([sx, sy], axis=1)


def _polyline(points, color, width=2, dash=None):
    valid = np.all(np.isfinite(points), axis=1)
    if np.count_nonzero(valid) < 2:
        return ""
    pairs = " ".join(f"{x:.1f},{y:.1f}" for x, y in points[valid])
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<polyline points="{pairs}" fill="none" stroke="{color}" '
        f'stroke-width="{width}"{dash_attr}/>'
    )


def _circle(point, color, radius=4):
    if not np.all(np.isfinite(point)):
        return ""
    return f'<circle cx="{point[0]:.1f}" cy="{point[1]:.1f}" r="{radius}" fill="{color}"/>'


def _write_html(path, records):
    width = 520
    height = 420
    padding = 44

    observed = np.asarray([record["human"] for record in records], dtype=np.float32)
    predictions = np.asarray([record["prediction"] for record in records], dtype=np.float32)
    all_points = np.concatenate([observed, predictions.reshape(-1, 3)], axis=0)
    finite = all_points[np.all(np.isfinite(all_points), axis=1)]
    if len(finite) == 0:
        mins = np.zeros(3, dtype=np.float32)
        maxs = np.ones(3, dtype=np.float32)
    else:
        mins = np.min(finite, axis=0)
        maxs = np.max(finite, axis=0)
    margin = np.maximum((maxs - mins) * 0.08, 0.02)
    mins -= margin
    maxs += margin

    views = [
        ("Top view: x/y", 0, 1, (mins[0], maxs[0], mins[1], maxs[1])),
        ("Side view: x/z", 0, 2, (mins[0], maxs[0], mins[2], maxs[2])),
        ("Front view: y/z", 1, 2, (mins[1], maxs[1], mins[2], maxs[2])),
    ]

    latest_prediction = predictions[-1] if len(predictions) else np.zeros((0, 3))
    panels = []
    for title, xi, yi, bounds in views:
        observed_2d = _project(observed, xi, yi, width, height, padding, bounds)
        pred_2d = _project(latest_prediction, xi, yi, width, height, padding, bounds)
        lines = [
            _polyline(observed_2d, "#2563eb", width=2),
            _polyline(pred_2d, "#dc2626", width=2, dash="7 5"),
        ]
        if len(observed_2d):
            lines.append(_circle(observed_2d[-1], "#2563eb"))
        if len(pred_2d):
            lines.append(_circle(pred_2d[-1], "#dc2626"))
        panels.append(
            f"""
            <section>
              <h2>{html.escape(title)}</h2>
              <svg viewBox="0 0 {width} {height}" role="img">
                <rect x="1" y="1" width="{width - 2}" height="{height - 2}" fill="#fff" stroke="#d4d4d8"/>
                <line x1="{padding}" y1="{height - padding}" x2="{width - padding}" y2="{height - padding}" stroke="#a1a1aa"/>
                <line x1="{padding}" y1="{padding}" x2="{padding}" y2="{height - padding}" stroke="#a1a1aa"/>
                {''.join(lines)}
              </svg>
            </section>
            """
        )

    path.write_text(
        f"""
        <!doctype html>
        <html>
        <head>
          <meta charset="utf-8">
          <title>Human trajectory prediction</title>
          <style>
            body {{ font-family: system-ui, sans-serif; margin: 24px; color: #18181b; }}
            h1 {{ font-size: 24px; margin: 0 0 8px; }}
            h2 {{ font-size: 16px; margin: 20px 0 8px; }}
            .legend {{ display: flex; gap: 18px; margin: 16px 0; }}
            .legend span {{ display: inline-flex; align-items: center; gap: 6px; }}
            .legend i {{ width: 24px; height: 3px; display: inline-block; }}
            .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 20px; }}
            svg {{ width: 100%; max-width: {width}px; background: #fafafa; }}
          </style>
        </head>
        <body>
          <h1>Human trajectory prediction</h1>
          <p>Recorded {len(records)} sampled points. The red dashed line is the latest constant-velocity prediction.</p>
          <div class="legend">
            <span><i style="background:#2563eb"></i>Observed human hand</span>
            <span><i style="background:#dc2626"></i>Predicted future</span>
          </div>
          <div class="grid">{''.join(panels)}</div>
        </body>
        </html>
        """,
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--idx",
        type=int,
        default=None,
        help="Benchmark scene index. If omitted, a random scene is used.",
    )
    parser.add_argument("--sample-every", type=int, default=10, help="Predict every N sim steps.")
    parser.add_argument("--max-steps", type=int, default=4000, help="Stop after this many sim steps.")
    parser.add_argument(
        "--history",
        type=int,
        default=8,
        help="Number of recent sampled positions used for constant-velocity prediction.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=12,
        help="Number of future sampled positions to predict.",
    )
    parser.add_argument("--out", type=Path, default=Path("results/trajectories/human_prediction"))
    parser.add_argument("--seed", type=int, default=None, help="Random seed for repeatable scene choice.")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    cfg = get_cfg()
    env = HandoverBenchmarkWrapper(gym.make(cfg.ENV.ID, cfg=cfg))

    idx = args.idx
    if idx is None:
        idx = int(rng.integers(env.num_scenes))

    env.reset(idx=idx)

    records = []
    history = []
    dt = args.sample_every * cfg.SIM.TIME_STEP

    print(f"Predicting human trajectory from scene index {idx}.", flush=True)
    for step in range(1, args.max_steps + 1):
        action = np.zeros(9, dtype=np.float32)
        _, _, done, info = env.step(action)

        if step % args.sample_every == 0 or step == 1 or done:
            human = _get_human_hand_pos(env)
            history.append(human)
            recent_history = history[-args.history :]
            prediction = _constant_velocity_predict(recent_history, args.horizon, dt)
            records.append(
                {
                    "frame": step,
                    "human": human,
                    "prediction": prediction.astype(np.float32),
                }
            )
            print(
                f"step {step:05d} human={human} next={prediction[0]} final={prediction[-1]}",
                flush=True,
            )

        if done:
            print("Episode finished with status:", info["status"], flush=True)
            break

    frames = np.asarray([record["frame"] for record in records], dtype=np.int32)
    human = np.asarray([record["human"] for record in records], dtype=np.float32)
    prediction = np.asarray([record["prediction"] for record in records], dtype=np.float32)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    npz_path = args.out.with_suffix(".npz")
    csv_path = args.out.with_suffix(".csv")
    html_path = args.out.with_suffix(".html")

    np.savez_compressed(npz_path, frame=frames, human=human, prediction=prediction)
    _write_csv(csv_path, records)
    _write_html(html_path, records)

    print("\nSaved:")
    print(" ", npz_path)
    print(" ", csv_path)
    print(" ", html_path)


if __name__ == "__main__":
    main()
