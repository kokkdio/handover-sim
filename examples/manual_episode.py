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


def _get_robot_hand_pos(env):
    hand_link = env.env.panda.LINK_IND_HAND
    return _as_numpy(env.env.panda.body.link_state[0, hand_link, 0:3])


def _get_object_pos(env):
    object_id = env.env.ycb.ids[0]
    object_body = env.env.ycb.bodies[object_id]
    return _as_numpy(object_body.link_state[0, 6, 0:3])


def _get_human_hand_pos(env):
    mano = env.env.mano
    if hasattr(mano, "_pose") and hasattr(mano, "_frame"):
        return _as_numpy(mano._pose[mano._frame, 0:3])
    return np.full(3, np.nan, dtype=np.float32)


def _write_csv(path, frames, robot_hand, human_hand, obj):
    columns = [
        "frame",
        "robot_hand_x",
        "robot_hand_y",
        "robot_hand_z",
        "human_hand_x",
        "human_hand_y",
        "human_hand_z",
        "object_x",
        "object_y",
        "object_z",
    ]
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for i, frame in enumerate(frames):
            writer.writerow(
                [frame]
                + robot_hand[i].tolist()
                + human_hand[i].tolist()
                + obj[i].tolist()
            )


def _project(points, x_index, y_index, width, height, padding, bounds):
    x_min, x_max, y_min, y_max = bounds
    x = points[:, x_index]
    y = points[:, y_index]
    sx = padding + (x - x_min) / max(x_max - x_min, 1e-6) * (width - 2 * padding)
    sy = height - padding - (y - y_min) / max(y_max - y_min, 1e-6) * (height - 2 * padding)
    return np.stack([sx, sy], axis=1)


def _polyline(points, color):
    valid = np.all(np.isfinite(points), axis=1)
    if np.count_nonzero(valid) < 2:
        return ""
    pairs = " ".join(f"{x:.1f},{y:.1f}" for x, y in points[valid])
    return f'<polyline points="{pairs}" fill="none" stroke="{color}" stroke-width="2"/>'


def _circle(points, color):
    valid = np.all(np.isfinite(points), axis=1)
    if not np.any(valid):
        return ""
    x, y = points[valid][-1]
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>'


def _write_html(path, frames, robot_hand, human_hand, obj):
    width = 520
    height = 420
    padding = 44
    series = {
        "Robot hand": (robot_hand, "#2563eb"),
        "Human hand": (human_hand, "#16a34a"),
        "Object": (obj, "#dc2626"),
    }
    all_points = np.concatenate([robot_hand, human_hand, obj], axis=0)
    finite = all_points[np.all(np.isfinite(all_points), axis=1)]
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

    panels = []
    for title, xi, yi, bounds in views:
        lines = []
        for _, (points, color) in series.items():
            projected = _project(points, xi, yi, width, height, padding, bounds)
            lines.append(_polyline(projected, color))
            lines.append(_circle(projected, color))
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

    legend = "".join(
        f'<span><i style="background:{color}"></i>{html.escape(name)}</span>'
        for name, (_, color) in series.items()
    )
    path.write_text(
        f"""
        <!doctype html>
        <html>
        <head>
          <meta charset="utf-8">
          <title>Handover trajectory</title>
          <style>
            body {{ font-family: system-ui, sans-serif; margin: 24px; color: #18181b; }}
            h1 {{ font-size: 24px; margin: 0 0 8px; }}
            h2 {{ font-size: 16px; margin: 20px 0 8px; }}
            .legend {{ display: flex; gap: 18px; margin: 16px 0; }}
            .legend span {{ display: inline-flex; align-items: center; gap: 6px; }}
            .legend i {{ width: 12px; height: 12px; border-radius: 999px; display: inline-block; }}
            .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 20px; }}
            svg {{ width: 100%; max-width: {width}px; background: #fafafa; }}
          </style>
        </head>
        <body>
          <h1>Handover trajectory</h1>
          <p>Recorded {len(frames)} sampled points. The dot marks the final sampled position.</p>
          <div class="legend">{legend}</div>
          <div class="grid">{''.join(panels)}</div>
        </body>
        </html>
        """,
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--idx", type=int, default=0, help="Benchmark scene index.")
    parser.add_argument("--sample-every", type=int, default=10, help="Record every N sim steps.")
    parser.add_argument("--max-steps", type=int, default=4000, help="Stop after this many sim steps.")
    parser.add_argument("--out", type=Path, default=Path("results/trajectories/manual_episode"))
    args = parser.parse_args()

    cfg = get_cfg()
    env = HandoverBenchmarkWrapper(gym.make(cfg.ENV.ID, cfg=cfg))
    env.reset(idx=args.idx)

    frames = []
    robot_hand = []
    human_hand = []
    obj = []

    print("Recording trajectory. Press Ctrl+C to stop early.", flush=True)
    for step in range(1, args.max_steps + 1):
        action = np.zeros(9, dtype=np.float32)
        _, _, done, info = env.step(action)

        if step % args.sample_every == 0 or step == 1 or done:
            frames.append(step)
            robot_hand.append(_get_robot_hand_pos(env))
            human_hand.append(_get_human_hand_pos(env))
            obj.append(_get_object_pos(env))
            print(
                f"step {step:05d} "
                f"robot={robot_hand[-1]} human={human_hand[-1]} object={obj[-1]}",
                flush=True,
            )

        if done:
            print("Episode finished with status:", info["status"], flush=True)
            break

    frames = np.asarray(frames, dtype=np.int32)
    robot_hand = np.asarray(robot_hand, dtype=np.float32)
    human_hand = np.asarray(human_hand, dtype=np.float32)
    obj = np.asarray(obj, dtype=np.float32)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    npz_path = args.out.with_suffix(".npz")
    csv_path = args.out.with_suffix(".csv")
    html_path = args.out.with_suffix(".html")

    np.savez_compressed(
        npz_path,
        frame=frames,
        robot_hand=robot_hand,
        human_hand=human_hand,
        object=obj,
    )
    _write_csv(csv_path, frames, robot_hand, human_hand, obj)
    _write_html(html_path, frames, robot_hand, human_hand, obj)

    print("\nSaved:")
    print(" ", npz_path)
    print(" ", csv_path)
    print(" ", html_path)


if __name__ == "__main__":
    main()
