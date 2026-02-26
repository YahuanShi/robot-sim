"""
Tile evaluation videos side by side into one labelled demo GIF (for READMEs).

Each input is LABEL=PATH; the videos are played in sync, a finished video holds its
last frame until the longest one ends.

Usage:
    python scripts/make_demo_gif.py --out docs/can_dp_vs_act.gif \
        "Diffusion Policy=outputs/dp_can/videos/seed100000_success.mp4" \
        "ACT=outputs/act_can/eval_franka_can_chunk/videos/seed100000_fail.mp4"
"""

import argparse

import cv2
import imageio.v2 as imageio
import numpy as np


def read_frames(path):
    reader = imageio.get_reader(path)
    frames = [np.asarray(f) for f in reader]
    reader.close()
    if not frames:
        raise SystemExit(f"No frames in {path}")
    return frames


def label(frame, text):
    out = frame.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(
        out,
        text,
        (6, 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("videos", nargs="+", metavar="LABEL=PATH")
    parser.add_argument("--out", required=True)
    parser.add_argument("--size", type=int, default=256, help="Tile size in pixels")
    parser.add_argument("--stride", type=int, default=2, help="Keep every n-th frame")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Cut every video after this many frames",
    )
    args = parser.parse_args()

    tiles = []
    for spec in args.videos:
        name, _, path = spec.partition("=")
        frames = read_frames(path)[: args.max_frames]
        frames = [
            cv2.resize(f, (args.size, args.size), interpolation=cv2.INTER_AREA)
            for f in frames
        ]
        tiles.append([label(f, name) for f in frames])

    n = max(len(t) for t in tiles)
    out = []
    for i in range(0, n, args.stride):
        out.append(np.concatenate([t[min(i, len(t) - 1)] for t in tiles], axis=1))
    out += [out[-1]] * args.fps  # hold the final state for a second
    imageio.mimsave(args.out, out, duration=1000 / args.fps, loop=0)
    print(f"saved {args.out} ({len(out)} frames)")


if __name__ == "__main__":
    main()
