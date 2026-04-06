# SPDX-License-Identifier: Apache-2.0
"""Via point CSV → target trajectory NPZ 변환.

Usage::
    python -m trajectory_generator allex_description/zz_dyna_cal_high_speed_group/Arm_R_theOne.csv -o target_rarm.npz --hz 200
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from trajectory_generator.hermite_spline import generate_trajectory_from_csv, parse_via_csv


def main():
    parser = argparse.ArgumentParser(description="Via point CSV → trajectory NPZ")
    parser.add_argument("csv", type=str, help="Via point CSV 파일 경로")
    parser.add_argument("-o", "--output", type=str, default=None, help="출력 NPZ 경로")
    parser.add_argument("--hz", type=float, default=200.0, help="샘플링 주파수 [Hz]")
    parser.add_argument("--duration", type=float, default=None, help="궤적 길이 [s]")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = csv_path.with_suffix(".npz")

    # via points 확인
    time_via, pos_via = parse_via_csv(csv_path)
    print(f"Via points: {len(time_via)} points, {pos_via.shape[1]} joints")
    print(f"  duration: {time_via[-1]:.2f}s")
    print(f"  pos range: [{np.degrees(pos_via.min()):.1f}, {np.degrees(pos_via.max()):.1f}] deg")

    # 궤적 생성
    t_out, pos_out = generate_trajectory_from_csv(
        csv_path, hz=args.hz, duration=args.duration,
    )

    print(f"\nGenerated trajectory:")
    print(f"  shape: {pos_out.shape} ({args.hz:.0f}Hz)")
    print(f"  duration: {t_out[-1]:.2f}s")
    print(f"  pos range: [{np.degrees(pos_out.min()):.1f}, {np.degrees(pos_out.max()):.1f}] deg")

    # 저장
    np.savez(
        str(out_path),
        time=t_out,
        position_rad=pos_out,
        hz=np.array(args.hz),
    )
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
