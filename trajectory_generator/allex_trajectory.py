# SPDX-License-Identifier: Apache-2.0
"""ALLEX 전체 관절 궤적 생성: 14개 CSV → 공통 시간 병합 → NPZ.

Usage::
    python -m trajectory_generator.allex_trajectory \\
        allex_description/zz_dyna_cal_high_speed_group/ \\
        -o target_all_joints.npz --hz 200
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from trajectory_generator.allex_joint_map import ALLEX_CSV_JOINT_MAP, ALLEX_TOTAL_DOFS
from trajectory_generator.hermite_spline import generate_trajectory, parse_via_csv


def generate_allex_trajectory(
    csv_dir: str | Path,
    hz: float = 200.0,
    duration: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """14개 CSV에서 전체 ALLEX 궤적 생성.

    Args:
        csv_dir: CSV 파일들이 있는 디렉토리.
        hz: 출력 샘플링 주파수 [Hz].
        duration: 출력 궤적 길이 [s]. None이면 가장 긴 CSV 기준.

    Returns:
        (time_out, pos_out_rad): shape (n_steps+1,), (n_steps+1, 60).
    """
    csv_dir = Path(csv_dir)

    # 각 CSV 파싱 + 최대 duration 결정
    group_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    max_dur = 0.0

    for csv_name, dof_indices in ALLEX_CSV_JOINT_MAP.items():
        csv_path = csv_dir / f"{csv_name}.csv"
        if not csv_path.exists():
            print(f"  WARNING: {csv_path.name} not found, skipping")
            continue
        t_via, pos_via = parse_via_csv(csv_path)
        if len(t_via) < 1:
            continue
        group_data[csv_name] = (t_via, pos_via)
        max_dur = max(max_dur, float(t_via[-1]))
        print(f"  {csv_name:25s}: {len(t_via):3d} via pts, "
              f"{t_via[-1]:.2f}s, {pos_via.shape[1]} joints → dof {dof_indices}")

    if duration is None:
        duration = max_dur

    # 공통 시간축
    dt = 1.0 / hz
    n_steps = int(duration / dt)
    t_out = np.arange(n_steps + 1) * dt

    # 전체 DOF 궤적 (초기값 0)
    pos_all = np.zeros((n_steps + 1, ALLEX_TOTAL_DOFS), dtype=np.float32)

    for csv_name, (t_via, pos_via) in group_data.items():
        dof_indices = ALLEX_CSV_JOINT_MAP[csv_name]
        n_joints_csv = pos_via.shape[1]

        if n_joints_csv != len(dof_indices):
            print(f"  WARNING: {csv_name} has {n_joints_csv} joints "
                  f"but map expects {len(dof_indices)}, skipping")
            continue

        # 각 joint별 spline 보간
        _, pos_interp = generate_trajectory(t_via, pos_via, hz=hz, duration=duration)

        for j, dof in enumerate(dof_indices):
            pos_all[:, dof] = pos_interp[:, j]

    return t_out.astype(np.float32), pos_all


def main():
    parser = argparse.ArgumentParser(
        description="ALLEX 14 CSV → 전체 관절 target trajectory NPZ"
    )
    parser.add_argument("csv_dir", type=str, help="CSV 파일 디렉토리")
    parser.add_argument("-o", "--output", type=str, default=None)
    parser.add_argument("--hz", type=float, default=200.0)
    parser.add_argument("--duration", type=float, default=None)
    args = parser.parse_args()

    csv_dir = Path(args.csv_dir)
    out_path = Path(args.output) if args.output else csv_dir / "target_all_joints.npz"

    print(f"Loading CSVs from: {csv_dir}")
    t_out, pos_out = generate_allex_trajectory(csv_dir, hz=args.hz, duration=args.duration)

    print(f"\nGenerated trajectory:")
    print(f"  shape: {pos_out.shape} ({args.hz:.0f}Hz)")
    print(f"  duration: {t_out[-1]:.2f}s")
    print(f"  non-zero DOFs: {np.sum(np.any(pos_out != 0, axis=0))}/{ALLEX_TOTAL_DOFS}")

    np.savez(str(out_path), time=t_out, position_rad=pos_out, hz=np.array(args.hz))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
