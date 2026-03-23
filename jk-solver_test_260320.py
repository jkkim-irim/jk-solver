#!/usr/bin/env python3
"""ALLEX MJCF를 Newton Model로 로드하고 ViewerGL로 렌더링한다."""

import os
import sys
from pathlib import Path

_newton_path = Path(__file__).parent.parent / "newton"
if _newton_path.exists():
    sys.path.insert(0, str(_newton_path.parent))
else:
    raise RuntimeError(f"Newton 경로를 찾을 수 없습니다: {_newton_path}")

_jk_solver_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_jk_solver_dir))

import argparse

import warp as wp

wp.config.quiet = True

import newton

from viewer.jk_solver_viewer_cfg import JkSolverViewerCfg
from viewer.jk_solver_viewer_gl import JkSolverViewerGL


def load_allex(mjcf_path: str) -> newton.Model:
    builder = newton.ModelBuilder()
    builder.add_mjcf(
        mjcf_path,
        floating=False,
        enable_self_collisions=False,
        parse_visuals=True,
        parse_mujoco_options=True,
        verbose=False,
    )
    return builder.finalize(requires_grad=False)


def main():
    parser = argparse.ArgumentParser(description="ALLEX MJCF 로드 및 ViewerGL 렌더")
    parser.add_argument(
        "--mjcf-path",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "allex_description", "mjcf", "ALLEX.xml"),
    )
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="OpenGL 창 없이(오프스크린) ViewerGL 초기화",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        metavar="N",
        help="N프레임 후 종료 (0=무한, 헤드리스·자동화 시 유용)",
    )
    args = parser.parse_args()

    cfg = JkSolverViewerCfg()

    wp.set_device(args.device)
    print(f"device={wp.get_device()}")
    print(f"mjcf={args.mjcf_path}")

    model = load_allex(args.mjcf_path)
    model.set_gravity((0.0, 0.0, -9.81))

    print(f"ALLEX 로드 완료: joint_dof_count={model.joint_dof_count}, device={model.device}")

    state = model.state()
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)

    viewer = JkSolverViewerGL(
        width=cfg.window_width,
        height=cfg.window_height,
        headless=args.headless,
        panel_initial_width=cfg.panel_initial_width,
        font_scale=cfg.font_scale,
    )
    viewer.set_model(model)

    sim_time = 0.0
    max_frames = args.max_frames
    if max_frames == 0:
        print("ViewerGL 실행 중 (창을 닫으면 종료).")
    else:
        print(f"ViewerGL: 최대 {max_frames}프레임 후 종료.")

    frame = 0
    while viewer.is_running():
        viewer.begin_frame(sim_time)
        viewer.log_state(state)
        viewer.end_frame()
        frame += 1
        if max_frames > 0 and frame >= max_frames:
            break

    viewer.close()


if __name__ == "__main__":
    main()
