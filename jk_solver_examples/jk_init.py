# SPDX-License-Identifier: Apache-2.0
"""Newton `newton.examples.init`과 동일하되, GL 뷰어만 `JkSolverViewerGL`로 연다."""

from __future__ import annotations

_JK_BROWSER_EXTRA = {
    "jk_nodiffsim_ball": "jk_solver_examples.examples.diffsim.example_jk_nodiffsim_ball",
    "jk_nodiffsim_cloth": "jk_solver_examples.examples.diffsim.example_jk_nodiffsim_cloth",
    "jk_nodiffsim_drone": "jk_solver_examples.examples.diffsim.example_jk_nodiffsim_drone",
}


def _register_jk_examples_in_browser() -> None:
    """`newton.examples.get_examples`에 JK nodiff 예제 등록 (Example 트리에 `diffsim` 아래 표시)."""
    import newton.examples as ne

    if getattr(ne, "_jk_browser_examples_registered", False):
        return

    _orig = ne.get_examples

    def get_examples_merged():
        out = _orig().copy()
        out.update(_JK_BROWSER_EXTRA)
        return out

    ne.get_examples = get_examples_merged
    ne._jk_browser_examples_registered = True


def init(parser=None):
    """`newton.examples.init`과 동일한 인자·동작, ``viewer=gl`` 일 때만 JK 패널 뷰어 사용."""
    import warp as wp  # noqa: PLC0415

    import newton.examples  # noqa: PLC0415

    _register_jk_examples_in_browser()
    import newton.viewer  # noqa: PLC0415

    from viewer.jk_solver_viewer_cfg import JkSolverViewerCfg
    from viewer.jk_solver_viewer_gl import JkSolverViewerGL

    if parser is None:
        parser = newton.examples.create_parser()
        args = parser.parse_known_args()[0]
    else:
        args = parser.parse_args()

    if args.quiet:
        wp.config.quiet = True

    if args.device:
        wp.set_device(args.device)

    if args.benchmark is not False:
        args.viewer = "null"

    if args.viewer == "gl":
        cfg = JkSolverViewerCfg()
        viewer = JkSolverViewerGL(
            width=cfg.window_width,
            height=cfg.window_height,
            headless=args.headless,
            panel_initial_width=cfg.panel_initial_width,
            font_scale=cfg.font_scale,
        )
    elif args.viewer == "usd":
        if args.output_path is None:
            raise ValueError("--output-path is required when using usd viewer")
        viewer = newton.viewer.ViewerUSD(output_path=args.output_path, num_frames=args.num_frames)
    elif args.viewer == "rerun":
        viewer = newton.viewer.ViewerRerun(address=args.rerun_address)
    elif args.viewer == "null":
        viewer = newton.viewer.ViewerNull(
            num_frames=args.num_frames,
            benchmark=args.benchmark is not False,
            benchmark_timeout=args.benchmark or None,
        )
    elif args.viewer == "viser":
        viewer = newton.viewer.ViewerViser()
    else:
        raise ValueError(f"Invalid viewer: {args.viewer}")

    return viewer, args
