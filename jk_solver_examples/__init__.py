# SPDX-License-Identifier: Apache-2.0
"""JK Solver 패키지 — viewer 초기화 + example switching/reset 지원.

Newton `newton.examples.init`과 동일하되, GL 뷰어만 `JkViewerGL`로 연다."""
from __future__ import annotations

import importlib


def _patch_run() -> None:
    """newton.examples.run()을 래핑하여 IRIM 브라우저 switch/reset 지원."""
    import warp as wp

    import newton.examples as ne

    if getattr(ne, "_jk_run_patched", False):
        return

    _orig_run = ne.run

    def _patched_run(example, args):
        viewer = example.viewer
        has_irim = hasattr(viewer, "_irim_switch_target")

        if not has_irim:
            return _orig_run(example, args)

        if hasattr(example, "gui") and hasattr(viewer, "register_ui_callback"):
            viewer.register_ui_callback(lambda ui, ex=example: ex.gui(ui), position="side")

        perform_test = args is not None and args.test
        test_post_step = perform_test and hasattr(example, "test_post_step")
        test_final = perform_test and hasattr(example, "test_final")

        while viewer.is_running():
            # IRIM switch → module import + Example 재생성
            if viewer._irim_switch_target is not None:
                target = viewer._irim_switch_target
                viewer._irim_switch_target = None
                mod = importlib.import_module(target)
                parser = getattr(mod.Example, "create_parser", ne.create_parser)()
                new_args = parser.parse_known_args()[0]
                example = mod.Example(viewer, new_args)
                if hasattr(example, "gui") and hasattr(viewer, "register_ui_callback"):
                    viewer.register_ui_callback(lambda ui, ex=example: ex.gui(ui), position="side")
                continue

            # IRIM reset → example.reset() 호출
            if viewer._irim_reset_requested:
                viewer._irim_reset_requested = False
                if hasattr(example, "reset"):
                    example.reset()
                continue

            if not viewer.is_paused():
                with wp.ScopedTimer("step", active=False):
                    example.step()
            if test_post_step:
                example.test_post_step()

            with wp.ScopedTimer("render", active=False):
                example.render()

        if perform_test:
            if test_final:
                example.test_final()
            elif not (test_post_step or test_final):
                raise NotImplementedError("Example does not have a test_final or test_post_step method")

        viewer.close()

    ne.run = _patched_run
    ne._jk_run_patched = True


def init(parser=None):
    """`newton.examples.init`과 동일한 인자·동작, ``viewer=gl`` 일 때만 JK 패널 뷰어 사용."""
    import warp as wp

    import newton.examples
    import newton.viewer

    _patch_run()

    from viewer.jk_viewer import JkViewerCfg, JkViewerGL

    if parser is None:
        parser = newton.examples.create_parser()
        args = parser.parse_known_args()[0]
    else:
        args = parser.parse_args()

    if args.quiet:
        wp.config.quiet = True

    if args.device:
        wp.set_device(args.device)

    if args.viewer == "gl":
        cfg = JkViewerCfg()
        viewer = JkViewerGL(
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
        viewer = newton.viewer.ViewerNull(num_frames=args.num_frames)
    elif args.viewer == "viser":
        viewer = newton.viewer.ViewerViser()
    else:
        raise ValueError(f"Invalid viewer: {args.viewer}")

    return viewer, args
