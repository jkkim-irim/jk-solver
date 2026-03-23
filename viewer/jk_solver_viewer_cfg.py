# SPDX-License-Identifier: Apache-2.0
"""JK Solver용 Newton ViewerGL 설정 (업스트림 newton 수정 없이 확장)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class JkSolverViewerCfg:
    """`JkSolverViewerGL` 생성 시 사용하는 기본 창·패널 옵션."""

    window_width: int = 1920
    window_height: int = 1080
    panel_initial_width: int = 600
    """왼쪽 ImGui 패널 너비 (DexblindNewtonViewerGL과 동일한 역할)."""
    font_scale: float = 2.5
    """ImGui 본문 스케일 (`font_scale_main` / `font_global_scale`). DexblindNewtonVisualizerCfg와 동일 기본값."""
