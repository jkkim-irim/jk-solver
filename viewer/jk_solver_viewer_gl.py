# SPDX-License-Identifier: Apache-2.0
"""
Newton `ViewerGL`의 왼쪽 패널만 DexblindNewtonViewerGL 스타일로 교체.

- 업스트림 `newton/_src/viewer/viewer_gl.py`는 수정하지 않음.
- 서브클래스에서 `_render_left_panel`만 오버라이드.
"""

from __future__ import annotations

import newton as nt
from newton.viewer import ViewerGL

DEFAULT_PANEL_WIDTH = 600
DEFAULT_FONT_SCALE = 2.5
AXIS_NAMES = ("X", "Y", "Z")


class JkSolverViewerGL(ViewerGL):
    """
    Dexblind `DexblindNewtonViewerGL`과 유사한 고정 폭·접이식 섹션 레이아웃.

    - `register_ui_callback(..., "side")`: \"JK Solver\" 섹션
    - `register_ui_callback(..., "panel")`: 패널 상단 (예제 브라우저 등)
    - `self._jk_metadata`: ``{\"num_envs\": int}`` 등 표시용 메타데이터
    """

    def __init__(
        self,
        *args,
        panel_initial_width: int = DEFAULT_PANEL_WIDTH,
        font_scale: float = DEFAULT_FONT_SCALE,
        **kwargs,
    ):
        self._jk_panel_width = kwargs.pop("panel_initial_width", panel_initial_width)
        self._jk_font_scale = kwargs.pop("font_scale", font_scale)
        super().__init__(*args, **kwargs)
        self._jk_metadata: dict = {}
        self._show_joint_state = False
        self._show_env_state = False
        self._show_privileged_obs = False
        self._show_torque_speed = False
        self._apply_jk_font_scale()

    def _apply_jk_font_scale(self) -> None:
        """DexblindNewtonVisualizer와 같이 ImGui 글자 크기 확대."""
        if getattr(self, "ui", None) is None or not getattr(self.ui, "is_available", False):
            return
        imgui = self.ui.imgui
        io = self.ui.io
        s = float(self._jk_font_scale)
        style = imgui.get_style()
        if hasattr(style, "font_scale_main"):
            style.font_scale_main = s
        elif hasattr(io, "font_global_scale"):
            io.font_global_scale = s

    def _render_left_panel(self):
        imgui = self.ui.imgui
        io = self.ui.io
        panel_h = io.display_size[1] - 20
        imgui.set_next_window_pos(imgui.ImVec2(0, 0), imgui.Cond_.always.value)
        imgui.set_next_window_size(
            imgui.ImVec2(self._jk_panel_width, panel_h),
            imgui.Cond_.always.value,
        )
        flags = imgui.WindowFlags_.no_resize.value | imgui.WindowFlags_.no_move.value
        if imgui.begin(f"Newton Viewer v{nt.__version__}", flags=flags):
            imgui.separator()

            for callback in self._ui_callbacks["panel"]:
                callback(self.ui.imgui)

            imgui.set_next_item_open(False, imgui.Cond_.appearing)
            if imgui.collapsing_header("JK Solver"):
                for callback in self._ui_callbacks["side"]:
                    callback(self.ui.imgui)

            imgui.spacing()
            if self.model is not None:
                imgui.set_next_item_open(False, imgui.Cond_.appearing)
                if imgui.collapsing_header("Model Information"):
                    imgui.separator()
                    num_envs = self._jk_metadata.get("num_envs", 0)
                    imgui.text(f"Environments: {num_envs}")
                    imgui.text(f"Up Axis: {AXIS_NAMES[self.model.up_axis]}")
                    g = self.model.gravity.numpy()[0]
                    imgui.text(f"Gravity: ({g[0]:.2f}, {g[1]:.2f}, {g[2]:.2f})")

                imgui.spacing()
                imgui.set_next_item_open(False, imgui.Cond_.appearing)
                if imgui.collapsing_header("Visualization"):
                    imgui.separator()
                    _, self.show_joints = imgui.checkbox("Show Joints", self.show_joints)
                    _, self.show_contacts = imgui.checkbox("Show Contacts", self.show_contacts)
                    _, self.show_com = imgui.checkbox("Show Center of Mass", self.show_com)
                    _, self.show_collision = imgui.checkbox("Show Collision", self.show_collision)
                    _, self.show_visual = imgui.checkbox("Show Visual", self.show_visual)

            imgui.spacing()
            imgui.set_next_item_open(False, imgui.Cond_.appearing)
            if imgui.collapsing_header("Observations"):
                imgui.separator()
                _, self._show_joint_state = imgui.checkbox("JointState", self._show_joint_state)
                _, self._show_env_state = imgui.checkbox("EnvState", self._show_env_state)
                _, self._show_privileged_obs = imgui.checkbox("PrivilegedObs", self._show_privileged_obs)
                _, self._show_torque_speed = imgui.checkbox("TorqueSpeed", self._show_torque_speed)

            imgui.spacing()
            imgui.set_next_item_open(False, imgui.Cond_.appearing)
            if imgui.collapsing_header("Rendering Options"):
                imgui.separator()
                _, self.renderer.draw_sky = imgui.checkbox("Sky", self.renderer.draw_sky)
                _, self.renderer.draw_shadows = imgui.checkbox("Shadows", self.renderer.draw_shadows)
                _, self.renderer.draw_wireframe = imgui.checkbox("Wireframe", self.renderer.draw_wireframe)

            imgui.spacing()
            imgui.set_next_item_open(False, imgui.Cond_.appearing)
            if imgui.collapsing_header("Viewer Controls"):
                imgui.separator()
                imgui.text("WASD - Forward/Left/Back/Right")
                imgui.text("QE - Down/Up")
                imgui.text("Left Click - Look around")
                imgui.text("Scroll - Zoom")
                imgui.text("Space - Pause/Resume Rendering")
                imgui.text("H - Toggle UI")
                imgui.text("ESC - Exit")

        imgui.end()
