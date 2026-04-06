# SPDX-License-Identifier: Apache-2.0
"""Newton ViewerGL 확장 — IRIM Physics Test 패널 + Param Tuner.

업스트림 `newton/_src/viewer/viewer_gl.py`는 수정하지 않음.
서브클래스에서 `_render_left_panel`만 오버라이드.
"""
from __future__ import annotations

from dataclasses import dataclass

import newton as nt
from newton.viewer import ViewerGL

AXIS_NAMES = ("X", "Y", "Z")

_IRIM_TESTS: list[tuple[str, str]] = [
    ("test_contact_force", "jk_solver_examples.contacts.test_contact_force"),
]


@dataclass
class JkViewerCfg:
    """JkViewerGL 생성 시 사용하는 기본 창·패널 옵션."""

    window_width: int = 1920
    window_height: int = 1080
    panel_initial_width: int = 600
    font_scale: float = 2.5


class JkViewerGL(ViewerGL):
    """고정 폭·접이식 섹션 레이아웃의 Newton ViewerGL 확장."""

    def __init__(
        self,
        *args,
        panel_initial_width: int = 600,
        font_scale: float = 2.5,
        **kwargs,
    ):
        self._jk_panel_width = kwargs.pop("panel_initial_width", panel_initial_width)
        self._jk_font_scale = kwargs.pop("font_scale", font_scale)
        super().__init__(*args, **kwargs)
        self._irim_switch_target: str | None = None
        self._irim_reset_requested: bool = False
        self._irim_param_tune_open: bool = False
        self._apply_font_scale()

    def reset_model(self, model):
        """set_model 1회 제한을 우회하여 model 교체."""
        self.model = None
        self._shape_instances = {}
        self.set_model(model)

    def _apply_font_scale(self) -> None:
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

            # ── IRIM Physics Test ──
            imgui.set_next_item_open(False, imgui.Cond_.appearing)
            if imgui.collapsing_header("IRIM Physics Test"):
                for name, module_path in _IRIM_TESTS:
                    clicked, _ = imgui.selectable(name, False)
                    if clicked:
                        self._irim_switch_target = module_path

            imgui.set_next_item_open(False, imgui.Cond_.appearing)
            if imgui.collapsing_header("Param Tuner"):
                if imgui.button("Reset"):
                    self._irim_reset_requested = True
                imgui.same_line()
                label = "Param Tune [ON]" if self._irim_param_tune_open else "Param Tune"
                if imgui.button(label):
                    self._irim_param_tune_open = not self._irim_param_tune_open
                imgui.separator()
                for callback in self._ui_callbacks["side"]:
                    callback(self.ui.imgui)

            imgui.spacing()
            if self.model is not None:
                imgui.set_next_item_open(False, imgui.Cond_.appearing)
                if imgui.collapsing_header("Model Information"):
                    imgui.separator()
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
