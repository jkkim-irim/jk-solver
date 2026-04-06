# SPDX-License-Identifier: Apache-2.0
"""ALLEX CSV 파일 → joint DOF index 매핑.

각 CSV의 joint_1, joint_2, ... 열이 ALLEX의 어떤 joint에 대응하는지 정의.
DOF index는 Newton Model의 joint_qd_start 기준.
"""
from __future__ import annotations

# CSV 파일명 → (joint DOF indices) 매핑
# CSV의 joint_1 → dof_indices[0], joint_2 → dof_indices[1], ...
ALLEX_CSV_JOINT_MAP: dict[str, list[int]] = {
    # Waist: joint_1=Waist_Yaw(0), joint_2=Waist_Lower_Pitch(2)
    "theOne_waist": [0, 2],

    # Neck: joint_1=Neck_Pitch(4), joint_2=Neck_Yaw(5)
    "theOne_neck": [4, 5],

    # Left Arm: joint_1~7 = L_Shoulder_Pitch~L_Wrist_Pitch (6~12)
    "Arm_L_theOne": [6, 7, 8, 9, 10, 11, 12],

    # Right Arm: joint_1~7 = R_Shoulder_Pitch~R_Wrist_Pitch (33~39)
    "Arm_R_theOne": [33, 34, 35, 36, 37, 38, 39],

    # Left Thumb: joint_1=Yaw(13), joint_2=CMC(14), joint_3=MCP(15)
    "Hand_L_thumb_wir": [13, 14, 15],
    # Left Index: joint_1=ABAD(17), joint_2=MCP(18), joint_3=PIP(19)
    "Hand_L_index_wir": [17, 18, 19],
    # Left Middle: joint_1=ABAD(21), joint_2=MCP(22), joint_3=PIP(23)
    "Hand_L_middle_wir": [21, 22, 23],
    # Left Ring: joint_1=ABAD(25), joint_2=MCP(26), joint_3=PIP(27)
    "Hand_L_ring_wir": [25, 26, 27],
    # Left Little: joint_1=ABAD(29), joint_2=MCP(30), joint_3=PIP(31)
    "Hand_L_little_wir": [29, 30, 31],

    # Right Thumb: joint_1=Yaw(40), joint_2=CMC(41), joint_3=MCP(42)
    "Hand_R_thumb_wir": [40, 41, 42],
    # Right Index: joint_1=ABAD(44), joint_2=MCP(45), joint_3=PIP(46)
    "Hand_R_index_wir": [44, 45, 46],
    # Right Middle: joint_1=ABAD(48), joint_2=MCP(49), joint_3=PIP(50)
    "Hand_R_middle_wir": [48, 49, 50],
    # Right Ring: joint_1=ABAD(52), joint_2=MCP(53), joint_3=PIP(54)
    "Hand_R_ring_wir": [52, 53, 54],
    # Right Little: joint_1=ABAD(56), joint_2=MCP(57), joint_3=PIP(58)
    "Hand_R_little_wir": [56, 57, 58],
}

ALLEX_TOTAL_DOFS = 60
