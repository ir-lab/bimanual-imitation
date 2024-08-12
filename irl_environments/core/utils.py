from enum import Enum

import numpy as np
from transforms3d.euler import quat2mat
from transforms3d.quaternions import mat2quat
from transforms3d.utils import normalized_vector as normed


def resample_by_interpolation(signal, n):
    resampled_signal = np.interp(
        np.linspace(0.0, 1.0, n, endpoint=False),  # where to interpret
        np.linspace(0.0, 1.0, len(signal), endpoint=False),  # known positions
        signal,  # known data points
    )
    return resampled_signal


def quat2sd(quat):
    assert quat.shape == (4,)
    return quat2mat(quat)[:, :2].flatten()


def sd2quat(in_sd):
    assert in_sd.shape == (6,)
    sd = in_sd.reshape(3, 2)
    a1, a2 = sd[:, 0], sd[:, 1]
    b1 = normed(a1)
    out_mat = np.zeros((3, 3))
    out_mat[:, 0] = b1
    b2 = normed(a2 - b1.dot(a2) * b1)
    out_mat[:, 1] = b2
    out_mat[:, 2] = np.cross(b1, b2)
    return mat2quat(out_mat)


def cart2polar(r_diff):
    xd, yd, zd = r_diff
    radius = np.linalg.norm([xd, yd, zd])
    polar = np.arccos(zd / np.linalg.norm([xd, yd, zd]))  # Vertical Angle
    azimuth = np.sign(yd) * np.arccos(xd / np.linalg.norm([xd, yd]))  # Horizontal Angle
    return [radius, polar, np.sin(azimuth), np.cos(azimuth)]


def get_enum_value(key, enum: Enum):
    assert key in [x.value for x in enum], f'Value "{key}" not part of {enum}'
    for match_key in enum:
        if key == match_key.value:
            return match_key


class GripType(Enum):
    OPEN = "open"
    CLOSE = "close"


class PursuitType(Enum):
    PURE_PURSUIT = "pure_pursuit"
    SIMPLE = "simple"
    INSERT = "insert"


class ActionGroup(Enum):
    # Path 2D
    SPEED = "speed"
    DELTA_STEER = "delta_steer"
    # Mujoco 3D
    DELTA_POSITION = "delta_position"
    DELTA_SIX_DOF = "delta_six_dof"
    DELTA_QUAT = "delta_quat"
    DELTA_EULER = "delta_euler"


class ObservationGroup(Enum):
    # Path 2D
    POSITION_2D = "position_2D"
    YAW = "yaw"
    # Mujoco 3D
    POSITION = "position"
    DELTA_POSITION_POLAR = "delta_position_polar"
    DELTA_TARGET_POS = "delta_target_pos"
    MALE_OBJ_POS = "male_obj_pos"
    MALE_OBJ_SIX_DOF = "male_obj_six_dof"
    FEMALE_OBJ_POS = "female_obj_pos"
    FEMALE_OBJ_SIX_DOF = "female_obj_six_dof"
    DELTA_OBJS_POS = "delta_objs_pos"
    DELTA_OBJS_SIX_DOF = "delta_objs_six_dof"
    DELTA_POS_QUAD_PEG_LEFT = "delta_pos_quad_peg_left"
    DELTA_POS_QUAD_PEG_LEFT_CBRT = "delta_pos_quad_peg_left_cbrt"
    DELTA_POS_NIST_PEG_LEFT_CBRT = "delta_pos_nist_peg_left_cbrt"
    DELTA_POS_QUAD_PEG_FRONT_LEFT_CBRT = "delta_pos_quad_peg_front_left_cbrt"
    DELTA_POS_QUAD_PEG_LEFT_POLAR = "delta_pos_quad_peg_left_polar"
    DELTA_POS_QUAD_PEG_RIGHT = "delta_pos_quad_peg_right"
    DELTA_POS_QUAD_PEG_RIGHT_CBRT = "delta_pos_quad_peg_right_cbrt"
    DELTA_POS_NIST_PEG_RIGHT_CBRT = "delta_pos_nist_peg_right_cbrt"
    DELTA_POS_QUAD_PEG_FRONT_RIGHT_CBRT = "delta_pos_quad_peg_front_right_cbrt"
    DELTA_POS_QUAD_PEG_RIGHT_POLAR = "delta_pos_quad_peg_right_polar"
    DUAL_PEG_DELTA_POSITIONS = "dual_peg_delta_positions"
    SIX_DOF = "six_dof"
    QUAT = "quat"
    EULER = "euler"
    TARGET_SIX_DOF = "target_six_dof"
    DELTA_TARGET_SIX_DOF = "delta_target_six_dof"
    TARGET_QUAT = "target_quat"
    DELTA_TARGET_QUAT = "delta_target_quat"
    TARGET_EULER = "target_euler"
    GRIP_FORCE = "grip_force"
    GRIP_FORCE_EWMA = "grip_force_ewma"
    GRIP_TORQUE = "grip_torque"
    GRIP_TORQUE_EWMA = "grip_torque_ewma"
    POSITION_DIFF_NORM = "position_diff_norm"
    BASE_ANGLE = "base_angle"


def get_action_group_dim(action_group):
    action_group_dims = {
        ActionGroup.SPEED: 1,
        ActionGroup.DELTA_STEER: 1,
        ActionGroup.DELTA_POSITION: 3,
        ActionGroup.DELTA_SIX_DOF: 6,
        ActionGroup.DELTA_QUAT: 4,
        ActionGroup.DELTA_EULER: 3,
    }

    return action_group_dims[action_group]


def get_observation_group_dim(obs_group):
    obs_group_dims = {
        ObservationGroup.POSITION_2D: 2,
        ObservationGroup.YAW: 1,
        ObservationGroup.POSITION: 3,
        ObservationGroup.DELTA_POSITION_POLAR: len(cart2polar([0.1, 0.1, 0.1])),
        ObservationGroup.DELTA_TARGET_POS: 3,
        ObservationGroup.MALE_OBJ_POS: 3,
        ObservationGroup.MALE_OBJ_SIX_DOF: 6,
        ObservationGroup.FEMALE_OBJ_POS: 3,
        ObservationGroup.FEMALE_OBJ_SIX_DOF: 6,
        ObservationGroup.DELTA_OBJS_POS: 3,
        ObservationGroup.DELTA_OBJS_SIX_DOF: 6,
        ObservationGroup.DELTA_POS_QUAD_PEG_LEFT: 3,
        ObservationGroup.DELTA_POS_QUAD_PEG_LEFT_CBRT: 3,
        ObservationGroup.DELTA_POS_NIST_PEG_LEFT_CBRT: 3,
        ObservationGroup.DELTA_POS_QUAD_PEG_FRONT_LEFT_CBRT: 3,
        ObservationGroup.DELTA_POS_QUAD_PEG_LEFT_POLAR: len(cart2polar([0.1, 0.1, 0.1])),
        ObservationGroup.DELTA_POS_QUAD_PEG_RIGHT: 3,
        ObservationGroup.DELTA_POS_QUAD_PEG_RIGHT_CBRT: 3,
        ObservationGroup.DELTA_POS_NIST_PEG_RIGHT_CBRT: 3,
        ObservationGroup.DELTA_POS_QUAD_PEG_FRONT_RIGHT_CBRT: 3,
        ObservationGroup.DELTA_POS_QUAD_PEG_RIGHT_POLAR: len(cart2polar([0.1, 0.1, 0.1])),
        ObservationGroup.DUAL_PEG_DELTA_POSITIONS: 6,
        ObservationGroup.SIX_DOF: 6,
        ObservationGroup.QUAT: 4,
        ObservationGroup.EULER: 3,
        ObservationGroup.TARGET_SIX_DOF: 6,
        ObservationGroup.DELTA_TARGET_SIX_DOF: 6,
        ObservationGroup.TARGET_QUAT: 4,
        ObservationGroup.DELTA_TARGET_QUAT: 4,
        ObservationGroup.TARGET_EULER: 3,
        ObservationGroup.GRIP_FORCE: 3,
        ObservationGroup.GRIP_FORCE_EWMA: 3,
        ObservationGroup.GRIP_TORQUE: 3,
        ObservationGroup.GRIP_TORQUE_EWMA: 3,
        ObservationGroup.POSITION_DIFF_NORM: 1,
        ObservationGroup.BASE_ANGLE: 1,
    }
    return obs_group_dims[obs_group]
