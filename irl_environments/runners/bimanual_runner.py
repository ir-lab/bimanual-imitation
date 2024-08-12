from collections import OrderedDict

import numpy as np
from transforms3d.euler import euler2quat, quat2euler

from irl_control.utils.target import Target
from irl_environments.core.state import StateMujoco3D
from irl_environments.core.target_course import TargetCourseMujoco3D
from irl_environments.core.track import TrackMujoco3D
from irl_environments.core.utils import resample_by_interpolation
from irl_environments.runners.base_runner import Runner


class BimanualRunner(Runner):

    @staticmethod
    def get_all_devices():
        return ["ur5left", "ur5right"]

    @staticmethod
    def get_track_type():
        return TrackMujoco3D

    def get_initial_targets(self):
        targets: "OrderedDict[str, Target]" = OrderedDict()
        for device_name in self.get_all_devices():
            if device_name in self.keys:
                track = self.tracks[device_name]
                x, y, z = track.course.pos[0]
                qw, qx, qy, qz = track.course.quat[0]

                x_noise = track.device_config["x_noise_init"]
                y_noise = track.device_config["y_noise_init"]
                z_noise = track.device_config["z_noise_init"]

                yaw_noise_deg = track.device_config["yaw_noise_deg_init"]
                pitch_noise_deg = track.device_config["pitch_noise_deg_init"]
                roll_noise_deg = track.device_config["roll_noise_deg_init"]

                if x_noise is not None:
                    x += np.random.uniform(low=x_noise[0], high=x_noise[1])
                if y_noise is not None:
                    y += np.random.uniform(low=y_noise[0], high=y_noise[1])
                if z_noise is not None:
                    z += np.random.uniform(low=z_noise[0], high=z_noise[1])

                roll, pitch, yaw = quat2euler([qw, qx, qy, qz])
                if yaw_noise_deg is not None:
                    yaw += np.deg2rad(
                        np.random.uniform(low=yaw_noise_deg[0], high=yaw_noise_deg[1])
                    )
                if pitch_noise_deg is not None:
                    pitch += np.deg2rad(
                        np.random.uniform(low=pitch_noise_deg[0], high=pitch_noise_deg[1])
                    )
                if roll_noise_deg is not None:
                    roll += np.deg2rad(
                        np.random.uniform(low=roll_noise_deg[0], high=roll_noise_deg[1])
                    )

                qw, qx, qy, qz = euler2quat(roll, pitch, yaw)
                track.state.reset(x, y, z, qw, qx, qy, qz, 0)
                targets[device_name] = Target(
                    [track.state.x, track.state.y, track.state.z, 0, 0, 0]
                )
                if len(set(track.action_space).intersection(track.orientation_groups)) > 0:
                    targets[device_name].set_quat(track.state.quat)
                else:
                    targets[device_name].set_quat(self.default_start_pt[device_name][3:])
            else:
                targets[device_name] = Target()
                targets[device_name].set_xyz(self.default_start_pt[device_name][:3])
                targets[device_name].set_quat(self.default_start_pt[device_name][3:])

        return targets

    def get_grip_ctrl(self, grip_state: str):
        grip_ctrl = []
        for key in self.keys:
            track = self.tracks[key]
            if grip_state == "reset":
                grip_type = track.gripper_reset_action
            elif grip_state == "step":
                grip_type = track.gripper_step_action
            else:
                raise ValueError
            gc = (track.gripper_idx, track.grip_forces[grip_type])
            grip_ctrl.append(gc)
        return grip_ctrl

    def get_course_from_expert(
        self, device_slice, lookahead_gain, lookahead_dist, device_name, resample
    ):
        ex_pos = device_slice[:, :3]
        ex_quat = device_slice[:, 3:7]
        if resample:
            print(f"Resampling to {self.expert_num_samples} for {device_name}!")
            cx = resample_by_interpolation(ex_pos[:, 0], self.expert_num_samples)
            cy = resample_by_interpolation(ex_pos[:, 1], self.expert_num_samples)
            cz = resample_by_interpolation(ex_pos[:, 2], self.expert_num_samples)
            cqw = resample_by_interpolation(ex_quat[:, 0], self.expert_num_samples)
            cqx = resample_by_interpolation(ex_quat[:, 1], self.expert_num_samples)
            cqy = resample_by_interpolation(ex_quat[:, 2], self.expert_num_samples)
            cqz = resample_by_interpolation(ex_quat[:, 3], self.expert_num_samples)
        else:
            cx = ex_pos[:, 0]
            cy = ex_pos[:, 1]
            cz = ex_pos[:, 2]
            cqw = ex_quat[:, 0]
            cqx = ex_quat[:, 1]
            cqy = ex_quat[:, 2]
            cqz = ex_quat[:, 3]
        course = TargetCourseMujoco3D(
            cx, cy, cz, cqw, cqx, cqy, cqz, lookahead_gain, lookahead_dist, self.last_course_idx
        )
        return course

    def get_initial_state(self, device_config: dict, course: TargetCourseMujoco3D):
        assert isinstance(course, TargetCourseMujoco3D)
        x, y, z = course.pos[0]
        qw, qx, qy, qz = course.quat[0]
        state = StateMujoco3D(x, y, z, qw, qx, qy, qz, 0.0)
        return state
