import numpy as np
from transforms3d.utils import normalized_vector as normed

from irl_environments.core.state import State3D
from irl_environments.core.target_course import TargetCourse3D
from irl_environments.core.track import Track3D
from irl_environments.core.utils import PursuitType
from irl_environments.path_envs.base_path_env import PathEnv
from irl_environments.runners.path_3D_runner import Runner3D


class Path3DEnv(Runner3D, PathEnv):
    def __init__(self, expert_proto):
        Runner3D.__init__(self, proto_file=expert_proto)
        PathEnv.__init__(self)
        self.min_T = self.get_expert_param("min_T")
        self.dt = self.get_path_env_param("dt")
        # This environment only uses/supports one device: point_mass
        self.key_single = list(self.keys)[0]
        assert self.key_single == "point_mass"

    def exec_reset(self, mj_reset_func):
        track = self.tracks[self.key_single]
        self.target_idx, _ = track.course.search_target_index(track.state)

    def exec_constrain_action(self, action):
        dx, dy, dz = action
        track: Track3D = self.tracks[self.key_single]
        if track.enforce_delta_xyz_bounds:
            dx_bounds, dy_bounds, dz_bounds = track.delta_xyz_bounds
            assert dx_bounds is not None and dy_bounds is not None and dz_bounds is not None
            dx = min(max(dx, dx_bounds[0]), dx_bounds[1])
            dy = min(max(dy, dy_bounds[0]), dy_bounds[1])
            dz = min(max(dz, dz_bounds[0]), dz_bounds[1])
        return np.array([dx, dy, dz])

    def exec_update_states(self, action, time, dt, sim_step_func):
        dx, dy, dz = action
        track: Track3D = self.tracks[self.key_single]
        track.state.update(dx, dy, dz, dt)
        time += dt
        return False, False, time

    def exec_obs(self, mj_obs_func):
        state: State3D = self.tracks[self.key_single].state
        return np.array([state.x, state.y, state.z])

    def exec_reward(self, mj_reward_func):
        track: Track3D = self.tracks[self.key_single]
        state: State3D = track.state
        time_penalty = self.get_path_env_param("reward_time_penalty")
        pos_scale = self.get_path_env_param("reward_pos_scale")
        closest_x_idx = np.argmin(np.abs(track.course.cx - state.x)).flat[0]
        expected_y_value = track.course.cy[closest_x_idx]
        return np.exp(-pos_scale * (state.y - expected_y_value) ** 2) + time_penalty

    def exec_done(self, time, mj_done_func):
        pos_tol = self.get_path_env_param("done_pos_tol")
        course: TargetCourse3D = self.tracks[self.key_single].course
        state: State3D = self.tracks[self.key_single].state
        reached_pt = state.calc_distance(*course.pos[self.last_course_idx]) < pos_tol
        done = reached_pt and time > self.min_T
        return done

    def __pure_pursuit(self):
        track: Track3D = self.tracks[self.key_single]
        course: TargetCourse3D = track.course
        state: State3D = track.state

        target_idx, _ = course.search_target_index(state)
        dpos = normed(course.pos[target_idx] - state.pos) * track.expert_target_speed
        return dpos

    def run_pursuit(self):
        if self.pursuit_type == PursuitType.PURE_PURSUIT:
            ctrl = self.__pure_pursuit()
        else:
            raise ValueError
        return ctrl

    @property
    def mj_dt(self):
        return self.dt

    # This is required by the DataRecorder to animate the path 2D Environment
    def path_env_get_course_idx_and_state(self):
        track = self.tracks[self.key_single]
        return track.course, track.course.target_idx, track.state
