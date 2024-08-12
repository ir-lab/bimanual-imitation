import numpy as np
from irl_environments.core.utils import PursuitType
from irl_environments.path_envs.base_path_env import PathEnv
from irl_environments.runners.path_2D_runner import Runner2D
from irl_environments.core.track import Track2D
from irl_environments.core.state import State2D
from irl_environments.core.target_course import TargetCourse2D


class Path2DEnv(Runner2D, PathEnv):
    def __init__(self, expert_proto):
        Runner2D.__init__(self, proto_file=expert_proto)
        PathEnv.__init__(self)
        self.min_T = self.get_expert_param('min_T')
        self.dt = self.get_path_env_param('dt')
        # This environment only uses/supports one device: bicycle
        self.key_single = list(self.keys)[0]
        assert self.key_single == 'bicycle'

    def exec_reset(self, mj_reset_func):
        track = self.tracks[self.key_single]
        self.target_idx, _ = track.course.search_target_index(track.state)

    def exec_constrain_action(self, action):
        vi, di = action
        track: Track2D = self.tracks[self.key_single]
        if track.enforce_velocity_bounds:
            vel_bounds = track.velocity_bounds ; assert vel_bounds is not None
            vi = min(max(vi, vel_bounds[0]), vel_bounds[1])
        if track.enforce_delta_bounds:
            delta_bounds = track.delta_bounds ; assert delta_bounds is not None
            di = min(max(di, delta_bounds[0]), delta_bounds[1])
        return np.array([vi, di])

    def exec_update_states(self, action, time, dt, sim_step_func):
        vi, di = action
        track: Track2D = self.tracks[self.key_single]
        track.state.update(vi, di, dt)
        time += dt
        return False, False, time

    def exec_obs(self, mj_obs_func):
        state: State2D = self.tracks[self.key_single].state
        return np.array([state.x, state.y, state.yaw])

    def exec_reward(self, mj_reward_func):
        track: Track2D = self.tracks[self.key_single]
        state: State2D = track.state
        time_penalty = self.get_path_env_param('reward_time_penalty')
        pos_scale = self.get_path_env_param('reward_pos_scale')
        closest_x_idx = np.argmin(np.abs(track.course.cx - state.x)).flat[0]
        expected_y_value = track.course.cy[closest_x_idx]
        return np.exp(-pos_scale*(state.y - expected_y_value)**2) + time_penalty

    def exec_done(self, time, mj_done_func):
        pos_tol = self.get_path_env_param('done_pos_tol')
        yaw_tol = self.get_path_env_param('done_yaw_tol')
        course: TargetCourse2D = self.tracks[self.key_single].course
        state: State2D = self.tracks[self.key_single].state
        reached_pt = np.hypot(course.cx[self.last_course_idx] - state.x, course.cy[self.last_course_idx] - state.y) < pos_tol

        last_dy = course.cy[self.last_course_idx] - course.cy[self.last_course_idx -1]
        last_dx = course.cx[self.last_course_idx] - course.cx[self.last_course_idx -1]
        target_end_yaw = np.arctan2(last_dy, last_dx)
        current_yaw = np.arctan2(np.sin(state.yaw), np.cos(state.yaw))
        reached_yaw = np.abs(current_yaw - target_end_yaw) < yaw_tol
        done = reached_pt and reached_yaw and (time > self.min_T)
        return done

    def __pure_pursuit(self):
        track: Track2D = self.tracks[self.key_single]
        course: TargetCourse2D = track.course
        state: State2D = track.state

        target_idx, lookahead = course.search_target_index(state)

        tx = course.cx[target_idx]
        ty = course.cy[target_idx]
        alpha = np.arctan2(ty - state.rear_y, tx - state.rear_x) - state.yaw
        delta = np.arctan2(2.0 * state.wheelbase * np.sin(alpha) / lookahead, 1.0)
        a = track.expert_target_speed - state.v
        v = state.v + a
        return np.array([v, delta])

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
