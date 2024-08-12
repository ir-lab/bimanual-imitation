import numpy as np

from irl_environments.core.state import State2D
from irl_environments.core.target_course import TargetCourse2D
from irl_environments.core.track import Track2D
from irl_environments.runners.base_runner import Runner


class Runner2D(Runner):

    @staticmethod
    def get_all_devices():
        return ["bicycle"]

    @staticmethod
    def get_track_type():
        return Track2D

    def get_track(
        self,
        course: TargetCourse2D,
        state: State2D,
        device_name: str,
        device_config: dict,
        spaces: dict,
    ):
        return Track2D(course, state, device_name, device_config, spaces)

    # For this demo, we don't need a proto to define the track
    # Return None, since "get_course_from_expert" will not use the slice and default_pt isn't used
    def get_slice_and_default_pt(self, ex_obs_T_Do, device_name):
        assert device_name == "bicycle"
        return None, None

    def get_course_from_expert(
        self, device_slice, lookahead_gain, lookahead_dist, device_name, resample
    ):
        initial_x = 0
        cx = np.linspace(initial_x, 2, self.expert_num_samples)
        cy_shift = np.random.uniform(low=-0.3, high=0.3)
        cy_func = lambda x: np.sin(5 * x) * x + cy_shift
        cy = np.array([cy_func(ix) for ix in cx])
        course = TargetCourse2D(cx, cy, lookahead_gain, lookahead_dist, self.last_course_idx)
        return course

    def get_initial_state(self, device_config: dict, course: TargetCourse2D):
        assert isinstance(course, TargetCourse2D)
        x, y = course.pos[0]
        x_noise = device_config["x_noise_init"]
        y_noise = device_config["y_noise_init"]
        if x_noise is not None:
            x += np.random.uniform(low=x_noise[0], high=x_noise[1])
        if y_noise is not None:
            y += np.random.uniform(low=y_noise[0], high=y_noise[1])
        wheelbase = device_config["wheelbase"]
        state = State2D(x=x, y=y, yaw=0.0, v=0.0, wheelbase=wheelbase)
        return state
