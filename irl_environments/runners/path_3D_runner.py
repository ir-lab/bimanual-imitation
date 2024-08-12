import numpy as np

from irl_environments.core.state import State3D
from irl_environments.core.target_course import TargetCourse3D
from irl_environments.core.track import Track3D
from irl_environments.runners.base_runner import Runner


class Runner3D(Runner):
    def __init__(self, *args, **kwargs):
        Runner.__init__(self, *args, **kwargs)

    @staticmethod
    def get_all_devices():
        return ["point_mass"]

    @staticmethod
    def get_track_type():
        return Track3D

    # For this demo, we don't need a proto to define the track
    # Return None, since "get_course_from_expert" will not use the slice and default_pt isn't used
    def get_slice_and_default_pt(self, ex_obs_T_Do, device_name):
        assert device_name == "point_mass"
        return None, None

    def get_course_from_expert(
        self, device_slice, lookahead_gain, lookahead_dist, device_name, resample
    ):
        num_samples = self.get_expert_param("num_samples")
        initial_x = 0
        cx = np.linspace(initial_x, 2, num_samples)
        cy_shift = np.random.uniform(low=-0.3, high=0.3)
        cy_func = lambda x: np.sin(5 * x) * x + cy_shift
        cy = np.array([cy_func(ix) for ix in cx])
        cz_func = lambda x: -1 * np.log(x - cx[0] + 1) if x > cx[0] else 0
        cz = np.array([cz_func(ix) for ix in cx])
        course = TargetCourse3D(cx, cy, cz, lookahead_gain, lookahead_dist, self.last_course_idx)
        return course

    def get_initial_state(self, device_config: dict, course: TargetCourse3D):
        assert isinstance(course, TargetCourse3D)
        x, y, z = course.pos[0]
        x_noise = device_config["x_noise_init"]
        y_noise = device_config["y_noise_init"]
        z_noise = device_config["z_noise_init"]
        if x_noise is not None:
            x += np.random.uniform(low=x_noise[0], high=x_noise[1])
        if y_noise is not None:
            y += np.random.uniform(low=y_noise[0], high=y_noise[1])
        if z_noise is not None:
            z += np.random.uniform(low=z_noise[0], high=z_noise[1])
        state = State3D(x=x, y=y, z=z, v=0.0)
        return state
