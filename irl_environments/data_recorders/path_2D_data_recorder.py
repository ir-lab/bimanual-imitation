from abc import abstractmethod

import matplotlib.pyplot as plt

from irl_environments.data_recorders.base_data_recorder import DataRecorder


class Path2DDataRecorder(DataRecorder):
    def __init__(self, render_animation):
        DataRecorder.__init__(self)
        self.__state_x_hist = []
        self.__state_y_hist = []
        self.__render_animation = render_animation

    @abstractmethod
    def get_course_idx_and_state(self):
        raise NotImplementedError

    def record_path_states(self, action, constrained_action, observation, reward, time):
        self._action_hist.append(action)
        self._observation_hist.append(observation)
        self._reward_hist.append(reward)
        self._env_time = time

        if self.__render_animation:
            course, target_idx, state = self.get_course_idx_and_state()

            self.__state_x_hist.append(state.x)
            self.__state_y_hist.append(state.y)

            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect(
                "key_release_event", lambda event: [exit(0) if event.key == "escape" else None]
            )
            # self.plot_arrow(state.x, state.y, self.state.yaw)
            plt.plot(course.cx, course.cy, "-r", label="course")
            plt.plot(self.__state_x_hist, self.__state_y_hist, "-b", label="trajectory")
            plt.plot(course.cx[target_idx], course.cy[target_idx], "xg", label="target")
            plt.axis("equal")
            plt.grid(True)
            # plt.title("Speed[km/h]:" + str(self.state.v * 3.6)[:4])
            plt.pause(0.001)
