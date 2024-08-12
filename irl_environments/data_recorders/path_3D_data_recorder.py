from abc import abstractmethod

import matplotlib.pyplot as plt

from irl_environments.data_recorders.base_data_recorder import DataRecorder


class Path3DDataRecorder(DataRecorder):
    def __init__(self, render):
        DataRecorder.__init__(self)
        self.__state_x_hist = []
        self.__state_y_hist = []
        self.__state_z_hist = []
        self._render = render
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(projection="3d")
        self.ax.view_init(elev=21, azim=-70, roll=0)
        # self.ax.view_init(elev=21, azim=-120, roll=0)
        self.ax.set_title("Expert Demonstration")
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.set_zlabel("Z (m)")

    @abstractmethod
    def get_course_idx_and_state(self):
        raise NotImplementedError

    def record_path_states(self, action, constrained_action, observation, reward, time):
        self._action_hist.append(action)
        self._observation_hist.append(observation)
        self._reward_hist.append(reward)
        self._env_time = time

        if self._render:
            course, target_idx, state = self.get_course_idx_and_state()

            self.__state_x_hist.append(state.x)
            self.__state_y_hist.append(state.y)
            self.__state_z_hist.append(state.z)

            self.ax.cla()
            self.ax.plot(course.cx, course.cy, course.cz, "-r", label="course")
            self.ax.plot(
                self.__state_x_hist,
                self.__state_y_hist,
                self.__state_z_hist,
                "-b",
                label="trajectory",
            )
            self.ax.plot(
                course.cx[target_idx],
                course.cy[target_idx],
                course.cz[target_idx],
                "xg",
                label="target",
            )
            self.ax.axis("equal")
            self.ax.grid(True)
            plt.pause(0.001)
