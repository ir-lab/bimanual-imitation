from abc import ABC, abstractmethod

import numpy as np
from irl_environments.core.state import State, State2D, State3D, StateMujoco3D


class TargetCourse(ABC):
    def __init__(self, cx, cy, lookahead_gain, lookahead_dist, last_course_idx):
        self.__cx = cx
        self.__cy = cy
        self.__lookahead_gain = lookahead_gain  # look forward gain
        self.__lookahead_dist = lookahead_dist  # [m] look-ahead distance
        self.__last_course_idx = last_course_idx
        self.__old_nearest_point_index = None
        self.__target_idx = 0

    def search_target_index(self, state: State):
        assert isinstance(state, self.state_instance)
        pind = self.__target_idx

        if self.__old_nearest_point_index is None:
            distances = [state.calc_distance(*pos_pt) for pos_pt in self.pos]
            ind = np.argmin(distances)
            self.__old_nearest_point_index = ind
        else:
            ind = self.__old_nearest_point_index
            distance_this_index = state.calc_distance(*self.pos[ind])
            while True:
                distance_next_index = state.calc_distance(*self.pos[ind + 1])
                if distance_this_index < distance_next_index or (ind >= self.__last_course_idx):
                    break
                ind = ind + 1 if ind < self.__last_course_idx else ind
                distance_this_index = distance_next_index
            self.__old_nearest_point_index = ind

        lookahead = (
            self.__lookahead_gain * state.v + self.__lookahead_dist
        )  # update look ahead distance

        # search look ahead target point index
        while lookahead > state.calc_distance(*self.pos[ind]):
            if ind >= self.__last_course_idx:
                break  # not exceed goal
            ind += 1

        if self.__target_idx >= ind:
            ind = pind

        if ind >= self.__last_course_idx:
            ind = self.__last_course_idx

        self.__target_idx = ind
        return ind, lookahead

    @property
    def cx(self):
        return self.__cx

    @property
    def cy(self):
        return self.__cy

    @property
    def target_idx(self):
        return self.__target_idx

    # This should be a matrix (N x D) of the positions on the target course
    @property
    @abstractmethod
    def pos(self):
        raise NotImplementedError

    # Used in search_target_index to verify we are receiving the correct state instance
    @property
    @abstractmethod
    def state_instance(self):
        raise NotImplementedError


class TargetCourse2D(TargetCourse):
    def __init__(self, cx, cy, lookahead_gain, lookahead_dist, last_course_idx):
        super().__init__(cx, cy, lookahead_gain, lookahead_dist, last_course_idx)
        self.__pos = np.vstack([self.cx, self.cy]).T
        assert self.pos.shape[1] == 2

    @property
    def pos(self):
        return self.__pos

    @property
    def state_instance(self):
        return State2D


class TargetCourse3D(TargetCourse):
    def __init__(self, cx, cy, cz, lookahead_gain, lookahead_dist, last_course_idx):
        super().__init__(cx, cy, lookahead_gain, lookahead_dist, last_course_idx)
        self.__cz = cz
        self.__pos = np.vstack([self.cx, self.cy, self.cz]).T
        assert self.__pos.shape[1] == 3

    @property
    def cz(self):
        return self.__cz

    @property
    def pos(self):
        return self.__pos

    @property
    def state_instance(self):
        return State3D


class TargetCourseMujoco3D(TargetCourse3D):
    def __init__(
        self, cx, cy, cz, cqw, cqx, cqy, cqz, lookahead_gain, lookahead_dist, last_course_idx
    ):
        super().__init__(cx, cy, cz, lookahead_gain, lookahead_dist, last_course_idx)
        self.__cqw = cqw
        self.__cqx = cqx
        self.__cqy = cqy
        self.__cqz = cqz
        self.__quat = np.vstack([self.cqw, self.cqx, self.cqy, self.cqz]).T
        assert self.__quat.shape[1] == 4

    @property
    def cqw(self):
        return self.__cqw

    @property
    def cqx(self):
        return self.__cqx

    @property
    def cqy(self):
        return self.__cqy

    @property
    def cqz(self):
        return self.__cqz

    @property
    def quat(self):
        return self.__quat

    @property
    def state_instance(self):
        return StateMujoco3D
