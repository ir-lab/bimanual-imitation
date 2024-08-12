import numpy as np
from abc import ABC, abstractmethod


class State(ABC):
    def __init__(self, x, y, v):
        self._x = x
        self._y = y
        self._v = v

    @abstractmethod
    def reset(self, *args):
        raise NotImplementedError

    @abstractmethod
    def update(self, *args):
        raise NotImplementedError

    @abstractmethod
    def calc_distance(self, *args):
        raise NotImplementedError

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def v(self):
        return self._v

    @property
    @abstractmethod
    def pos(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def quat(self):
        raise NotImplementedError

class State2D(State):
    def __init__(self, x, y, yaw, v, wheelbase):
        super().__init__(x,y,v)
        self._wheelbase = wheelbase
        self._yaw = yaw
        self._rear_x = self.x - ((self.wheelbase / 2) * np.cos(self.yaw))
        self._rear_y = self.y - ((self.wheelbase / 2) * np.sin(self.yaw))

    def reset(self, *args, **kwargs):
        raise NotImplementedError

    def update(self, v, delta, dt):
        self._v = v
        self._x += self.v * np.cos(self.yaw) * dt
        self._y += self.v * np.sin(self.yaw) * dt
        yaw_unconstrained = self.yaw + self.v / self.wheelbase * np.tan(delta) * dt
        self._yaw = np.arctan2(np.sin(yaw_unconstrained), np.cos(yaw_unconstrained))
        self._rear_x = self.x - ((self.wheelbase / 2) * np.cos(self.yaw))
        self._rear_y = self.y - ((self.wheelbase / 2) * np.sin(self.yaw))

    def calc_distance(self, point_x, point_y):
        dx = self.rear_x - point_x
        dy = self.rear_y - point_y
        return np.linalg.norm([dx, dy])

    @property
    def yaw(self):
        return self._yaw

    @property
    def wheelbase(self):
        return self._wheelbase

    @property
    def rear_x(self):
        return self._rear_x

    @property
    def rear_y(self):
        return self._rear_y

    @property
    def pos(self):
        return np.array([self.x, self.y])

    @property
    def quat(self):
        print("State 2D contains only position! Did you mean to get the yaw value?")
        raise NotImplementedError

class State3D(State):
    def __init__(self, x, y, z, v):
        super().__init__(x, y, v)
        self._z = z

    def reset(self, *args, **kwargs):
        raise NotImplementedError

    def update(self, dx, dy, dz, dt):
        self._x += dx*dt
        self._y += dy*dt
        self._z += dz*dt
        self._v = np.linalg.norm([dx, dy, dz])

    def calc_distance(self, point_x, point_y, point_z):
        return np.linalg.norm([self.x - point_x, self.y - point_y, self.z - point_z])

    @property
    def z(self):
        return self._z

    @property
    def pos(self):
        return np.array([self.x, self.y, self.z])

    @property
    def quat(self):
        print("State 3D contains only position! Did you mean to use State3DQuat?")
        raise NotImplementedError

class State3DQuat(State3D):
    def __init__(self, x, y, z, qw, qx, qy, qz, v):
        super().__init__(x,y,z,v)
        self._qw = qw
        self._qx = qx
        self._qy = qy
        self._qz = qz

    @property
    def qw(self):
        return self._qw

    @property
    def qx(self):
        return self._qx

    @property
    def qy(self):
        return self._qy

    @property
    def qz(self):
        return self._qz

    @property
    def quat(self):
        return np.array([self.qw, self.qx, self.qy, self.qz])


class StateMujoco3D(State3DQuat):
    def __init__(self, x, y, z, qw, qx, qy, qz, v):
        super().__init__(x, y, z, qw, qx, qy, qz, v)

    def reset(self, init_x, init_y, init_z, init_qw, init_qx, init_qy, init_qz, v):
        self._x = init_x
        self._y = init_y
        self._z = init_z
        self._qw = init_qw
        self._qx = init_qx
        self._qy = init_qy
        self._qz = init_qz
        self._v = v

    def update(self, true_x, true_y, true_z, true_qw, true_qx, true_qy, true_qz, v):
        self._x = true_x
        self._y = true_y
        self._z = true_z
        self._qw = true_qw
        self._qx = true_qx
        self._qy = true_qy
        self._qz = true_qz
        self._v = v