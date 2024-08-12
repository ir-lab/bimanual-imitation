from typing import List

import numpy as np
from transforms3d.euler import euler2quat, quat2euler


class Target:
    """
    The Target class holds a target vector for both orientation (quaternion) and position (xyz)
    NOTE: Quat is stored as w, x, y, z
    """

    def __init__(self, xyz_abg: List = np.zeros(6), xyz_abg_vel: List = np.zeros(6)):
        assert len(xyz_abg) == 6 and len(xyz_abg_vel) == 6
        assert np.all([isinstance(x, float) or isinstance(x, int) for x in xyz_abg])
        self.__xyz = np.array(xyz_abg)[:3]
        self.__xyz_vel = np.array(xyz_abg_vel)[:3]
        self.__quat = np.array(euler2quat(*xyz_abg[3:]))
        self.__quat_vel = np.array(euler2quat(*xyz_abg_vel[3:]))
        self.active = True

    def get_xyz(self):
        return self.__xyz

    def get_xyz_vel(self):
        return self.__xyz_vel

    def get_quat(self):
        return self.__quat

    def get_quat_vel(self):
        return np.asarray(self.__quat_vel)

    def get_abg(self):
        return np.asarray(quat2euler(self.__quat))

    def get_abg_vel(self):
        return np.asarray(quat2euler(self.__quat_vel))

    def set_xyz(self, xyz):
        assert len(xyz) == 3
        self.__xyz = np.asarray(xyz)

    def set_xyz_vel(self, xyz_vel):
        assert len(xyz_vel) == 3
        self.__xyz_vel = np.asarray(xyz_vel)

    def set_quat(self, quat):
        assert len(quat) == 4
        self.__quat = np.asarray(quat)

    def set_quat_vel(self, quat_vel):
        assert len(quat_vel) == 4
        self.__quat_vel = np.asarray(quat_vel)

    def set_abg(self, abg):
        assert len(abg) == 3
        self.__quat = np.asarray(euler2quat(*abg))

    def set_abg_vel(self, abg_vel):
        assert len(abg_vel) == 3
        self.__quat_vel = np.asarray(euler2quat(*abg_vel))

    def set_all_quat(self, xyz, quat):
        assert len(xyz) == 3 and len(quat) == 4
        self.__xyz = np.asarray(xyz)
        self.__quat = np.asarray(quat)

    def set_all_abg(self, xyz, abg):
        assert len(xyz) == 3 and len(abg) == 3
        self.__xyz = np.asarray(xyz)
        self.__quat = np.asarray(euler2quat(*abg))

    @property
    def x(self):
        return self.__xyz[0]

    @x.setter
    def x(self, val):
        tmp = self.get_xyz()
        tmp[0] = val
        self.set_xyz(tmp)

    @property
    def y(self):
        return self.__xyz[1]

    @y.setter
    def y(self, val):
        tmp = self.get_xyz()
        tmp[1] = val
        self.set_xyz(tmp)

    @property
    def z(self):
        return self.__xyz[2]

    @z.setter
    def y(self, val):
        tmp = self.get_xyz()
        tmp[2] = val
        self.set_xyz(tmp)

    def check_ob(self, x_bounds, y_bounds, z_bounds, set=False):
        ob = False
        if self.x < x_bounds[0]:
            if set:
                self.x = x_bounds[0]
            ob = True

        if self.x > x_bounds[1]:
            if set:
                self.x = x_bounds[1]
            ob = True

        if self.y < y_bounds[0]:
            if set:
                self.y = y_bounds[0]
            ob = True

        if self.y > y_bounds[1]:
            if set:
                self.y = y_bounds[1]
            ob = True

        if self.z < z_bounds[0]:
            if set:
                self.z = z_bounds[0]
            ob = True

        if self.z > z_bounds[1]:
            if set:
                self.z = z_bounds[1]
            ob = True

        return ob

    @property
    def pos(self):
        return self.get_xyz()

    @property
    def quat(self):
        return self.get_quat()
