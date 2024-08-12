import datetime
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from irl_data import proto_logger
from irl_data.constants import EXPERT_TRAJS_DIR
from irl_data.trajectory import TrajBatch, Trajectory


class DataRecorder(ABC):
    def __init__(self):
        self._action_hist = []
        self._observation_hist = []
        self._reward_hist = []
        self._env_time = None

        self.__proto_export_suffix = None
        self.__record_proto = False

        self._gif_export_dir = None
        self._gif_export_prefix = None
        self._gif_export_suffix = None
        self._record_gif = False

    def set_proto_recording(self, export_suffix=None):
        assert self.__record_proto is False
        assert len(self._action_hist) == len(self._observation_hist) == len(self._reward_hist) == 0
        self.__proto_export_suffix = str(export_suffix)
        self.__record_proto = True

    def get_run_traj(self):
        assert (
            len(self._action_hist) > 0
            and len(self._observation_hist) > 0
            and len(self._reward_hist) > 0
        )
        assert len(self._action_hist) == len(self._observation_hist) == len(self._reward_hist)

        obs_T_Do = np.asarray(self._observation_hist)
        obsfeat_T_Df = np.ones((obs_T_Do.shape[0], 1)) * np.nan
        adist_T_Pa = np.ones((obs_T_Do.shape[0], 1)) * np.nan
        a_T_Da = np.asarray(self._action_hist)
        r_T = np.asarray(self._reward_hist)

        print(
            f"Action shape: {a_T_Da.shape[1]}, Observation shape: {obs_T_Do.shape[1]}, Length: {len(r_T)}"
        )

        run_traj = Trajectory(obs_T_Do, obsfeat_T_Df, adist_T_Pa, a_T_Da, r_T)
        return run_traj

    def export_proto_recording(self):
        # This function is no longer used, in favor of get_run_traj
        # which batch exports all trajs after the runs are completed
        assert self.__record_proto is True

        traj = self.get_run_traj()
        trajbatch = TrajBatch.FromTrajs([traj])

        if self.__proto_export_suffix is None:
            self.__proto_export_suffix = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

        fname = EXPERT_TRAJS_DIR / f"storage/{self.env_name}_{self.__proto_export_suffix}.proto"

        proto_logger.export_trajs(trajbatch, fname)

    def set_gif_recording(self, export_dir, export_prefix, export_suffix):
        export_dir = Path(export_dir)
        assert self._record_gif is False
        assert export_dir.exists(), f"{export_dir} does not exist"
        assert isinstance(export_prefix, str) and isinstance(export_suffix, str)
        self._gif_export_dir = export_dir
        self._gif_export_prefix = export_prefix
        self._gif_export_suffix = export_suffix
        self._record_gif = True

    def print_run_info(self):
        print(
            f"Reward: {sum(self._reward_hist)}, Length: {len(self._reward_hist)}, Time: {self._env_time}"
        )

    @property
    @abstractmethod
    def env_name(self):
        raise NotImplementedError
