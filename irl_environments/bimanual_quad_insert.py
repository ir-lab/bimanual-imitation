import os
from abc import abstractmethod
from pathlib import Path

import gymnasium as gym
import numpy as np

from irl_data.constants import EXPERT_TRAJS_DIR, EXPERT_VAL_TRAJS_DIR
from irl_environments.constants import IRL_ENVIRONMENTS_BASE_DIR
from irl_environments.data_recorders.bimanual_data_recorder import BimanualDataRecorder, export_video
from irl_environments.path_envs.quad_insert_path_env import QuadInsertPathEnv
from irl_environments.runners.base_runner import get_gym_spaces


class BaseQuadInsert(BimanualDataRecorder, QuadInsertPathEnv):
    def __init__(
        self,
        proto_idx=None,
        render_mode="rgb_array",
        randomize_track_length=False,
        validation_set=False,
        pred_horizon=1,
        obs_horizon=1,
        action_horizon=1,
        verbose=False,
    ):
        if verbose:
            print(f"[BaseQuadInsert] Got: pred_horizon={pred_horizon}, obs_horizon={obs_horizon}")

        self.__raw_expert_dir_name = "raw_quad_insert"
        if validation_set:
            print("Using validation set!")
            self.__raw_expert_dir = EXPERT_VAL_TRAJS_DIR / self.__raw_expert_dir_name
        else:
            self.__raw_expert_dir = EXPERT_TRAJS_DIR / self.__raw_expert_dir_name

        self.__yaml_param_file = IRL_ENVIRONMENTS_BASE_DIR / f"param/{self.env_name}.yaml"
        self.__render_mode = render_mode
        self.__randomize_track_length = randomize_track_length
        self.__proto_idx = proto_idx
        self.observation_space, self.action_space = get_gym_spaces(self, pred_horizon, obs_horizon)
        self.__scene_file = "quad_insert.xml"
        self.__robot_config_file = "quad_insert.yaml"
        self.__debug_mode = False

    def load_new_env(self, seed=None):
        if seed is not None:
            raise NotImplementedError

        if self.__proto_idx is None:
            max_n = len(list(self.__raw_expert_dir.glob("*.proto")))
            proto_idx = np.random.randint(low=0, high=max_n)
        else:
            proto_idx = self.__proto_idx
            print(f"Using proto idx: {proto_idx}")

        self.proto_suffix = f"{proto_idx:04}"
        raw_expert_proto = (
            self.__raw_expert_dir / f"{self.__raw_expert_dir_name}_{self.proto_suffix}.proto"
        )

        QuadInsertPathEnv.__init__(
            self,
            raw_expert_proto,
            self.__scene_file,
            self.observation_space,
            self.action_space,
            self.__robot_config_file,
            self.__render_mode,
            self.__randomize_track_length,
        )
        BimanualDataRecorder.__init__(self)

    def get_mujoco_renders(self):
        return self._rendered_frames

    def get_slice_and_default_pt(self, ex_obs_T_Do, device_name):
        if device_name == "ur5left":
            device_pos = ex_obs_T_Do[:, :3]
            device_quat = ex_obs_T_Do[:, 3:7]
            device_slice = np.hstack([device_pos, device_quat])
            default_pt = np.concatenate([device_pos[0], device_quat[0]])
            return device_slice, default_pt
        elif device_name == "ur5right":
            device_pos = ex_obs_T_Do[:, 7:10]
            device_quat = ex_obs_T_Do[:, 10:14]
            device_slice = np.hstack([device_pos, device_quat])
            default_pt = np.concatenate([device_pos[0], device_quat[0]])
            return device_slice, default_pt
        else:
            print("device not found!")
            raise NotImplementedError

    def convert_raw_expert(self):
        self.reset()
        self.set_proto_recording(export_suffix=self.proto_suffix)
        self.run_sequence()
        run_traj = self.get_run_traj()
        return run_traj

    @property
    def debug_mode(self):
        return self.__debug_mode

    @property
    def yaml_param_file(self) -> Path:
        return self.__yaml_param_file

    @property
    def runtime_record_gif(self) -> bool:
        # IBC uses tf-agents, so the workaround for recording IBC gifs is to
        # set an environment variable before running the eval agent
        gif_loc = os.environ.get("IBC_EXPORT_GIF")
        return self._record_gif or (gif_loc is not None)

    def maybe_export_gif(self):
        # IBC uses tf-agents, so the workaround for recording IBC gifs is to
        # set an environment variable before running the eval agent
        video_path = os.environ.get("IBC_EXPORT_GIF")
        
        if video_path is not None:
            frames = self.get_mujoco_renders()
            export_video(frames, video_path)


    @property
    @abstractmethod
    def env_name(self):
        raise NotImplementedError


class QuadInserta0o0(BaseQuadInsert):

    @property
    def env_name(self):
        return "quad_insert_a0o0"


class QuadInsertaLoL(BaseQuadInsert):

    @property
    def env_name(self):
        return "quad_insert_aLoL"


class QuadInsertaMoM(BaseQuadInsert):

    @property
    def env_name(self):
        return "quad_insert_aMoM"


class QuadInsertChunking(BaseQuadInsert):
    def __init__(self, *args, **kwargs):
        # TODO: these are hardcoded for now
        self._pred_chunk_horizon = kwargs.pop("pred_horizon", 1)
        self._obs_chunk_horizon = kwargs.pop("obs_horizon", 1)
        self._action_chunk_horizon = kwargs.pop("action_horizon", 1)

        if "verbose" in kwargs and kwargs["verbose"]:
            print(
                f"[QuadInsertChunking] Got chunking: pred_horizon={self._pred_chunk_horizon},"
                + f"obs_horizon={self._obs_chunk_horizon}, action_horizon={self._action_chunk_horizon}"
            )

        self._single_horizon = bool(
            np.all(
                [
                    self._pred_chunk_horizon == 1,
                    self._obs_chunk_horizon == 1,
                    self._action_chunk_horizon == 1,
                ]
            )
        )

        BaseQuadInsert.__init__(
            self,
            *args,
            **kwargs,
            pred_horizon=self._pred_chunk_horizon,
            obs_horizon=self._obs_chunk_horizon,
        )

    def reset(self):
        obs, _ = super().reset()
        self._chunk_obs = [obs] * self._obs_chunk_horizon
        self._chunk_rewards = list()
        obs_chunk = np.stack(self._chunk_obs).flatten()
        return obs_chunk, {}

    def step(self, _action):
        full_action_chunk = _action.reshape(
            -1, self.action_space.shape[0] // self._pred_chunk_horizon
        )

        start = self._obs_chunk_horizon - 1
        end = start + self._action_chunk_horizon
        action_chunk = full_action_chunk[start:end, :]

        num_steps = 0
        truncated = None
        info = None
        for sub_action in action_chunk:
            # stepping env
            obs, reward, done, truncated, info = super().step(sub_action)
            # save observations
            self._chunk_obs.append(obs)
            # and reward/vis
            self._chunk_rewards.append(reward)

            num_steps += 1
            if done:
                break

        # if we break early, just return prior observations
        obs_chunk = np.stack(self._chunk_obs[-self._obs_chunk_horizon :]).flatten()
        reward_chunk = np.sum(self._chunk_rewards[-num_steps:])

        return obs_chunk, reward_chunk, done, truncated, info


class QuadInserta0o0Chunking(QuadInsertChunking):

    @property
    def env_name(self):
        return "quad_insert_a0o0"


class QuadInsertaLoLChunking(QuadInsertChunking):

    @property
    def env_name(self):
        return "quad_insert_aLoL"


class QuadInsertaMoMChunking(QuadInsertChunking):

    @property
    def env_name(self):
        return "quad_insert_aMoM"


if __name__ == "__main__":
    env = gym.make("quad_insert_a0o0", render_mode="human").unwrapped
    env.reset()
    # env.set_gif_recording(Path.cwd(), "test", "01")
    env.run_sequence()
    # env.export_gif_recording()
    # env.plot_trajs()
