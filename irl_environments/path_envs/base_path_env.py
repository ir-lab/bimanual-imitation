from abc import ABC, abstractmethod

import numpy as np


class PathEnv(ABC):
    def __init__(self):
        self.__reset_success = False
        self.expert_target_idx = 1
        # out of bounds penalty
        self.__ob_penalty = -1

    def __obs(self):
        obs = self.exec_obs(self.mj_obs_func)
        return obs

    def __reward(self):
        reward = self.exec_reward(self.mj_reward_func)
        return reward

    def __is_done(self, time):
        done = self.exec_done(time, self.mj_done_func)
        return done

    def reset(self, seed=None, options={}):
        # initialize_reset should load the new environment
        if self.debug_mode:
            print("################### start reset ###################")
        self.render_idx = 0
        self.__time = 0.0
        self.load_new_env(seed=seed)
        self.exec_reset(self.mj_reset_func)
        self.__reset_success = True
        self.__prev_obs = self.__obs()
        self.expert_target_idx = 1
        if self.debug_mode:
            print("################### end reset ###################")

        obs = np.array(self.__prev_obs, dtype=np.float32)
        return obs, {}

    def step(self, action):
        if self.debug_mode:
            print("################### start step ###################")
            print(f"Action: {action}")
        assert self.__reset_success, "env.reset() method must be called before running!"
        constrained_action = self.exec_constrain_action(action)
        if self.debug_mode:
            print(f"Constrained Action {constrained_action}")
        sim_err, ob, self.__time = self.exec_update_states(
            constrained_action, self.__time, self.mj_dt, self.mj_update_states_func
        )
        obs = self.__obs() if not sim_err else self.__prev_obs
        self.__prev_obs = obs
        reward = self.__reward() if not ob else self.__ob_penalty
        done = self.__is_done(self.__time)
        if done:
            reward += 100
        self.record_path_states(action, constrained_action, obs, reward, self.__time)
        self.expert_target_idx += 1
        if self.debug_mode:
            print("################### end step ###################")
            import pdb

            pdb.set_trace()

        # TODO: Should we enfore the env is always wrapped with gym?
        # Otherwise, the spec is not instantiated, and line below will fail
        if done or self.expert_target_idx == self.spec.max_episode_steps:
            self.maybe_export_gif()

        truncated = False
        info = {}
        obs = np.array(obs, dtype=np.float32)
        return obs, reward, done, truncated, info

    def run_sequence(self):
        done = False
        while not done:
            # Calc control input
            ctrl = self.run_pursuit()
            _, _, done, _, _ = self.step(ctrl)
        self.print_run_info()

    def print_run_info(self):
        ## Method should be filled in by the class which implements "record_path_states"
        # (abstract method not enforced here)
        pass

    def maybe_export_gif(self, *args, **kwargs):
        pass

    @abstractmethod
    def exec_constrain_action(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def exec_update_states(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def exec_obs(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def load_new_env(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def exec_reset(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def exec_reward(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def exec_done(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def run_pursuit(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def record_path_states(self, *args, **kwargs):
        raise NotImplementedError

    @property
    @abstractmethod
    def debug_mode(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def mj_dt(self):
        raise NotImplementedError

    def mj_reset_func(self, *args, **kwargs):
        raise NotImplementedError

    def mj_obs_func(self, *args, **kwargs):
        raise NotImplementedError

    def mj_update_states_func(self, *args, **kwargs):
        raise NotImplementedError

    def mj_reward_func(self, *args, **kwargs):
        raise NotImplementedError

    def mj_done_func(self, *args, **kwargs):
        raise NotImplementedError
