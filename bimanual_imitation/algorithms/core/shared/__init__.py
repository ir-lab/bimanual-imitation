import binascii
import multiprocessing
from collections import namedtuple
from time import sleep

import numpy as np

from bimanual_imitation.algorithms.configs import ALG
from bimanual_imitation.algorithms.core.shared import util
from irl_data.trajectory import TrajBatch, Trajectory


# State/action spaces
class Space(object):
    @property
    def storage_size(self):
        raise NotImplementedError

    @property
    def storage_type(self):
        raise NotImplementedError


class FiniteSpace(Space):
    def __init__(self, size):
        self._size = size

    @property
    def storage_size(self):
        return 1

    @property
    def storage_type(self):
        return int

    @property
    def size(self):
        return self._size


class ContinuousSpace(Space):
    def __init__(self, dim):
        self._dim = dim

    @property
    def storage_size(self):
        return self._dim

    @property
    def storage_type(self):
        return float

    @property
    def dim(self):
        return self._dim


# MDP stuff


class Simulation(object):
    def step(self, action):
        """
        Returns: reward
        """
        raise NotImplementedError

    @property
    def obs(self):
        """
        Get current observation. The caller must not assume that the contents of
        this array will never change, so this should usually be followed by a copy.

        Returns:
            numpy array
        """
        raise NotImplementedError

    @property
    def done(self):
        """
        Is this simulation done?

        Returns:
            boolean
        """
        raise NotImplementedError

    def draw(self):
        raise NotImplementedError


SimConfig = namedtuple("SimConfig", "min_num_trajs min_total_sa batch_size max_traj_len")


class MDP(object):
    """General MDP"""

    @property
    def obs_space(self):
        """Observation space"""
        raise NotImplementedError

    @property
    def action_space(self):
        """Action space"""
        raise NotImplementedError

    def new_sim(self, init_state=None):
        raise NotImplementedError

    def sim_single(
        self,
        policy_fn,
        obsfeat_fn,
        max_traj_len,
        init_state=None,
        alg=ALG.GAIL,
        dagger_action_beta=0.7,
        record_gif=False,
        gif_export_dir=None,
        gif_prefix="rollout",
        gif_export_suffix=None,
        dagger_eval=False,
    ):
        """Simulate a single trajectory"""

        assert alg in ALG
        if dagger_eval:
            assert alg == ALG.DAGGER

        obs, obsfeat, actions, actiondists, rewards = [], [], [], [], []

        sim = self.new_sim(init_state=init_state)
        obs.append(sim.reset()[None, ...])
        obsfeat.append(obsfeat_fn(obs[-1]))

        if record_gif:
            # must call this after performing env.reset()
            sim.env.unwrapped.set_gif_recording(gif_export_dir, gif_prefix, gif_export_suffix)

        if alg != ALG.DAGGER:
            # Normal simulation loop
            for step in range(max_traj_len):
                a, adist = policy_fn(obsfeat[-1])
                actions.append(a)
                actiondists.append(adist)
                rewards.append(sim.step(a[0, :]))

                if sim.done or (step == max_traj_len - 1):
                    break

                obs.append(sim.obs[None, ...])
                obsfeat.append(obsfeat_fn(obs[-1]))
        else:
            # Dagger simulation loop
            for step in range(max_traj_len):
                if dagger_eval:
                    # For evaluating DAgger, we must take the policy's action
                    expert_a, novice_a = policy_fn(obsfeat[-1], sim.env)
                    actions.append(novice_a)
                    actiondists.append([[-1, -1]])
                    rewards.append(sim.step(novice_a[0, :]))
                else:
                    # For training Dagger
                    expert_a, novice_a = policy_fn(obsfeat[-1], sim.env)
                    actions.append(expert_a)
                    actiondists.append([[-1, -1]])
                    action = dagger_action_beta * expert_a + (1.0 - dagger_action_beta) * novice_a
                    rewards.append(sim.step(action[0, :]))

                if sim.done or (step == max_traj_len - 1):
                    break

                obs.append(sim.obs[None, ...])
                obsfeat.append(obsfeat_fn(obs[-1]))

        if record_gif:
            sim.env.unwrapped.export_gif_recording()

        obs_T_Do = np.concatenate(obs)
        assert obs_T_Do.shape == (len(obs), self.obs_space.storage_size)
        obsfeat_T_Df = np.concatenate(obsfeat)
        assert obsfeat_T_Df.shape[0] == len(obs)
        adist_T_Pa = np.concatenate(actiondists)
        assert adist_T_Pa.ndim == 2 and adist_T_Pa.shape[0] == len(obs)
        a_T_Da = np.concatenate(actions)
        assert a_T_Da.shape == (len(obs), self.action_space.storage_size)
        r_T = np.asarray(rewards)
        assert r_T.shape == (len(obs),)
        return Trajectory(obs_T_Do, obsfeat_T_Df, adist_T_Pa, a_T_Da, r_T)

    def set_mkl(self):
        with set_mkl_threads(1):
            pool = multiprocessing.Pool(processes=2, maxtasksperchild=200)
            pool.close()
            pool.join()

    def sim_mp(
        self,
        policy_fn,
        obsfeat_fn,
        cfg,
        maxtasksperchild=200,
        alg=ALG.GAIL,
        dagger_action_beta=0.7,
        record_gif=False,
        gif_export_dir=None,
        gif_prefix="rollout",
        gif_export_suffix=None,
        dagger_eval=False,
        exact_num_trajs=False,
    ):
        """
        Multiprocessed simulation
        Not thread safe! But why would you want this to be thread safe anyway?
        """
        num_processes = (
            cfg.batch_size if cfg.batch_size is not None else multiprocessing.cpu_count() // 2
        )

        if record_gif and gif_export_suffix is None:
            self.gif_suffix = 0
            from threading import Lock

            self.gif_suffix_mutex = Lock()

        def get_next_gif_suffix():
            self.gif_suffix_mutex.acquire()
            suffix = f"{self.gif_suffix:03d}"
            self.gif_suffix += 1
            self.gif_suffix_mutex.release()
            return suffix

        # Bypass multiprocessing if only using one process
        if num_processes == 1:
            trajs = []
            num_sa = 0
            while True:
                if record_gif and gif_export_suffix is None:
                    suffix = get_next_gif_suffix()
                else:
                    suffix = gif_export_suffix
                t = self.sim_single(
                    policy_fn,
                    obsfeat_fn,
                    cfg.max_traj_len,
                    alg=alg,
                    dagger_action_beta=dagger_action_beta,
                    record_gif=record_gif,
                    gif_export_dir=gif_export_dir,
                    gif_prefix=gif_prefix,
                    gif_export_suffix=suffix,
                    dagger_eval=dagger_eval,
                )
                trajs.append(t)
                num_sa += len(t)
                if len(trajs) >= cfg.min_num_trajs and num_sa >= cfg.min_total_sa:
                    break
            return TrajBatch.FromTrajs(trajs)

        global _global_sim_info
        _global_sim_info = (self, policy_fn, obsfeat_fn, cfg.max_traj_len)

        trajs = []
        num_sa = 0

        with set_mkl_threads(1):
            # Thanks John
            pool = multiprocessing.Pool(processes=num_processes, maxtasksperchild=maxtasksperchild)
            pending = []
            done = False
            submit_count = 0
            while True:
                if len(pending) < num_processes and not done:
                    if record_gif and gif_export_suffix is None:
                        suffix = get_next_gif_suffix()
                    else:
                        suffix = gif_export_suffix
                    pending.append(
                        pool.apply_async(
                            _rollout,
                            args=(
                                alg,
                                dagger_action_beta,
                                record_gif,
                                gif_export_dir,
                                gif_prefix,
                                suffix,
                                dagger_eval,
                            ),
                        )
                    )
                    submit_count += 1
                stillpending = []
                for job in pending:
                    if job.ready():
                        traj = job.get()
                        trajs.append(traj)
                        num_sa += len(traj)
                    else:
                        stillpending.append(job)
                pending = stillpending

                if exact_num_trajs:
                    done = bool(submit_count == cfg.min_num_trajs)
                else:
                    done = bool(len(trajs) >= cfg.min_num_trajs and num_sa >= cfg.min_total_sa)

                if done:
                    if len(pending) == 0:
                        break
                sleep(0.001)
            pool.close()
            pool.join()

            assert (
                len(trajs) >= cfg.min_num_trajs
                and sum(len(traj) for traj in trajs) >= cfg.min_total_sa
            )
            return TrajBatch.FromTrajs(trajs)


_global_sim_info = None


def _rollout(
    alg,
    dagger_action_beta,
    record_gif,
    gif_export_dir,
    gif_prefix,
    gif_export_suffix,
    dagger_eval,
):
    try:
        import os
        import random

        random.seed(os.urandom(4))
        np.random.seed(int(binascii.hexlify(os.urandom(4)), 16))
        global _global_sim_info
        mdp, policy_fn, obsfeat_fn, max_traj_len = _global_sim_info
        # from threadpoolctl import threadpool_limits
        # with threadpool_limits(limits=1, user_api='blas'):
        return mdp.sim_single(
            policy_fn,
            obsfeat_fn,
            max_traj_len,
            alg=alg,
            dagger_action_beta=dagger_action_beta,
            record_gif=record_gif,
            gif_export_dir=gif_export_dir,
            gif_prefix=gif_prefix,
            gif_export_suffix=gif_export_suffix,
            dagger_eval=dagger_eval,
        )
    except KeyboardInterrupt:
        pass


# Stuff for temporarily disabling MKL threading during multiprocessing
# http://stackoverflow.com/a/28293128
import ctypes

mkl_rt = None
try:
    mkl_rt = ctypes.CDLL("libmkl_rt.so")
    mkl_set_num_threads = mkl_rt.MKL_Set_Num_Threads
    mkl_get_max_threads = mkl_rt.MKL_Get_Max_Threads
except OSError:  # library not found
    pass
    # util.warn(
    #     "MKL runtime not found. Will not attempt to disable multithreaded MKL for parallel rollouts."
    # )
from contextlib import contextmanager


@contextmanager
def set_mkl_threads(n):
    if mkl_rt is not None:
        # orig = mkl_get_max_threads()
        mkl_set_num_threads(n)
    yield
    # NOTE: Setting num_threads to `orig` doesn't work for SLURM, so just set num threads to 1 every time
    # if mkl_rt is not None:
    #     mkl_set_num_threads(orig)
