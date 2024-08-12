import resource

import numpy as np
import theano

from bimanual_imitation.algorithms.configs import ALG
from bimanual_imitation.algorithms.core.shared import ContinuousSpace, TrajBatch, rl, util


class DAggerOptimizer(object):
    def __init__(
        self,
        mdp,
        policy,
        lr,
        sim_cfg,
        ex_obs,
        ex_a,
        ex_t,
        val_ex_obs,
        val_ex_a,
        val_ex_t,
        eval_freq,
        num_epochs=64,
        minibatch_size=256,
        beta_start=1.0,
        beta_decay=0.95,
        init_bclone=False,
        subsample_rate=1,
    ):
        self.mdp, self.policy, self.lr, self.sim_cfg = mdp, policy, lr, sim_cfg
        self.ex_obs, self.ex_a, self.ex_t = ex_obs, ex_a, ex_t
        self.val_ex_obs, self.val_ex_a, self.val_ex_t = val_ex_obs, val_ex_a, val_ex_t
        self.eval_freq = eval_freq
        self.num_epochs = num_epochs
        self.minibatch_size = minibatch_size
        self.beta = beta_start
        self.beta_decay = beta_decay
        self.all_trajs = []
        self.total_num_trajs = 0
        self.total_num_sa = 0
        self.total_time = 0.0
        self.curr_iter = 0
        self.init_bclone = init_bclone
        self.subsample_rate = subsample_rate

        if isinstance(self.policy, rl.DeterministicPolicy):
            print("DAgger Optimizer: Training with Deterministic Policy")
            self._deterministic_training = True
        else:
            print("DAgger Optimizer: Training with Non-Deterministic Policy")
            self._deterministic_training = False

        self.mdp.set_mkl()
        print(f"DAgger Optimizer: Using beta start of {self.beta} and beta decay {self.beta_decay}")

        if self.init_bclone:
            print("Initializing BC weights")
            # 15_000 is arbitrary, but should be reasonable for initializing the weights
            for _epoch in range(15_000):
                inds = np.random.choice(self.ex_obs.shape[0], size=int(self.minibatch_size))
                batch_obsfeat_B_Do = self.ex_obs[inds, :]
                batch_a_B_Da = self.ex_a[inds, :]
                # Take step
                loss = self.policy.step_bclone(batch_obsfeat_B_Do, batch_a_B_Da, 1e-4)

                if _epoch % 1000 == 0:
                    print(f"Epoch {_epoch} ...")

            print("Initialized BC weights!")

    def policy_fn(self, obsfeat_B_Df, env, deterministic):
        expert_action = env.dagger_expert_policy_fn()
        expert_action = np.array(expert_action, dtype=theano.config.floatX).reshape(1, -1)
        predicted_action, _ = self.policy.sample_actions(obsfeat_B_Df, deterministic=deterministic)
        return expert_action, predicted_action

    def step_bclone_minibatch(self, obs_data, act_data, lr, minibatch_size=64):
        num_data = obs_data.shape[0]
        num_minibatches = int(num_data / minibatch_size)
        losses = []
        for i in range(num_minibatches):
            start_idx = i * minibatch_size
            end_idx = (i + 1) * minibatch_size
            obs_minibatch = obs_data[start_idx:end_idx]
            act_minibatch = act_data[start_idx:end_idx]
            loss = self.policy.step_bclone(obs_minibatch, act_minibatch, lr)
            losses.append(loss)
        avg_minibatch_loss = np.mean(losses)
        return avg_minibatch_loss

    def step(self):
        with util.Timer() as t_all:
            with util.Timer() as t_sample:
                sampbatch: TrajBatch = self.mdp.sim_mp(
                    policy_fn=lambda obsfeat_B_Df, env: self.policy_fn(
                        obsfeat_B_Df, env, self._deterministic_training
                    ),
                    obsfeat_fn=lambda obs: obs,
                    cfg=self.sim_cfg,
                    alg=ALG.DAGGER,
                    dagger_action_beta=self.beta,
                    dagger_eval=False,
                )

            self.all_trajs += sampbatch.trajs
            all_traj_batch = TrajBatch.FromTrajs(self.all_trajs)

            obs_data = all_traj_batch.obs.stacked[:: self.subsample_rate]
            act_data = all_traj_batch.a.stacked[:: self.subsample_rate]
            assert obs_data.shape[0] == act_data.shape[0]

            # Do policy updates here
            for _epoch in range(self.num_epochs):
                loss = self.step_bclone_minibatch(
                    obs_data, act_data, self.lr, minibatch_size=self.minibatch_size
                )

        val_loss = val_acc = np.nan
        if self.eval_freq != 0 and (self.curr_iter + 1) % self.eval_freq == 0:
            val_loss = self.policy.compute_bclone_loss(self.val_ex_obs, self.val_ex_a)
            # Evaluate validation accuracy (independent of standard deviation)
            if isinstance(self.mdp.action_space, ContinuousSpace):
                val_acc = (
                    -np.square(self.policy.compute_actiondist_mean(self.val_ex_obs) - self.val_ex_a)
                    .sum(axis=1)
                    .mean()
                )
            else:
                assert self.val_ex_a.shape[1] == 1
                # val_acc = (self.policy.sample_actions(self.val_ex_obsfeat)[1].argmax(axis=1) == self.val_ex_a[1]).mean()
                val_acc = -val_loss  # val accuracy doesn't seem too meaningful so just use this

        self.total_num_trajs += len(sampbatch)
        self.total_num_sa += sum(len(traj) for traj in sampbatch)
        self.total_time += t_all.dt
        self.curr_iter += 1
        self.beta *= self.beta_decay

        fields = [
            ("iter", self.curr_iter, int),
            # average return for this batch of trajectories
            ("trueret", sampbatch.r.padded(fill=0.0).sum(axis=1).mean(), float),
            # average traj length
            ("avglen", int(np.mean([len(traj) for traj in sampbatch])), int),
            # total number of state-action pairs sampled over the course of training
            ("nsa", self.total_num_sa, int),
            # total number of trajs sampled over the course of training
            ("ntrajs", self.total_num_trajs, int),
            # supervised learning loss
            ("bcloss", loss, float),
            # loss on validation set
            ("valloss", val_loss, float),
            # loss on validation set
            ("valacc", val_acc, float),
            # mixing coefficient
            ("beta", self.beta, float),
            # subsample rate for fitting policy
            # ("samprt", self.subsample_rate, int),
            # total number of epochs
            ("nepochs", self.num_epochs, int),
            # time for sampling
            ("tsamp", t_sample.dt, float),
            # total time
            ("ttotal", self.total_time, float),
            # max mem usage
            ("max_mem", int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000.0), int),
            # cur mem usage
            # ("cur_mem", int(psutil.Process(os.getpid()).memory_info().rss / 1024**2), int),
        ]

        return fields
