import resource

import numpy as np

from bimanual_imitation.algorithms.core.shared import ContinuousSpace, util


class BehavioralCloningOptimizer(object):
    def __init__(
        self,
        mdp,
        policy,
        lr,
        batch_size,
        obsfeat_fn,
        ex_obs,
        ex_a,
        val_ex_obs,
        val_ex_a,
        eval_sim_cfg,
        eval_freq,
    ):
        self.mdp, self.policy, self.lr, self.batch_size, self.obsfeat_fn = (
            mdp,
            policy,
            lr,
            batch_size,
            obsfeat_fn,
        )

        self.mdp.set_mkl()
        assert ex_obs.shape[0] == ex_a.shape[0]
        assert val_ex_obs.shape[0] == val_ex_a.shape[0]

        print(f"{ex_obs.shape[0]} training examples and {val_ex_obs.shape[0]} validation examples")
        self.train_ex_obsfeat, self.train_ex_a = self.obsfeat_fn(ex_obs), ex_a
        self.val_ex_obsfeat, self.val_ex_a = self.obsfeat_fn(val_ex_obs), val_ex_a

        self.eval_sim_cfg = eval_sim_cfg
        self.eval_freq = eval_freq

        self.total_time = 0.0
        self.curr_iter = 0
        self.total_num_sa = 0

    def step(self):
        with util.Timer() as t_all:
            # Subsample expert transitions for SGD
            inds = np.random.choice(self.train_ex_obsfeat.shape[0], size=self.batch_size)
            batch_obsfeat_B_Do = self.train_ex_obsfeat[inds, :]
            batch_a_B_Da = self.train_ex_a[inds, :]
            # Take step
            loss = self.policy.step_bclone(batch_obsfeat_B_Do, batch_a_B_Da, self.lr)

        # Roll out trajectories when it's time to evaluate our policy
        val_loss = val_acc = trueret = avgr = avglen = ent = np.nan
        if self.eval_freq != 0 and (self.curr_iter + 1) % self.eval_freq == 0:
            val_loss = self.policy.compute_bclone_loss(self.val_ex_obsfeat, self.val_ex_a)
            # Evaluate validation accuracy (independent of standard deviation)
            if isinstance(self.mdp.action_space, ContinuousSpace):
                val_acc = (
                    -np.square(
                        self.policy.compute_actiondist_mean(self.val_ex_obsfeat) - self.val_ex_a
                    )
                    .sum(axis=1)
                    .mean()
                )
            else:
                assert self.val_ex_a.shape[1] == 1
                # val_acc = (self.policy.sample_actions(self.val_ex_obsfeat)[1].argmax(axis=1) == self.val_ex_a[1]).mean()
                val_acc = -val_loss  # val accuracy doesn't seem too meaningful so just use this

        # Log
        self.total_num_sa += self.batch_size
        self.total_time += t_all.dt
        self.curr_iter += 1

        fields = [
            ("iter", self.curr_iter, int),
            ("bcloss", loss, float),  # supervised learning loss
            ("valloss", val_loss, float),  # loss on validation set
            ("valacc", val_acc, float),  # loss on validation set
            ("trueret", trueret, float),  # true average return for this batch of trajectories
            ("avgr", avgr, float),  # average reward encountered
            ("avglen", avglen, float),  # average traj length
            ("ent", ent, float),  # entropy of action distributions
            ("ttotal", self.total_time, float),  # total time
            ("nsa", self.total_num_sa, int),  # total time
            # max mem usage
            ("max_mem", int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000.0), int),
            # cur mem usage
            # ("cur_mem", int(psutil.Process(os.getpid()).memory_info().rss / 1024**2), int),
        ]
        return fields
