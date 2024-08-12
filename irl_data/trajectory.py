import numpy as np


class Trajectory(object):
    __slots__ = ("obs_T_Do", "obsfeat_T_Df", "adist_T_Pa", "a_T_Da", "r_T")

    def __init__(self, obs_T_Do, obsfeat_T_Df, adist_T_Pa, a_T_Da, r_T):
        assert (
            obs_T_Do.ndim == 2
            and obsfeat_T_Df.ndim == 2
            and adist_T_Pa.ndim == 2
            and a_T_Da.ndim == 2
            and r_T.ndim == 1
            and obs_T_Do.shape[0]
            == obsfeat_T_Df.shape[0]
            == adist_T_Pa.shape[0]
            == a_T_Da.shape[0]
            == r_T.shape[0]
        )
        self.obs_T_Do = obs_T_Do
        self.obsfeat_T_Df = obsfeat_T_Df
        self.adist_T_Pa = adist_T_Pa
        self.a_T_Da = a_T_Da
        self.r_T = r_T

    def __len__(self):
        return self.obs_T_Do.shape[0]

    # Saving/loading discards obsfeat
    def save_h5(self, grp, **kwargs):
        grp.create_dataset("obs_T_Do", data=self.obs_T_Do, **kwargs)
        grp.create_dataset("adist_T_Pa", data=self.adist_T_Pa, **kwargs)
        grp.create_dataset("a_T_Da", data=self.a_T_Da, **kwargs)
        grp.create_dataset("r_T", data=self.r_T, **kwargs)

    @classmethod
    def LoadH5(cls, grp, obsfeat_fn):
        """
        obsfeat_fn: used to fill in observation features. if None, the raw observations will be copied over.
        """
        obs_T_Do = grp["obs_T_Do"][...]
        obsfeat_T_Df = obsfeat_fn(obs_T_Do) if obsfeat_fn is not None else obs_T_Do.copy()
        return cls(
            obs_T_Do, obsfeat_T_Df, grp["adist_T_Pa"][...], grp["a_T_Da"][...], grp["r_T"][...]
        )


# Utilities for dealing with batches of trajectories with different lengths


def raggedstack(arrays, fill=0.0, axis=0, raggedaxis=1):
    """
    Stacks a list of arrays, like np.stack with axis=0.
    Arrays may have different length (along the raggedaxis), and will be padded on the right
    with the given fill value.
    """
    assert axis == 0 and raggedaxis == 1, "not implemented"
    arrays = [a[None, ...] for a in arrays]
    assert all(a.ndim >= 2 for a in arrays)

    outshape = list(arrays[0].shape)
    outshape[0] = sum(a.shape[0] for a in arrays)
    outshape[1] = max(a.shape[1] for a in arrays)  # take max along ragged axes
    outshape = tuple(outshape)

    out = np.full(outshape, fill, dtype=arrays[0].dtype)
    pos = 0
    for a in arrays:
        out[pos : pos + a.shape[0], : a.shape[1], ...] = a
        pos += a.shape[0]
    assert pos == out.shape[0]
    return out


class RaggedArray(object):
    def __init__(self, arrays, lengths=None):
        from bimanual_imitation.algorithms.core.shared import util

        if lengths is None:
            # Without provided lengths, `arrays` is interpreted as a list of arrays
            # and self.lengths is set to the list of lengths for those arrays
            self.arrays = arrays
            self.stacked = np.concatenate(arrays, axis=0)
            self.lengths = np.array([len(a) for a in arrays])
        else:
            # With provided lengths, `arrays` is interpreted as concatenated data
            # and self.lengths is set to the provided lengths.
            self.arrays = np.split(arrays, np.cumsum(lengths)[:-1])
            self.stacked = arrays
            self.lengths = np.asarray(lengths, dtype=int)
        assert all(len(a) == l for a, l in util.safezip(self.arrays, self.lengths))
        self.boundaries = np.concatenate([[0], np.cumsum(self.lengths)])
        assert self.boundaries[-1] == len(self.stacked)

    def __len__(self):
        return len(self.lengths)

    def __getitem__(self, idx):
        return self.stacked[self.boundaries[idx] : self.boundaries[idx + 1], ...]

    def padded(self, fill=0.0):
        return raggedstack(self.arrays, fill=fill, axis=0, raggedaxis=1)


class TrajBatch(object):
    def __init__(self, trajs, obs, obsfeat, adist, a, r, time):
        self.trajs, self.obs, self.obsfeat, self.adist, self.a, self.r, self.time = (
            trajs,
            obs,
            obsfeat,
            adist,
            a,
            r,
            time,
        )

    @classmethod
    def FromTrajs(cls, trajs):
        assert all(isinstance(traj, Trajectory) for traj in trajs)
        obs = RaggedArray([t.obs_T_Do for t in trajs])
        obsfeat = RaggedArray([t.obsfeat_T_Df for t in trajs])
        adist = RaggedArray([t.adist_T_Pa for t in trajs])
        a = RaggedArray([t.a_T_Da for t in trajs])
        r = RaggedArray([t.r_T for t in trajs])
        time = RaggedArray([np.arange(len(t), dtype=float) for t in trajs])
        return cls(trajs, obs, obsfeat, adist, a, r, time)

    def with_replaced_reward(self, new_r):
        from bimanual_imitation.algorithms.core.shared import util

        new_trajs = [
            Trajectory(traj.obs_T_Do, traj.obsfeat_T_Df, traj.adist_T_Pa, traj.a_T_Da, traj_new_r)
            for traj, traj_new_r in util.safezip(self.trajs, new_r)
        ]
        return TrajBatch(new_trajs, self.obs, self.obsfeat, self.adist, self.a, new_r, self.time)

    def __len__(self):
        return len(self.trajs)

    def __getitem__(self, idx):
        return self.trajs[idx]

    def save_h5(self, f, starting_id=0, **kwargs):
        for i, traj in enumerate(self.trajs):
            traj.save_h5(f.require_group("%06d" % (i + starting_id)), **kwargs)

    @classmethod
    def LoadH5(cls, dset, obsfeat_fn):
        return cls.FromTrajs([Trajectory.LoadH5(v, obsfeat_fn) for k, v in dset.iteritems()])

    def to_dataframe(self, **kwargs):
        import numpy as np
        import pandas as pd

        df_list = []
        for traj_idx, traj in enumerate(self.trajs):
            # Create data dictionary
            tdata = {"time": np.arange(len(traj)), "reward": traj.r_T}
            # Add action columns
            tdata.update({f"act_{i}": traj.a_T_Da[:, i] for i in range(traj.a_T_Da.shape[1])})
            # Add observation columns
            tdata.update({f"obs_{i}": traj.obs_T_Do[:, i] for i in range(traj.obs_T_Do.shape[1])})
            # Convert dictionary to DataFrame
            tdf = pd.DataFrame(tdata)
            tdf["rollout"] = traj_idx

            # Add extra columns from kwargs
            for key, val in kwargs.items():
                tdf[key] = val

            df_list.append(tdf)

        # Concatenate all DataFrames at once
        traj_df = pd.concat(df_list, ignore_index=True)
        return traj_df
