from typing import List

import numpy as np

from irl_data.trajectory import TrajBatch, Trajectory
from irl_data.trajectory_proto.build.py.trajectory_pb2 import floatList, multiFloatList, trajectory


def export_trajs(trajbatch: TrajBatch, filename):
    """
    Export expert data to a protobuf file
    """
    assert isinstance(trajbatch, TrajBatch)

    lengths = np.array([len(traj) for traj in trajbatch])

    traj = trajectory()
    for obs in trajbatch.obs:
        mfl = multiFloatList()
        for dim in range(obs.shape[1]):
            fl = floatList()
            fl.value.extend(obs[:, dim])
            mfl.sub_lists.append(fl)
        traj.observations.append(mfl)

    for act in trajbatch.a:
        mfl = multiFloatList()
        for dim in range(act.shape[1]):
            fl = floatList()
            fl.value.extend(act[:, dim])
            mfl.sub_lists.append(fl)
        traj.actions.append(mfl)

    for rs in trajbatch.r:
        fl = floatList()
        fl.value.extend(rs)
        traj.rewards.append(fl)

    traj.lengths.extend(lengths)
    f = open(filename, "wb")
    f.write(traj.SerializeToString())
    f.close()
    print(f"Exported {filename}!")


def load_trajs(filename: str) -> List[Trajectory]:
    """
    Load expert data from a protobuf file and return a list of Trajectories
    """
    f = open(filename, "rb")
    traj = trajectory()
    traj.ParseFromString(f.read())
    f.close()
    trajs = []
    for idx in range(len(traj.observations)):
        act = np.vstack([x.value for x in traj.actions[idx].sub_lists]).T
        obs = np.vstack([x.value for x in traj.observations[idx].sub_lists]).T
        rewards = np.array(traj.rewards[idx].value)
        t = Trajectory(obs, np.zeros(obs.shape), np.zeros((obs.shape[0], 2)), act, rewards)
        trajs.append(t)

    return trajs


def load_theano_dataset(proto_file):
    """
    This method is only used to provide expert data to theano-based algorithms
    """
    f = open(proto_file, "rb")
    traj = trajectory()
    traj.ParseFromString(f.read())
    f.close()
    num_trajs, max_len = len(traj.lengths), max(traj.lengths)
    action_dim = len(traj.actions[0].sub_lists)
    obs_dim = len(traj.observations[0].sub_lists)
    actions_all = np.zeros((num_trajs, max_len, action_dim))
    obs_all = np.zeros((num_trajs, max_len, obs_dim))
    rewards_all = np.zeros((num_trajs, max_len))
    for idx in range(len(traj.observations)):
        act = np.vstack([x.value for x in traj.actions[idx].sub_lists]).T
        actions_all[idx, : act.shape[0], :] = act
        obs = np.vstack([x.value for x in traj.observations[idx].sub_lists]).T
        obs_all[idx, : obs.shape[0], :] = obs
        rewards = traj.rewards[idx].value
        rewards_all[idx, : len(rewards)] = rewards
    lengths = np.array(traj.lengths)
    return obs_all, actions_all, rewards_all, lengths
