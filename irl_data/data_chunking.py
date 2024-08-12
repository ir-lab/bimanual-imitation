from enum import Enum

import numpy as np

from irl_data import proto_logger
from irl_data.constants import IRL_DATA_BASE_DIR
from irl_data.trajectory import TrajBatch

"""
|o|o|                                       obs_horizon: 2
|a|a^|a^|a^|a^|a^|a^|a^|                    action_horizon: 8
|p|p |p |p |p |p |p |p |p|p|p|p|p|p|p|p|    pred_horizon: 16

The p's aligning with the a^'s are sent to the simulator
we discard the a's (previous action prediction) and any p's ater the last a^
"""


def create_sample_indices(
    episode_ends: np.ndarray, sequence_length: int, pad_before: int = 0, pad_after: int = 0
):
    indices = list()
    for i in range(len(episode_ends)):
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i - 1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after

        # range stops one idx before end
        for idx in range(min_start, max_start + 1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx + sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx + start_idx)
            end_offset = (idx + sequence_length + start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            indices.append([buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx])
    indices = np.array(indices)
    return indices


def sample_sequence(
    train_data, sequence_length, buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx
):
    result = dict()
    for key, input_arr in train_data.items():
        sample = input_arr[buffer_start_idx:buffer_end_idx]
        data = sample
        if (sample_start_idx > 0) or (sample_end_idx < sequence_length):
            data = np.zeros(shape=(sequence_length,) + input_arr.shape[1:], dtype=input_arr.dtype)
            if sample_start_idx > 0:
                data[:sample_start_idx] = sample[0]
            if sample_end_idx < sequence_length:
                data[sample_end_idx:] = sample[-1]
            data[sample_start_idx:sample_end_idx] = sample
        result[key] = data
    return result


class StatsType(Enum):
    NO_NORMALIZE = 0
    NORMALIZE = 1


# normalize data
def get_data_stats(data, normalize):
    if normalize:
        data = data.reshape(-1, data.shape[-1])
        stats = {
            "min": np.min(data, axis=0),
            "max": np.max(data, axis=0),
            "type": StatsType.NORMALIZE,
        }
    else:
        stats = {"type": StatsType.NO_NORMALIZE}
    return stats


def normalize_data(data, stats):
    if stats["type"] == StatsType.NO_NORMALIZE:
        ndata = data
    elif stats["type"] == StatsType.NORMALIZE:
        # nomalize to [0,1]
        ndata = (data - stats["min"]) / (stats["max"] - stats["min"])
        # normalize to [-1, 1]
        ndata = ndata * 2 - 1
    else:
        raise ValueError("Invalid stats type")

    return ndata


def unnormalize_data(ndata, stats):
    if stats["type"] == StatsType.NO_NORMALIZE:
        data = ndata
    elif stats["type"] == StatsType.NORMALIZE:
        ndata = (ndata + 1) / 2
        data = ndata * (stats["max"] - stats["min"]) + stats["min"]
    else:
        raise ValueError("Invalid stats type")
    return data


def create_chunking_dataset(
    environment,
    stage,
    pred_horizon,
    obs_horizon,
    action_horizon,
    limit_trajs: int = None,
    normalize=True,
):
    try:
        from torch.utils.data import Dataset

        print("Loading Torch Dataset")
    except ImportError:
        print("Loading Non-Torch Dataset")
        Dataset = object  # Use `object` as a fallback base class

    class QuadInsertChunkingDataset(Dataset):
        def __init__(
            self, environment, stage, pred_horizon, obs_horizon, action_horizon, normalize=True
        ):
            if stage == "train":
                prefix = "expert_trajectories"
            elif stage == "val":
                prefix = "expert_validation_trajectories"

            prefix = "expert_trajectories"

            expert_proto = IRL_DATA_BASE_DIR / f"{prefix}/{environment}.proto"
            self.train_trajs = proto_logger.load_trajs(expert_proto)

            if limit_trajs is not None:
                assert isinstance(limit_trajs, int)
                self.train_trajs = self.train_trajs[:limit_trajs]
                print(f"Using a subset of the training trajectories: {len(self.train_trajs)}")

            self.single_horizon = np.all(
                [
                    pred_horizon == 1,
                    obs_horizon == 1,
                    action_horizon == 1,
                ]
            )
            if self.single_horizon:
                print("[Dataset] Using single horizon!")

            trajbatch = TrajBatch.FromTrajs(self.train_trajs)

            train_data = {
                "action": np.array(trajbatch.a.stacked, dtype=np.float32),
                "obs": np.array(trajbatch.obs.stacked, dtype=np.float32),
            }
            # Marks one-past the last index for each episode

            if self.single_horizon:
                indices = np.arange(train_data["action"].shape[0])
            else:
                episode_ends = np.cumsum(trajbatch.a.lengths)

                # compute start and end of each state-action sequence
                # also handles padding
                indices = create_sample_indices(
                    episode_ends=episode_ends,
                    sequence_length=pred_horizon,
                    # add padding such that each timestep in the dataset are seen
                    pad_before=obs_horizon - 1,
                    pad_after=action_horizon - 1,
                )

                # compute statistics and normalized data to [-1,1]
            stats = dict()
            normalized_train_data = dict()
            for key, data in train_data.items():
                stats[key] = get_data_stats(data, normalize)
                normalized_train_data[key] = normalize_data(data, stats[key])

            self.indices = indices
            self.stats = stats
            self.normalized_train_data = normalized_train_data
            self.pred_horizon = pred_horizon
            self.action_horizon = action_horizon
            self.obs_horizon = obs_horizon

        def __len__(self):
            # all possible segments of the dataset
            return len(self.indices)

        def __getitem__(self, idx):
            # get the start/end indices for this datapoint
            if self.single_horizon:
                nsample = dict()
                nsample["action"] = self.normalized_train_data["action"][idx : idx + 1]
                nsample["obs"] = self.normalized_train_data["obs"][idx : idx + 1]
            else:
                buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = self.indices[
                    idx
                ]

                # get nomralized data using these indices
                nsample = sample_sequence(
                    train_data=self.normalized_train_data,
                    sequence_length=self.pred_horizon,
                    buffer_start_idx=buffer_start_idx,
                    buffer_end_idx=buffer_end_idx,
                    sample_start_idx=sample_start_idx,
                    sample_end_idx=sample_end_idx,
                )

                # discard unused observations
                nsample["obs"] = nsample["obs"][: self.obs_horizon, :]

            return nsample

    return QuadInsertChunkingDataset(
        environment, stage, pred_horizon, obs_horizon, action_horizon, normalize
    )
