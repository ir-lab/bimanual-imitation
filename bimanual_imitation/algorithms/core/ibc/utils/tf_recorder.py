import os
import tempfile

import numpy as np
import tensorflow as tf
from absl import logging
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import Trajectory as TfAgentTrajectory
from tf_agents.utils import example_encoding

from bimanual_imitation.algorithms.core.ibc.utils.constants import SPEC_STRING
from irl_data import proto_logger
from irl_data.constants import IRL_DATA_BASE_DIR
from irl_data.data_chunking import create_chunking_dataset


def encode_spec_to_file(output_path, tensor_data_spec):
    """Save a tensor data spec to a tfrecord file.
    Args:
        output_path: The path to the TFRecord file which will contain the spec.
        tensor_data_spec: Nested list/tuple or dict of TensorSpecs, describing the
                        shape of the non-batched Tensors.
    """
    spec_proto = tensor_spec.to_proto(tensor_data_spec)
    with tf.io.TFRecordWriter(output_path) as writer:
        writer.write(spec_proto.SerializeToString())


class TFRecorder(object):
    """Observer for writing experience to TFRecord file.
    To use this observer, create an instance using a trajectory spec object
    and a dataset path:
    trajectory_spec = agent.collect_data_spec
    dataset_path = '/tmp/my_example_dataset'
    tfrecord_observer = TFRecordObserver(dataset_path, trajectory_spec)
    Then add it to the observers kwarg for the driver:
    collect_op = MyDriver(
                ...
                observers=[..., tfrecord_observer],
                num_steps=collect_steps_per_iteration).run()
    *Note*: Depending on your driver you may have to do
            `common.function(tfrecord_observer)` to handle the use of a callable with no
            return within a `tf.group` operation.
    """

    def __init__(
        self, output_path, tensor_data_spec, py_mode=False, compress_image=False, image_quality=95
    ):
        """Creates observer object.
        Args:
            output_path: The path to the TFRecords file.
            tensor_data_spec: Nested list/tuple or dict of TensorSpecs, describing the
                            shape of the non-batched Tensors.
            py_mode: Whether the observer is being used in a py_driver.
            compress_image: Whether to compress image. It is assumed that any uint8
                            tensor of rank 3 with shape (w,h,c) is an image.
            image_quality: An optional int. Defaults to 95. Quality of the compression
                            from 0 to 100 (higher is better and slower).
        Raises:
            ValueError: if the tensors and specs have incompatible dimensions or
            shapes.
        """
        _SPEC_FILE_EXTENSION = ".spec"
        self._py_mode = py_mode
        self._array_data_spec = tensor_spec.to_nest_array_spec(tensor_data_spec)
        self._encoder = example_encoding.get_example_serializer(
            self._array_data_spec, compress_image=compress_image, image_quality=image_quality
        )
        # Two output files: a tfrecord file and a file with the serialized spec
        self.output_path = output_path
        tf.io.gfile.makedirs(os.path.dirname(self.output_path))
        self._writer = tf.io.TFRecordWriter(self.output_path)
        logging.info("Writing dataset to TFRecord at %s", self.output_path)
        # Save the tensor spec used to write the dataset to file
        spec_output_path = self.output_path + _SPEC_FILE_EXTENSION
        encode_spec_to_file(spec_output_path, tensor_data_spec)

    def write(self, *data):
        """Encodes and writes (to file) a batch of data.
        Args:
            *data: (unpacked) list/tuple of batched np.arrays.
        """
        # dataspec = tf.data.DatasetSpec(data, dataset_shape=())
        if self._py_mode:
            structured_data = data
        else:
            # data = nest_utils.unbatch_nested_array(data)
            # structured_data = tf.nest.pack_sequence_as(self._array_data_spec, data)
            raise NotImplementedError
        self._writer.write(self._encoder(structured_data))

    def flush(self):
        """Manually flush TFRecord writer."""
        self._writer.flush()

    def close(self):
        """Close the TFRecord writer."""
        self._writer.close()
        logging.info("Closing TFRecord file at %s", self.output_path)

    def __call__(self, data):
        """If not in py_mode Wraps write() into a TF op for eager execution."""
        if self._py_mode:
            self.write(data)
        else:
            flat_data = tf.nest.flatten(data)
            tf.numpy_function(self.write, flat_data, [], name="encoder_observer")


def export_to_tfrecord(
    environment,
    tfrecord_dir,
    limit_trajs,
    full_action_size,
    full_obs_size,
    pred_horizon,
    action_horizon,
    obs_horizon,
):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as pbtext_file:
        # Write the string to the temporary file
        spec_str = SPEC_STRING.substitute(
            full_obs_size=full_obs_size, full_action_size=full_action_size
        )
        pbtext_file.write(spec_str.encode("utf-8"))
        pbtext_file.flush()  # Ensure data is written
        pbtext_file_name = pbtext_file.name

    dataspec = tensor_spec.from_pbtxt_file(pbtext_file_name)
    os.remove(pbtext_file_name)

    file_path = os.path.join(tfrecord_dir, "temp.tfrecord")
    recorder = TFRecorder(file_path, dataspec, py_mode=True, compress_image=True)

    single_horizon = np.all(
        [
            pred_horizon == 1,
            obs_horizon == 1,
            action_horizon == 1,
        ]
    )

    if single_horizon:
        train_dataset = None
        expert_proto = IRL_DATA_BASE_DIR / f"expert_trajectories/{environment}.proto"
        train_trajs = proto_logger.load_trajs(expert_proto)

        for traj in train_trajs:
            for t in range(len(traj)):
                # We can omit the step type and next step type
                # This info does not get used for just BC training
                # We'll include the correct ones here just for good measure
                if t == 0:
                    step_type = 0
                    next_step_type = 1
                elif t == len(traj) - 2:
                    step_type = 1
                    next_step_type = 2
                elif t == len(traj) - 1:
                    step_type = 2
                    next_step_type = 0
                else:
                    step_type = 1
                    next_step_type = 1

                tensor_traj = TfAgentTrajectory(
                    step_type=np.array(step_type, dtype=np.int32),
                    observation=np.array(traj.obs_T_Do[t], dtype=np.float32),
                    action=np.array(traj.a_T_Da[t], dtype=np.float32),
                    policy_info=(),
                    next_step_type=np.array(next_step_type, dtype=np.int32),
                    reward=np.array(traj.r_T[t], dtype=np.float32),
                    discount=np.array(1, dtype=np.float32),
                )
                recorder(tensor_traj)

    else:
        train_dataset = create_chunking_dataset(
            environment=environment,
            stage="train",
            pred_horizon=pred_horizon,
            obs_horizon=obs_horizon,
            action_horizon=action_horizon,
            limit_trajs=limit_trajs,
            normalize=False,
        )

        for idx in range(len(train_dataset)):
            data = train_dataset[idx]
            obs_history = data["obs"][:obs_horizon].flatten()
            act_history = data["action"].flatten()

            # We can omit the step type and next step type
            # This info does not get used for just BC training
            tensor_traj = TfAgentTrajectory(
                step_type=np.array(1, dtype=np.int32),
                observation=np.array(obs_history, dtype=np.float32),
                action=np.array(act_history, dtype=np.float32),
                policy_info=(),
                next_step_type=np.array(1, dtype=np.int32),
                reward=np.array(0, dtype=np.float32),
                discount=np.array(1, dtype=np.float32),
            )

            recorder(tensor_traj)

    return train_dataset
