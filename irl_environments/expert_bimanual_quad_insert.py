import argparse
import time as time_lib
from enum import Enum
from pathlib import Path
from typing import Dict

import mujoco
import numpy as np
import yaml
from gymnasium.spaces import Box
from transforms3d.affines import compose
from transforms3d.euler import euler2mat, euler2quat, quat2euler, quat2mat
from transforms3d.quaternions import mat2quat

from irl_control.constants import IRL_CONTROL_BASE_DIR
from irl_control.device import DeviceState
from irl_control.mujoco_gym_app import MujocoGymAppHighFidelity
from irl_control.utils.target import Target
from irl_data import proto_logger
from irl_data.trajectory import TrajBatch, Trajectory

# Define the default orientations of the end effectors
DEFAULT_EE_ROT = np.deg2rad([0, -90, -90])
DEFAULT_EE_ORIENTATION = quat2euler(euler2quat(*DEFAULT_EE_ROT, "sxyz"), "rxyz")
DEFAULT_EE_QUAT = euler2quat(*DEFAULT_EE_ROT)
MAX_WAYPOINT_TIME = 60.0  # seconds


class Action(Enum):
    """
    Action Enums are used to force the action sequence instructions (strings)
    to be converted into valid actions
    """

    WP = 0
    GRIP = 1


class BaseQuadInsertionTask:
    """
    In this example, the robot performs a variety of insertion tasks,
    using the male and female adapters, which are generated at fixed
    or random locations (depending on the demo being run).
    """

    def __init__(self, initial_config_path=None, render_expert=False):
        self.__action_config_name = "quad_insert.yaml"
        self.__action_object_names = ["action_objects"]
        self.__action_object_name = self.__action_object_names[0]

        self.__got_action_config = False
        action_config = self._get_action_config(self.__action_config_name)

        self.__targets: Dict[str, Target] = {}
        self.__device_names = []
        device_config = action_config["device_config"]
        for device_name, controller in zip(device_config["devices"], device_config["controllers"]):
            self.__targets[device_name] = Target()
            self.__device_names.append(device_name)

        if initial_config_path is not None:
            assert isinstance(initial_config_path, Path)
            with open(initial_config_path, "r") as init_config_file:
                init_config = yaml.safe_load(init_config_file)
            self.__initial_config = init_config
            action_config[self.__action_object_name] = self.__initial_config[
                self.__action_object_name
            ]
        else:
            self.__initial_config = None

        self.__action_config = action_config

        self.__device_names = ["base", "ur5left", "ur5right"]

        self.__action_map = self.get_action_map()

        # Get the default parameters for the actions
        self.__DEFAULT_PARAMS: Dict[Action, Dict] = dict(
            [(action, self.get_default_action_ctrl_params(action)) for action in Action]
        )
        self.__render_expert = render_expert

    @property
    def action_objects(self):
        return self.__action_config[self.__action_object_name]

    def get_default_action_ctrl_params(self, action):
        """
        Get the default gain, velocity, and gripper values for the insertion task.
        These can be changed, but experimentally these values have been found to work
        well with the insertion action sequence.
        """
        # Waypoint action defaults
        if action == Action.WP:
            param_dict = {
                "max_error": 0.05,
                "gripper_force": -0.08,
                "noise": [0.0, 0.0],
            }
        # Grip action defaults
        elif action == Action.GRIP:
            param_dict = {"gripper_force": 0.0, "gripper_duation": 1.0}
        elif action == Action.INTERP:
            param_dict = {"method": "linear", "steps": 2}
        return param_dict

    def get_action_map(self):
        """
        Return the functions associated with the action defined in the action sequence.
        """
        action_map: Dict[Action, function] = {
            Action.WP: self.go_to_waypoint,
            Action.GRIP: self.grip,
        }
        return action_map

    def _get_action_config(self, config_file: str):
        """
        Return the dictionary formatted data structure of the
        configuration file passed into the function.
        config_file should be the name of the yaml file in the
        action_sequence_configs directory.
        """
        assert self.__got_action_config == False
        # Load the action config from the action_sequence_configs directory
        action_obj_config_path = IRL_CONTROL_BASE_DIR / "action_sequence_configs" / config_file
        with open(action_obj_config_path, "r") as file:
            action_config = yaml.safe_load(file)

        self.__got_action_config = True
        return action_config

    def string2action(self, string: str):
        """
        Return the Enum associated with the action token.
        """
        if string == "WP":
            return Action.WP
        elif string == "GRIP":
            return Action.GRIP
        else:
            raise ValueError(f"Not implemented for {string}")

    def grip(self, params):
        """
        This is an action which is responsbile for solely operating the gripper.
        This method assumes that self.__targets is set for the arms beforehand, such that
        the arms will remain in the current position (since no target is applied here).
        """
        assert params["action"] == "GRIP"
        self.update_action_ctrl_params(params, Action.GRIP)
        # Apply gripper forces for duration specified
        r_pos_hist = []
        r_quat_hist = []
        l_pos_hist = []
        l_quat_hist = []
        grip_duration = float(params["gripper_duration"])
        tot_time = 0
        while tot_time < grip_duration:
            ctrlr_output = self.controller.generate(self.__targets)
            ctrl = np.zeros(self.ctrl_action_space.shape)
            for force_idx, force in zip(*ctrlr_output):
                ctrl[force_idx] = force

            r_gripper_idx = 7
            l_gripper_idx = 14
            # Apply gripper force to the active arm
            gripper_force = params["gripper_force"]
            ctrl[r_gripper_idx] = gripper_force
            ctrl[l_gripper_idx] = gripper_force

            self.do_simulation(ctrl, self.frame_skip)
            if self.__render_expert:
                self.render()
            tot_time += self.model.opt.timestep * self.frame_skip

            state = self.robot.get_device_states()
            r_pos_hist.append(state["ur5right"][DeviceState.EE_XYZ])
            r_quat_hist.append(state["ur5right"][DeviceState.EE_QUAT])
            l_pos_hist.append(state["ur5left"][DeviceState.EE_XYZ])
            l_quat_hist.append(state["ur5left"][DeviceState.EE_QUAT])

        return (r_pos_hist, r_quat_hist, l_pos_hist, l_quat_hist)

    def is_done(self, max_error, step):
        """
        Determines whether an action is done based on (currently)
        the velocities of the devices. Alternative options include
        the L2 error (commented out)
        """
        # Based on steps
        if step < 25:
            return False

        # Based on DQ Error
        vel = []
        for device_name in self.__device_names:
            if device_name != "base":
                vel += list(self.robot.get_device(device_name).get_state(DeviceState.EE_XYZ_VEL))
                vel += np.linalg.norm(
                    self.controller.calc_error(
                        self.__targets[device_name], self.robot.get_device(device_name)
                    )
                )

        vel = np.asarray(vel)
        if np.all(np.isclose(np.zeros_like(vel), vel, rtol=max_error, atol=max_error)):
            return True

        return False

    def set_waypoint_targets(self, params, dynamic_grip=False):
        """
        Set the targets for the robot devices (arms) based on the values
        specified in the action sequence.
        """
        if "target_xyz" in params.keys() and not dynamic_grip:
            # NOTE: target_xyz can be a string (instead of a list); there are 2 possibilites
            # 1) 'starting_pos': Must be set in python before running the WP Action
            # 2) '<action_object_name>': a string which must an action object, where the coordinates
            #    of this object are retrieved from the simulator
            if isinstance(params["target_xyz"], str):
                # Parse the action parameter for the target xyz location
                for d, t in zip(self.__targets.keys(), target_obj):
                    target_xyz = (
                        np.asarray(t)
                        + np.random.normal(
                            params["noise"][0], params["noise"][1], size=np.asarray(t).shape
                        )
                    ).tolist()
                    self.__targets[d].set_xyz(target_xyz)
            elif isinstance(params["target_xyz"], list):
                for idx, (d, t) in enumerate(zip(self.__targets.keys(), params["target_xyz"])):
                    if isinstance(t, str):
                        t_split = t.split(".")
                        if len(t_split) == 1:
                            # Apply the necessary yaw offet to the end effector
                            target_obj = self.action_objects[t_split[0]]
                            # Get the quaternion for the target object
                            joint_name = target_obj["joint_name"]
                            joint_id = mujoco.mj_name2id(
                                self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
                            )
                            start_idx = self.model.jnt_qposadr[joint_id]
                            obj_pos = self.sim.data.get_joint_qpos(target_obj["joint_name"])[
                                start_idx : start_idx + 3
                            ]
                        elif len(t_split) == 2:
                            assert "offset" in t_split[1]
                            # Apply the necessary yaw offet to the end effector
                            target_obj = self.action_objects[t_split[0]]
                            # Get the quaternion for the target object
                            joint_name = target_obj["joint_name"]
                            joint_id = mujoco.mj_name2id(
                                self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
                            )
                            start_idx = self.model.jnt_qposadr[joint_id]
                            female_pos = self.data.qpos[start_idx : start_idx + 3]
                            female_quat = self.data.qpos[start_idx + 3 : start_idx + 7]
                            f1 = compose(female_pos, quat2mat(female_quat), [1, 1, 1])
                            offset = np.array(self.action_objects[t_split[0]][t_split[1]])[idx]
                            f2 = compose(offset, np.eye(3), [1, 1, 1])
                            f12 = np.matmul(f1, f2)
                            obj_pos = f12[:3, -1].flatten()
                        else:
                            raise NotImplementedError
                    else:
                        obj_pos = np.asarray(t)

                    target_xyz = obj_pos  # + np.random.normal(params['noise'][0], params['noise'][1], size=np.asarray(t).shape)).tolist()
                    self.__targets[d].set_xyz(target_xyz)
            else:
                print("Invalid type for target_xyz!")
                raise ValueError

        # Set the target orientations for the arm
        if "target_quat" in params.keys():
            for d, t in zip(self.__targets.keys(), params["target_quat"]):
                if isinstance(t, str):
                    t_split = t.split(".")
                    if len(t_split) == 1:
                        target_obj = self.action_objects[t_split[0]]
                        joint_name = target_obj["joint_name"]
                        joint_id = mujoco.mj_name2id(
                            self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
                        )
                        start_idx = self.model.jnt_qposadr[joint_id]
                        obj_quat = self.data.qpos[start_idx + 3 : start_idx + 7]
                        if dynamic_grip:
                            obj_eul = np.array(quat2euler(obj_quat))
                            obj_eul[0] = 0
                            obj_eul[1] = 0
                            obj_quat = euler2quat(*obj_eul)
                        grip_eul = DEFAULT_EE_ROT + [0, 0, np.deg2rad(target_obj["grip_yaw"])]
                        tfmat_obj = compose([0, 0, 0], quat2mat(obj_quat), [1, 1, 1])
                        tfmat_grip = compose([0, 0, 0], euler2mat(*grip_eul), [1, 1, 1])
                        tfmat = np.matmul(tfmat_obj, tfmat_grip)
                        obj_quat = np.array(mat2quat(tfmat[:3, :3]))
                    elif len(t_split) == 2:
                        raise NotImplementedError

                else:
                    obj_quat = t

                self.__targets[d].set_quat(obj_quat)

    def update_action_ctrl_params(self, params, action: Action):
        """
        Apply the default values to the parameter, if it is not specified in the action.
        """
        for key, default_val in self.__DEFAULT_PARAMS[action].items():
            params[key] = params[key] if key in params.keys() else default_val

    def go_to_waypoint(self, params):
        """
        This is the main action used in the insertion demo.
        Applies forces to the robot and gripper (as opposed to the gripper only, in the grip action)
        using the parameters specified by the action.
        """
        assert params["action"] == "WP"
        # Apply default parameter values to those that are unspecified
        self.update_action_ctrl_params(params, Action.WP)
        self.set_waypoint_targets(params)

        r_pos_hist = []
        r_quat_hist = []
        l_pos_hist = []
        l_quat_hist = []
        step = 0
        start_time = time_lib.time()
        # Iterate the controller until the desired level of error is achieved
        while (not self.is_done(params["max_error"], step)) and (
            time_lib.time() - start_time < MAX_WAYPOINT_TIME
        ):
            self.set_waypoint_targets(params, dynamic_grip=True)
            step += 1
            ctrlr_output = self.controller.generate(self.__targets)
            ctrl = np.zeros(self.ctrl_action_space.shape)
            for force_idx, force in zip(*ctrlr_output):
                ctrl[force_idx] = force

            r_gripper_idx = 7
            l_gripper_idx = 14

            # Apply gripper force to the active arm
            gripper_force = params["gripper_force"]
            ctrl[r_gripper_idx] = gripper_force
            ctrl[l_gripper_idx] = gripper_force

            self.do_simulation(ctrl, self.frame_skip)
            if self.__render_expert:
                self.render()

            state = self.robot.get_device_states()
            r_pos_hist.append(state["ur5right"][DeviceState.EE_XYZ])
            r_quat_hist.append(state["ur5right"][DeviceState.EE_QUAT])
            l_pos_hist.append(state["ur5left"][DeviceState.EE_XYZ])
            l_quat_hist.append(state["ur5left"][DeviceState.EE_QUAT])

        return (r_pos_hist, r_quat_hist, l_pos_hist, l_quat_hist)

    def initialize_action_objects(self):
        """
        Apply the initial positions and orientations specified by the
        objects inside of the action_objects (action sequence file).
        """
        for obj_name in self.action_objects:
            obj = self.action_objects[obj_name]
            # Convert the degrees in the yaml file to radians
            target_quat = obj["initial_pos_quat"] if "initial_pos_quat" in obj.keys() else None
            # Parse the initial xyz position
            target_pos = obj["initial_pos_xyz"] if "initial_pos_xyz" in obj.keys() else None
            # Set the position and quaternion of the simulator object
            self.set_free_joint_qpos(obj["joint_name"], quat=target_quat, pos=target_pos)

    def run_insertion_sequence(self, action_sequence):
        # self.start_pos = np.copy(self.active_arm.get_state(DeviceState.EE_XYZ))
        trajs = []
        for action_idx, action_entry in enumerate(action_sequence):
            # print(f"Running {action_entry['name']}")
            action = self.string2action(action_entry["action"])
            action_func = self.__action_map[action]
            traj_data = action_func(action_entry)
            trajs.append(traj_data)
        return trajs

    def initialize_action_objects_random(self):
        """
        Randomly generate the positions of the objects in the scene,
        such that the male/female do not fall onto each other
        and that the objects are within the coordinates given below
        """
        assert self.__initial_config is None
        # Use these coordinates to randomly place the male object
        male_x = round(np.random.uniform(low=0.57, high=0.74), 3)
        male_y = round(np.random.uniform(low=0.22, high=0.27), 3)

        z_offset = np.random.uniform(0.0, 0.08)
        self.__action_config[self.__action_object_name]["male_object"]["hover3_offset"][1][
            2
        ] += z_offset
        self.__action_config[self.__action_object_name]["male_object"]["hover3_offset"][2][
            2
        ] += z_offset

        # Apply the random orientation and position to the male object
        male_obj = self.action_objects["male_object"]
        yaw_male = int(np.random.uniform(50, 80))
        male_obj["initial_pos_quat"] = [
            round(float(x), 4) for x in euler2quat(*[0, 0, np.deg2rad(yaw_male)])
        ]
        male_obj["initial_pos_xyz"][0] = male_x
        male_obj["initial_pos_xyz"][1] = male_y

        # Set the position and quaternion of the simulator object
        self.set_free_joint_qpos(
            male_obj["joint_name"],
            quat=male_obj["initial_pos_quat"],
            pos=male_obj["initial_pos_xyz"],
        )

        # Use these coordinates to randomly place the female object
        female_x = 0  # round(np.random.uniform(low=0.0, high=0.3), 3)
        female_y = 0.7  # round(np.random.uniform(low=0.5, high=0.7), 3)

        # Apply the random orientation and position to the female object
        female_obj = self.action_objects["female_object"]
        # yaw_female = int(np.random.uniform(-20, 20))
        # female_obj['initial_pos_abg'] = [0, 0, yaw_female]
        # female_obj['initial_pos_xyz'][0] = female_x # if arm_name == 'ur5right' else -1*fx
        # female_obj['initial_pos_xyz'][1] = female_y
        # self.set_free_joint_qpos(
        #     female_obj["joint_name"],
        #     quat=euler2quat(*np.deg2rad(female_obj["initial_pos_abg"])),
        #     pos=female_obj["initial_pos_xyz"],
        # )

        # Set the position and quaternion of the simulator object
        self.set_free_joint_qpos(
            female_obj["joint_name"],
            quat=female_obj["initial_pos_quat"],
            pos=female_obj["initial_pos_xyz"],
        )

        export_init_config = dict()
        export_init_config[self.__action_object_name] = dict()
        export_init_config[self.__action_object_name]["male_object"] = male_obj
        export_init_config[self.__action_object_name]["female_object"] = female_obj
        return export_init_config

    def run_insertion_main_sequence(
        self, export_proto_filename=None, export_init_config: dict = None
    ):
        action_sequence_name = "demo_sequence"
        trajs = self.run_insertion_sequence(self.__action_config[action_sequence_name])

        l_pos_data = np.vstack(trajs[0][2])
        l_quat_data = np.vstack(trajs[0][3])

        r_pos_data = np.vstack(trajs[0][0])
        r_quat_data = np.vstack(trajs[0][1])

        obs_T_Do = np.hstack([l_pos_data, l_quat_data, r_pos_data, r_quat_data])
        print(obs_T_Do.shape[0])

        if export_proto_filename is not None:
            assert export_init_config is not None

            # Save the observations from the first sub-action sequence
            # Fill in NaN for non-used values
            num_obs = obs_T_Do.shape[0]
            a_T_Da = np.ones((num_obs, 1)) * np.nan
            obsfeat_T_Df = np.ones((num_obs, 1)) * np.nan
            adist_T_Pa = np.ones((num_obs, 1)) * np.nan
            r_T = np.ones(num_obs) * np.nan

            # Export the trajectory states
            single_traj = Trajectory(obs_T_Do, obsfeat_T_Df, adist_T_Pa, a_T_Da, r_T)
            trajbatch = TrajBatch.FromTrajs([single_traj])
            proto_logger.export_trajs(trajbatch, export_proto_filename)

            # Convert initial config dict to yaml file
            # Export yaml file using same stem as export_proto_filename
            export_proto_file = Path(export_proto_filename)
            yaml_filename = export_proto_file.parent / f"{export_proto_file.stem}.yaml"
            with open(yaml_filename, "w") as outfile:
                yaml.dump(export_init_config, outfile, default_flow_style=False)

    def run_insertion_reset_sequence(self):
        """
        Runs the insertion demo by either using the randomly generated positions
        or the positions/orientations specified by the action objects
        in the action sequence file
        """
        # Specify the desired action sequence file,
        # the action sequence within this file, and the action objects within this file
        action_sequence_name = "pickup_sequence"

        # self.initialize_action_objects_random(arm_name)
        if self.__initial_config is None:
            initial_config = self.initialize_action_objects_random()
        else:
            self.initialize_action_objects()
            initial_config = None

        # Run the sequence of all actions (main loop)
        self.run_insertion_sequence(self.__action_config[action_sequence_name])
        return initial_config


class InsertionTask(MujocoGymAppHighFidelity, BaseQuadInsertionTask):
    def __init__(self):
        # Initialize the Parent class with the config file
        robot_config_file = "quad_insert.yaml"
        scene_file = "quad_insert.xml"
        render_mode = "human"

        observation_space = Box(low=-np.inf, high=np.inf)
        action_space = Box(low=-100, high=100)

        MujocoGymAppHighFidelity.__init__(
            self,
            robot_config_file,
            scene_file,
            observation_space,
            action_space,
            render_mode=render_mode,
        )
        BaseQuadInsertionTask.__init__(self, render_expert=bool(render_mode == "human"))

    @property
    def default_start_pt(self):
        return None


# Main entrypoint
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--export_proto_filename", type=str, default=None)
    args = parser.parse_args()
    export_proto_filename = args.export_proto_filename

    # Initialize the insertion Task
    demo = InsertionTask()

    # Run the raw expert
    export_init_config = demo.run_insertion_reset_sequence()
    demo.run_insertion_main_sequence(export_proto_filename, export_init_config)
