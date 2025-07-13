# Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
import re
from typing import Optional

import carb
import numpy as np
from isaacsim.core.api.robots.robot import Robot
from isaacsim.core.utils.prims import define_prim, get_prim_at_path
from isaacsim.core.utils.types import ArticulationAction


class AckermannRobot(Robot):
    """
    This class wraps and manages the articualtion for an ackermann wheeled robot. It is designed to be managed by the `World` simulation context and provides and API for applying actions, retrieving dof parameters, etc...

    Creating a wheeled robot can be done in a number of different ways, depending on the use case.

    * Most commonly, the robot and stage are preloaded, in which case only the prim path to the articulation root and the joint labels are required. 
    
    Joint labels can take the form of either the joint names or the joint indices in the articulation. They must match their respective joint type also steering = position and throttle = velocity

    .. code-block:: python

        leatherabck = AckermannRobot(prim_path="/path/to/leatherback",
                            name="Goat",
                            wheel_dof_names=["left_wheel_joint", "right_wheel_joint"]
                            )

        armbot = WheeledRobot(prim_path="path/to/armbot"
                                name="Weird_Arm_On_Wheels_Bot",
                                wheel_dof_indices=[7, 8]
                            )

    * Alternatively, this class can create and populate a new reference on the stage.  This is done with the `create_robot` parameter set to True.

    .. code-block:: python

        carter = WheeledRobot(prim_path="/desired/path/to/carter",
                                name = "Jimmy",
                                wheel_dof_names=["joint_wheel_left", "joint_wheel_right"],
                                create_robot = True,
                                usd_path = "/Isaac/Robots/NVIDIA/Carter/nova_carter/nova_carter.usd",
                                position = np.array([0,1,0])
                            )

    .. hint::

        In all cases, either `wheel_dof_names` or `wheel_dof_indices` must be specified!


    Args:
        prim_path (str):                              The stage path to the root prim of the articulation.
        name ([str]):                                 The name used by `World` to identify the robot. Defaults to "wheeled_robot"
        robot_path (optional[str]):                   relative path from prim path to the robot.
        wheel_dof_names (semi-optional[str]):         names of the wheel joints. Either this or the wheel_dof_indicies must be specified.
        wheel_dof_indices: (semi-optional[int]):      indices of the wheel joints in the articulation. Either this or the wheel_dof_names must be specified.
        usd_path (optional[str]):                     nucleus path to the robot asset. Used if create_robot is True
        create_robot (bool):                          create robot at prim_path using asset from usd_path. Defaults to False
        position (Optional[np.ndarray], optional):    The location to create the robot when create_robot is True. Defaults to None.
        orientation (Optional[np.ndarray], optional): The orientation of the robot when crate_robot is True. Defaults to None.
    """

# Instead of wheel_dof_names must split into the throttle_dof_name and steering_dof_name
    def __init__(
        self,
        prim_path: str,
        name: str, #= "wheeled_robot",
        robot_path: Optional[str] = None,
        # wheel_dof_names: Optional[str] = None,
        throttle_dof_names: Optional[str] = None,
        steering_dof_names: Optional[str] = None,
        throttle_dof_indices: Optional[int] = None,
        steering_dof_indices: Optional[int] = None,
        # wheel_dof_indices: Optional[int] = None,
        usd_path: Optional[str] = None,
        create_robot: Optional[bool] = False,
        position: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
    ) -> None:
        prim = get_prim_at_path(prim_path)
        if not prim.IsValid():
            if create_robot:
                prim = define_prim(prim_path, "Xform")
                if usd_path:
                    prim.GetReferences().AddReference(usd_path)
                else:
                    carb.log_error("no valid usd path defined to create new robot")
            else:
                carb.log_error("no prim at path %s", prim_path)

        if robot_path is not None:
            robot_path = "/" + robot_path
            # regex: remove all prefixing "/", need at least one prefix "/" to work
            robot_path = re.sub("^([^\/]*)\/*", "", "/" + robot_path)
            prim_path = prim_path + "/" + robot_path

        super().__init__(
            prim_path=prim_path, name=name, position=position, orientation=orientation, articulation_controller=None
        )
        # self._wheel_dof_names = wheel_dof_names
        self._throttle_dof_names = throttle_dof_names
        self._steering_dof_names = steering_dof_names
        self._throttle_dof_indices = throttle_dof_indices
        self._steering_dof_indices = steering_dof_indices
        # might remove the _wheel_dof_indices
        # self._wheel_dof_indices = wheel_dof_indices
        # TODO: check the default state and how to reset
        return

    # region remove
    # might remove the _wheel_dof_indices
    # @property
    # def wheel_dof_indices(self):
    #     """[summary]

    #     Returns:
    #         int: [description]
    #     """
    #     return self._wheel_dof_indices
    
    @property
    def throttle_dof_indices(self):
        """[summary]

        Returns:
            int: [description]
        """
        return self._throttle_dof_indices
    
    @property
    def steering_dof_indices(self):
        """[summary]

        Returns:
            int: [description]
        """
        return self._steering_dof_indices

    def get_wheel_positions(self):
        """[summary]

        Returns:
            List[float]: [description]
        """
        full_dofs_positions = self.get_joint_positions()
        wheel_joint_positions = [full_dofs_positions[i] for i in self._steering_dof_indices]
        return wheel_joint_positions

    def set_wheel_positions(self, positions) -> None:
        """[summary]

        Args:
            positions (Tuple[float, float]): [description]
        """
        full_dofs_positions = [None] * self._num_steering_dof
        for i in range(self._num_steering_dof):
            full_dofs_positions[self._steering_dof_indices[i]] = positions[i]
        self.set_joint_positions(positions=np.array(full_dofs_positions))
        return

    def get_wheel_velocities(self):
        """[summary]

        Returns:
            Tuple[np.ndarray, np.ndarray]: [description]
        """

        full_dofs_velocities = self.get_joint_velocities()
        wheel_dof_velocities = [full_dofs_velocities[i] for i in self._throttle_dof_indices]
        return wheel_dof_velocities

    def set_wheel_velocities(self, velocities) -> None:
        """[summary]

        Args:
            velocities (Tuple[float, float]): [description]
        """
        full_dofs_velocities = [None] * self._num_wheel_dof
        for i in range(self._num_wheel_dof):
            full_dofs_velocities[self._throttle_dof_indices[i]] = velocities[i]
        self.set_joint_velocities(velocities=np.array(full_dofs_velocities))
        return

    def apply_wheel_actions(self, actions: ArticulationAction) -> None:
        """applying action to the wheels to move the robot

        Args:
            actions (ArticulationAction): [description]
        
        ArticulationActions has the following variables
            self.joint_positions = joint_positions
            self.joint_velocities = joint_velocities
            self.joint_efforts = joint_efforts
            self.joint_indices = joint_indices
        
        joint_indices (Optional[Union[np.ndarray, List, torch.Tensor, wp.array]], optional): joint indices to specify which joints
                                                                                 to manipulate. Shape (K,).
                                                                                 Where K <= num of dofs.
                                                                                 Defaults to None (i.e: all dofs).
        joint_names (Optional[List[str]]): joint names to specify which joints to manipulate
                                            (can't be sppecified together with joint_indices). Shape (K,).
                                            Where K <= num of dofs. Defaults to None (i.e: all dofs).
        In the Arituclation Controler
        self._articulation_view.apply_action(
            ArticulationActions(
                joint_positions=joint_positions,
                joint_velocities=joint_velocities,
                joint_efforts=joint_efforts,
                joint_indices=control_actions.joint_indices,
            )
        )
        """
        actions_length = actions.get_length()
        if actions_length is not None and actions_length != 4:#(self._num_wheel_dof + self._num_steering_dof):
            raise Exception("ArticulationAction passed should be the same length as the number of wheels")
        joint_actions = ArticulationAction()
        # TODO: The Steering, Throttle and Efforts must be inside a for loop. Can only send Apply Action to a specific joint type and its indices at a time.

        
        # Steering
        # joint_actions.joint_positions[self._steering_dof_indices[i]] = actions.joint_positions[i]
        # IndexError: index 2 is out of bounds for axis 0 with size 2
        # joint_actions has variables like the joint_positions and joint_velocities, it is not giving the indices
        if actions.joint_positions is not None: # must change this to only change the steering positions
            joint_actions.joint_positions = np.zeros(self._num_steering_dof)  
            print(actions) # {'joint_positions': (-0.6287975571457335, -0.7854), 'joint_velocities': (20.0, 20.0, 20.0, 12.499972450911129), 'joint_efforts': None}
            print(self._steering_dof_indices) # [2, 3]
            print(self._throttle_dof_indices) # [5, 4, 0, 1]
            for i in range(self._num_steering_dof):  
                # joint_actions.joint_positions[self._steering_dof_indices[i]] = actions.joint_positions[i]
                # joint_actions.joint_indices[i] = self._steering_dof_indices[i]
                joint_actions.joint_positions[i] = actions.joint_positions[i]
        # Throttle
        if actions.joint_velocities is not None: # must change this to change only the throttle
            joint_actions.joint_velocities = np.zeros(self._num_wheel_dof)
            for i in range(self._num_wheel_dof):
                # joint_actions.joint_velocities[self._throttle_dof_indices[i]] = actions.joint_velocities[i]
                joint_actions.joint_velocities[i] = actions.joint_velocities[i]
        # region review
        # might need remove
        # if actions.joint_efforts is not None:
        #     joint_actions.joint_efforts = np.zeros(self.num_dof)
        #     for i in range(self._num_wheel_dof):
        #         joint_actions.joint_efforts[self._wheel_dof_indices[i]] = actions.joint_efforts[i]
        
        # Apply all actions to the respective joints
        self.apply_action(control_actions=joint_actions)
        return

    def initialize(self, physics_sim_view=None) -> None:
        """
        _wheel_dof_indices replaced by _throttle_dof_indices and _steering_dof_indices
        _throttle_dof_indices the joint indices related to the _throttle_dof_names
        
        """
        super().initialize(physics_sim_view=physics_sim_view)
        if self._throttle_dof_names is not None:
            self._throttle_dof_indices = [
                self.get_dof_index(self._throttle_dof_names[i]) for i in range(len(self._throttle_dof_names))
            ]
        if self._steering_dof_names is not None:
            self._steering_dof_indices = [
                self.get_dof_index(self._steering_dof_names[i]) for i in range(len(self._steering_dof_names))
            ]
        elif self._throttle_dof_indices or self._steering_dof_indices is None:
            carb.log_error("need to have either joint names or joint indices")

        self._num_wheel_dof = len(self._throttle_dof_indices)
        self._num_steering_dof = len(self._steering_dof_indices)

        return

    def post_reset(self) -> None:
        """[summary]"""
        super().post_reset()
        self._articulation_controller.switch_control_mode(mode="velocity")
        return
    
    # Confused about what this is useful for
    def get_articulation_controller_properties(self):
        return self._throttle_dof_names, self._steering_dof_names, self._steering_dof_indices, self._throttle_dof_indices
