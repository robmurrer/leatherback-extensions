from typing import Optional

import numpy as np
import omni
import omni.kit.commands
from isaacsim.core.utils.rotations import quat_to_rot_matrix
from isaacsim.core.utils.types import ArticulationAction

# must experiment with the policy controller - general vs bespoke
from leatherback.policy.example.controllers import PolicyController

from isaacsim.storage.native import get_assets_root_path

class LeatherbackPolicy(PolicyController):
    """The Leatherback racer"""

    def __init__(
        self,
        prim_path: str,
        root_path: Optional[str] = None,
        name: str = "spot",
        usd_path: Optional[str] = None,
        policy_path: Optional[str] = None,
        position: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
    ) -> None:
        """
        Initialize robot and load RL policy.

        Args:
            prim_path (str) -- prim path of the robot on the stage
            root_path (Optional[str]): The path to the articulation root of the robot
            name (str) -- name of the quadruped
            usd_path (str) -- robot usd filepath in the directory
            policy_path (str) -- 
            position (np.ndarray) -- position of the robot
            orientation (np.ndarray) -- orientation of the robot

        """
        assets_root_path = get_assets_root_path()
        if usd_path == None:
            # Later might be able to add an Asset to show it doesnt exist - easter egg
            # usd_path = assets_root_path + "/Isaac/Robots/BostonDynamics/spot/spot.usd"
            print("File not found")
        
        super().__init__(name, prim_path, root_path, usd_path, policy_path, position, orientation)

        if policy_path == None:
            # self.load_policy(
            #     assets_root_path + "/Isaac/Samples/Policies/Spot_Policies/spot_policy.pt",
            #     assets_root_path + "/Isaac/Samples/Policies/Spot_Policies/spot_env.yaml",
            # )
            print("Policy not found")
        else:
            self.load_policy(
                    policy_path + "/policy_agent.onnx", # policy_path + "/spot_policy.pt",
                    policy_path + "/env.yaml",  # policy_path + "/spot_env.yaml",
                ) 
    
        self._action_scale = 1 # 0.2
        # Leatherback has action space = 2
        self._previous_action = np.zeros(2)
        self._policy_counter = 0

    # region Observation
    # This need to be replaced to something similar like the IsaacLab get observations
    def _compute_observation(self, command):
        """
        Compute the observation vector for the policy

        Argument:
        command (np.ndarray) -- the waypoint goal (x, y, z)

        Returns:
        np.ndarray -- The observation vector.

        Observations:
        position error
        heading error cosine
        heading error sine
        root_lin_vel_b[:, 0]
        root_lin_vel_b[:, 1]
        root_ang_vel_w[:, 2]
        _throttle_state
        _steering_state

        """
        # this is using the articulation type 
        # from isaacsim.core.prims import SingleArticulation
        # self.robot = SingleArticulation(prim_path=prim_path, name=name, position=position, orientation=orientation)
        lin_vel_I = self.robot.get_linear_velocity()
        ang_vel_I = self.robot.get_angular_velocity()
        # position, orientation = prim.get_world_pose()
        pos_IB, q_IB = self.robot.get_world_pose() 
        
        R_IB = quat_to_rot_matrix(q_IB)
        R_BI = R_IB.transpose()
        lin_vel_b = np.matmul(R_BI, lin_vel_I)
        ang_vel_b = np.matmul(R_BI, ang_vel_I)
        gravity_b = np.matmul(R_BI, np.array([0.0, 0.0, -1.0]))

        # region pos_error
        # Calculate the Position Error
        # current_target_positions = self._target_positions[self.leatherback._ALL_INDICES, self._target_index]
        # self._position_error_vector = current_target_positions - self.leatherback.data.root_pos_w[:, :2]
        # self._position_error = torch.norm(self._position_error_vector, dim=-1) # had placed dim=1
        # should it be np.linalg.norm()
        _position_error_vector = command - pos_IB
        _position_error = np.linalg.norm(_position_error_vector) # , axis=-1
        # _position_error = 0
        # end region pos_error
        # region heading_error
        # Calculate the Heading Error
        
        # FORWARD_VEC_B = torch.tensor((1.0, 0.0, 0.0), device=self.device).repeat(self._root_physx_view.count, 1)
        # Create FORWARD_VEC_B in NumPy
        # FORWARD_VEC_B = np.tile(np.array([1.0, 0.0, 0.0]), (self._root_physx_view.count, 1))  # Shape: (N, 3)
        FORWARD_VEC_B = np.array([1.0, 0.0, 0.0]) # Do I need _root_physx_view ?
        quat = q_IB.reshape(-1, 4)
        vec = FORWARD_VEC_B.reshape(-1, 3)
        xyz = quat[:, 1:]
        # In torch: t = xyz.cross(vec, dim=-1) * 2
        # Cross product: t = 2 * cross(xyz, vec)
        t = 2 * np.cross(xyz, vec)

        # forward_w = (vec + quat[:, 0:1] * t + xyz.cross(t, dim=-1)).view(shape)
        # Compute rotated vector: forward_w = vec + w * t + cross(xyz, t)
        w = quat[:, 0:1]  # shape (N, 1)
        forward_w = vec + w * t + np.cross(xyz, t)

        # heading_w = torch.atan2(forward_w[:, 1], forward_w[:, 0])
        # Get heading from world-frame forward vector (X and Y components)
        heading_w = np.arctan2(forward_w[:, 1], forward_w[:, 0])  # shape (N,)

        target_heading_w = np.arctan2(command[1]-pos_IB[1], command[0]-pos_IB[0])
        _heading_error = target_heading_w - heading_w

        # end region heading_error
        
        # throttle state
        # which joint is the throttle and will this return the right values ?
        # must define the joint_indices
        # current_joint_vel = self.robot.get_joint_velocities()
        # initiate the throttle
        # throttle_scale = 1
        # throttle_action = np.repeat(self.action[:, 0], 4).reshape(-1, 4) * throttle_scale

        # The order of observations matter
        # leatherback has 8 observations
        obs = np.zeros(8)
        # Position Error
        obs[:1] = _position_error
        # Heading error
        obs[1:2] = np.cos(_heading_error)[:, np.newaxis]
        obs[2:3] = np.sin(_heading_error)[:, np.newaxis]
        # Linear Velocity X and Y
        obs[3:4] = lin_vel_b[0]
        obs[4:5] = lin_vel_b[1]
        # Angular Velocity vZ
        obs[5:6] = ang_vel_b[2]
        # _throttle_state
        obs[6:7] = self._previous_action[0]
        # _steering_state
        # which joint is steering and will this return the right value ?
        # current_joint_pos = self.robot.get_joint_positions()
        obs[7:] = self._previous_action[1]

        return obs

    def forward(self, dt, command):
        """
        Compute the desired torques and apply them to the articulation

        Argument:
        dt (float) -- Timestep update in the world.
        command (np.ndarray) -- the robot command (v_x, v_y, w_z)

        """
        if self._policy_counter % self._decimation == 0:
            obs = self._compute_observation(command)
            self.action, self.actions = self._compute_action(obs)
            self.repeated_arr = np.repeat(self.action, [4, 2])
            self._previous_action = self.action.copy()
        # action = ArticulationAction(joint_positions=self.default_pos + (self.repeated_arr * self._action_scale))
        # action = ArticulationAction(joint_velocities = self.actions.joint_velocities, joint_positions = self.actions.joint_positions)
        # self.robot.apply_action(action)

        self.robot.apply_wheel_actions(self.actions)

        self._policy_counter += 1
