
import io
from typing import Optional

import carb
import numpy as np
import omni
import torch
from isaacsim.core.api.controllers.base_controller import BaseController
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.prims import define_prim, get_prim_at_path

from .config_loader import get_articulation_props, get_physics_properties, get_robot_joint_properties, parse_env_config

# Adding the ONNX runtime
import os
import onnxruntime as ort

class PolicyController(BaseController):
    """
    A controller that loads and executes a policy from a file.

    Args:
        name (str): The name of the controller.
        prim_path (str): The path to the prim in the stage.
        root_path (Optional[str], None): The path to the articulation root of the robot
        usd_path (Optional[str], optional): The path to the USD file. Defaults to None.
        position (Optional[np.ndarray], optional): The initial position of the robot. Defaults to None.
        orientation (Optional[np.ndarray], optional): The initial orientation of the robot. Defaults to None.

    Attributes:
        robot (SingleArticulation): The robot articulation.
    """

    def __init__(
        self,
        name: str,
        prim_path: str,
        root_path: Optional[str] = None,
        usd_path: Optional[str] = None,
        policy_path: Optional[str] = None,
        position: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
    ) -> None:
        prim = get_prim_at_path(prim_path)

        if not prim.IsValid():
            prim = define_prim(prim_path, "Xform")
            if usd_path:
                prim.GetReferences().AddReference(usd_path)
            else:
                carb.log_error("unable to add robot usd, usd_path not provided")

        if root_path == None:
            self.robot = SingleArticulation(prim_path=prim_path, name=name, position=position, orientation=orientation)
        else:
            self.robot = SingleArticulation(prim_path=root_path, name=name, position=position, orientation=orientation)

    def load_policy(self, policy_file_path, policy_env_path) -> None:
        """
        Loads a policy from a file.

        Args:
            policy_file_path (str): The path to the policy file. Example: spot_policy.pt
            policy_env_path (str): The path to the environment configuration file. Example: spot_env.yaml
        """
        if policy_file_path.endswith('.pt') or policy_file_path.endswith('.pth'):
            file_content = omni.client.read_file(policy_file_path)[2]
            file = io.BytesIO(memoryview(file_content).tobytes())
            # Loading a Torch JIT file for Inference
            self.policy = torch.jit.load(file)
            self._isJIT = 1
        # region ONNX
        # Add here an option for ONNX inference
        elif policy_file_path.endswith('.onnx'):
            # Unnecessary Byte stream for now
            file_content = omni.client.read_file(policy_file_path)[2]
            file = io.BytesIO(memoryview(file_content).tobytes())
            # Load ONNX model 
            self.session = ort.InferenceSession(policy_file_path)
            self._isJIT = 0
        # end of region ONNX
        self.policy_env_params = parse_env_config(policy_env_path)

        self._decimation, self._dt, self.render_interval = get_physics_properties(self.policy_env_params)

    def initialize(
        self,
        physics_sim_view: omni.physics.tensors.SimulationView = None,
        effort_modes: str = "force",
        control_mode: str = "position",
        set_gains: bool = True,
        set_limits: bool = True,
        set_articulation_props: bool = True,
    ) -> None:
        """
        Initializes the robot and sets up the controller.

        Args:
            physics_sim_view (optional): The physics simulation view.
            effort_modes (str, optional): The effort modes. Defaults to "force".
            control_mode (str, optional): The control mode. Defaults to "position".
            set_gains (bool, optional): Whether to set the joint gains. Defaults to True.
            set_limits (bool, optional): Whether to set the limits. Defaults to True.
            set_articulation_props (bool, optional): Whether to set the articulation properties. Defaults to True.
        """
        self.robot.initialize(physics_sim_view=physics_sim_view)
        self.robot.get_articulation_controller().set_effort_modes(effort_modes)
        self.robot.get_articulation_controller().switch_control_mode(control_mode)
        max_effort, max_vel, stiffness, damping, self.default_pos, self.default_vel = get_robot_joint_properties(
            self.policy_env_params, self.robot.dof_names
        )
        if set_gains:
            self.robot._articulation_view.set_gains(stiffness, damping)
        if set_limits:
            self.robot._articulation_view.set_max_efforts(max_effort)
            self.robot._articulation_view.set_max_joint_velocities(max_vel)
        if set_articulation_props:
            self._set_articulation_props()

    def _set_articulation_props(self) -> None:
        """
        Sets the articulation root properties from the policy environment parameters.
        """
        articulation_prop = get_articulation_props(self.policy_env_params)

        solver_position_iteration_count = articulation_prop.get("solver_position_iteration_count")
        solver_velocity_iteration_count = articulation_prop.get("solver_velocity_iteration_count")
        stabilization_threshold = articulation_prop.get("stabilization_threshold")
        enabled_self_collisions = articulation_prop.get("enabled_self_collisions")
        sleep_threshold = articulation_prop.get("sleep_threshold")

        if solver_position_iteration_count not in [None, float("inf")]:
            self.robot.set_solver_position_iteration_count(solver_position_iteration_count)
        if solver_velocity_iteration_count not in [None, float("inf")]:
            self.robot.set_solver_velocity_iteration_count(solver_velocity_iteration_count)
        if stabilization_threshold not in [None, float("inf")]:
            self.robot.set_stabilization_threshold(stabilization_threshold)
        if isinstance(enabled_self_collisions, bool):
            self.robot.set_enabled_self_collisions(enabled_self_collisions)
        if sleep_threshold not in [None, float("inf")]:
            self.robot.set_sleep_threshold(sleep_threshold)

    # This is general, it is getting the Observations and returning the inference output
    def _compute_action(self, obs: np.ndarray) -> np.ndarray:
        """
        Computes the action from the observation using the loaded policy.

        Args:
            obs (np.ndarray): The observation.

        Returns:
            np.ndarray: The action.
        """
        if self._isJIT == 1:
            with torch.no_grad():
                obs = torch.from_numpy(obs).view(1, -1).float()
                action = self.policy(obs).detach().view(-1).numpy()
        # region ONNX
        # Add support to compute actions using the ONNX runtime   
        # Need to fix the Tensor giving output and input need o match     
        elif self._isJIT == 0:
            # Prepare inputs assuming input_tensor is a single input
            obs = torch.from_numpy(obs).view(1, -1).float() # seems reduntant but I thought I had to mess with data types so left here
            ort_inputs = {self.session.get_inputs()[0].name: obs.numpy()}
            output_names = [output.name for output in self.session.get_outputs()]
            outputs = self.session.run(output_names, ort_inputs)
            # Get output and flatten to 1D array like .view(-1).numpy()
            action = outputs[0].reshape(-1)
        # end region ONNX
        
        return action

    # These are implemented in the leatherback/leatherback.py
    def _compute_observation(self) -> NotImplementedError:
        """
        Computes the observation. Not implemented.
        """

        raise NotImplementedError(
            "Compute observation need to be implemented, expects np.ndarray in the structure specified by env yaml"
        )

    # These are implemented in the leatherback/leatherback.py
    def forward(self) -> NotImplementedError:
        """
        Forwards the controller. Not implemented.
        """
        raise NotImplementedError(
            "Forward needs to be implemented to compute and apply robot control from observations"
        )

    def post_reset(self) -> None:
        """
        Called after the controller is reset.
        """
        self.robot.post_reset()

# TODO
# Experiment with bindings, iostreams and better logic for the ONNX vs JIT
    # # Create an ONNX Runtime session with the provided model
    # def create_session(model: str) -> onnxruntime.InferenceSession:
    #     providers = ['CPUExecutionProvider']
    #     if torch.cuda.is_available():
    #         providers.insert(0, 'CUDAExecutionProvider')
    #     return onnxruntime.InferenceSession(model, providers=providers)