# How Articulations Work

Notes about how the simulator is controlling the articulations

## IsaacLab stuff

Inside the IsaacLab there is an example how the Isaaclab leverages the builtin APIs in IsaacSim to control the articulations in a programmatically defined number of environments.

Inside the articulation_data.py in the line 230:

```python
##
# Joint commands -- Set into simulation.
##

joint_pos_target: torch.Tensor = None
"""Joint position targets commanded by the user. Shape is (num_instances, num_joints).

For an implicit actuator model, the targets are directly set into the simulation.
For an explicit actuator model, the targets are used to compute the joint torques (see :attr:`applied_torque`),
which are then set into the simulation.
"""

joint_vel_target: torch.Tensor = None
"""Joint velocity targets commanded by the user. Shape is (num_instances, num_joints).

For an implicit actuator model, the targets are directly set into the simulation.
For an explicit actuator model, the targets are used to compute the joint torques (see :attr:`applied_torque`),
which are then set into the simulation.
"""

joint_effort_target: torch.Tensor = None
"""Joint effort targets commanded by the user. Shape is (num_instances, num_joints).

For an implicit actuator model, the targets are directly set into the simulation.
For an explicit actuator model, the targets are used to compute the joint torques (see :attr:`applied_torque`),
which are then set into the simulation.
"""

```

We can see that IsaacLab is sending the control_actions throught the method ```_apply_actuator_model()``` using a tensor instantiated in the object ```_data```

Inside the articulation.py in the line 1443:

```python
def _apply_actuator_model(self):
    """Processes joint commands for the articulation by forwarding them to the actuators.

    The actions are first processed using actuator models. Depending on the robot configuration,
    the actuator models compute the joint level simulation commands and sets them into the PhysX buffers.
    """
    # process actions per group
    for actuator in self.actuators.values():
        # prepare input for actuator model based on cached data
        # TODO : A tensor dict would be nice to do the indexing of all tensors together
        control_action = ArticulationActions(
            joint_positions=self._data.joint_pos_target[:, actuator.joint_indices],
            joint_velocities=self._data.joint_vel_target[:, actuator.joint_indices],
            joint_efforts=self._data.joint_effort_target[:, actuator.joint_indices],
            joint_indices=actuator.joint_indices,
        )
        # compute joint command from the actuator model
        control_action = actuator.compute(
            control_action,
            joint_pos=self._data.joint_pos[:, actuator.joint_indices],
            joint_vel=self._data.joint_vel[:, actuator.joint_indices],
        )
        # update targets (these are set into the simulation)
        if control_action.joint_positions is not None:
            self._joint_pos_target_sim[:, actuator.joint_indices] = control_action.joint_positions
        if control_action.joint_velocities is not None:
            self._joint_vel_target_sim[:, actuator.joint_indices] = control_action.joint_velocities
        if control_action.joint_efforts is not None:
            self._joint_effort_target_sim[:, actuator.joint_indices] = control_action.joint_efforts
        # update state of the actuator model
        # -- torques
        self._data.computed_torque[:, actuator.joint_indices] = actuator.computed_effort
        self._data.applied_torque[:, actuator.joint_indices] = actuator.applied_effort
        # -- actuator data
        self._data.soft_joint_vel_limits[:, actuator.joint_indices] = actuator.velocity_limit
        # TODO: find a cleaner way to handle gear ratio. Only needed for variable gear ratio actuators.
        if hasattr(actuator, "gear_ratio"):
            self._data.gear_ratio[:, actuator.joint_indices] = actuator.gear_ratio
```

The object self.actuators is a dictionary where the Keys are 

```python
actuators: dict[str, ActuatorBase]
"""Dictionary of actuator instances for the articulation.

The keys are the actuator names and the values are the actuator instances. The actuator instances
are initialized based on the actuator configurations specified in the :attr:`ArticulationCfg.actuators`
attribute. They are used to compute the joint commands during the :meth:`write_data_to_sim` function.
"""
```

It can only be a type of joint like joint_positions or joint_velocities or joint_efforts and its associated joint_indices.

```python
joint_positions=self._data.joint_pos_target[:, actuator.joint_indices]
joint_velocities=self._data.joint_vel_target[:, actuator.joint_indices]
joint_efforts=self._data.joint_effort_target[:, actuator.joint_indices]
```

ArticulationData is inside the articulation_data.py at isaaclab assets articulation


