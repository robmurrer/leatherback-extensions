from isaacsim import SimulationApp

# Start the application
simulation_app = SimulationApp({"headless": False})

# # Get the utility to enable extensions
from isaacsim.core.utils.extensions import enable_extension

# for some reason it cannot find the extension by its name
enable_extension("leatherback.policy.example")

# simulation_app.update()

import carb
import numpy as np
import omni.appwindow  # Contains handle to keyboard
from isaacsim.core.api import World
from isaacsim.core.utils.prims import define_prim, get_prim_at_path

from leatherback.policy.example.leatherback import LeatherbackPolicy
# the reference extension
# from isaacsim.robot.policy.examples.robots import SpotFlatTerrainPolicy

from isaacsim.storage.native import get_assets_root_path

import os
script_dir = os.path.dirname(__file__)
relative_path = os.path.join("..", "leatherback")
full_path = os.path.abspath(os.path.join(script_dir, relative_path))
usd_path = os.path.abspath(os.path.join(full_path, "leatherback_simple_better.usd"))

first_step = True
reset_needed = False

# initialize robot on first step, run robot advance
def on_physics_step(step_size) -> None:
    global first_step
    global reset_needed
    if first_step:
        spot.initialize()
        first_step = False
    elif reset_needed:
        my_world.reset(True)
        reset_needed = False
        first_step = True
    else:
        spot.forward(step_size, base_command)


# spawn world
my_world = World(stage_units_in_meters=1.0, physics_dt=1 / 500, rendering_dt=1 / 50)
assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not find Isaac Sim assets folder")

# spawn warehouse scene
prim = define_prim("/World/Ground", "Xform")
asset_path = assets_root_path + "/Isaac/Environments/Grid/default_environment.usd"
prim.GetReferences().AddReference(asset_path)

# spawn robot
"""
Initialize robot and load RL policy.

Args:
    prim_path (str) -- prim path of the robot on the stage
    root_path (Optional[str]): The path to the articulation root of the robot
    name (str) -- name of the quadruped
    usd_path (str) -- robot usd filepath in the directory if none it gets from Nucleus
    position (np.ndarray) -- position of the robot
    orientation (np.ndarray) -- orientation of the robot

"""
spot = LeatherbackPolicy(
    prim_path="/World/leatherback",
    name="leatherback",
    policy_path = full_path,
    usd_path = usd_path,
    position=np.array([0, 0, 0.8]),
)
my_world.reset()
my_world.add_physics_callback("physics_step", callback_fn=on_physics_step)

# robot command
# The position of the waypoint in X , Y , Z
base_command = np.zeros(3)

i = 0
while simulation_app.is_running():
    my_world.step(render=True)
    if my_world.is_stopped():
        reset_needed = True
    if my_world.is_playing():
        # The idea is to drive in a triangle
        if i >= 0 and i < 80:
            # X1
            base_command = np.array([4, 0, 0])
        elif i >= 80 and i < 130:
            # X2
            base_command = np.array([2, 4, 0])
        elif i >= 130 and i < 200:
            # X3
            base_command = np.array([0, 0, 0])
        elif i == 200:
            i = 0
        i += 1

simulation_app.close()
