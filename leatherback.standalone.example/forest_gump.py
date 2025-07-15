import os
import numpy as np
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

from isaacsim.core.utils.extensions import enable_extension
import carb
import omni.appwindow  # Contains handle to keyboard
from isaacsim.core.api import World
from isaacsim.core.utils.prims import define_prim, get_prim_at_path
from leatherback.policy.example.leatherback import LeatherbackPolicy
from isaacsim.storage.native import get_assets_root_path

script_dir = os.path.dirname(__file__)
relative_path = os.path.join("..", "leatherback")
full_path = os.path.abspath(os.path.join(script_dir, relative_path))
usd_path = os.path.abspath(os.path.join(full_path, "leatherback_simple_better.usd"))


import onnxruntime as rt
policy_path = os.path.join(script_dir, "../leatherback/policy_agent.onnx")
sess = rt.InferenceSession(policy_path)
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name

# pos_error, heading_error_cos, heading_error_sin, vel_x, vel_y, vel_z, omega_x, omega_y
def on_physics_step(step_size) -> None:
    X_test = np.array([[-6, 0, 0, 1, 0, 0, 1, 0.5]])  # Example input 
    pred_onx = sess.run([label_name], {input_name: X_test.astype(np.float32)})[0]
    print(pred_onx)

    #wheel_vel = 16
    #robot_art.set_joint_velocities([[wheel_vel, wheel_vel, 0, 0, wheel_vel, wheel_vel]])
    #robot_art.set_joint_positions([[0, 0, step_angle, -step_angle, 0, 0]])


my_world = World(stage_units_in_meters=1.0, physics_dt=1 / 60, rendering_dt=1 / 50)
assets_root_path = get_assets_root_path()

prim = define_prim("/World/Ground", "Xform")
asset_path = assets_root_path + "/Isaac/Environments/Grid/default_environment.usd"
prim.GetReferences().AddReference(asset_path)


lb_prim_path = "/World/Leatherback"
from isaacsim.core.utils.stage import add_reference_to_stage
add_reference_to_stage(usd_path, lb_prim_path)

from isaacsim.core.prims import Articulation
robot_art = Articulation(prim_paths_expr=lb_prim_path, name="Leatherback")

robot_art.set_world_poses(positions=np.array([[0,0,0.5]]))
my_world.reset() # required to have joints available

#print("joint names: ", robot_art.joint_names)
# just for ordering reference
wheel_rr = 'Wheel__Upright__Rear_Right'
wheel_rl = 'Wheel__Upright__Rear_Left'
knuckle_fr = 'Knuckle__Upright__Front_Right'
knuckle_fl = 'Knuckle__Upright__Front_Left'
wheel_knuckle_fr = 'Wheel__Knuckle__Front_Right'
wheel_knuckle_fl = 'Wheel__Knuckle__Front_Left'


my_world.add_physics_callback("physics_step", callback_fn=on_physics_step)

while simulation_app.is_running():
    my_world.step(render=True)

simulation_app.close()
