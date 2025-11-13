import os
import hydra
import rclpy
import torch
import time
import math
import argparse
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on running the cartpole RL environment.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
from crazyfly.cf_env import QuadcopterEnvCfg, camera_follow,debug_draw
import env.sim_env as sim_env
import crazyfly.cf_sensors as cf_sensors
import omni
import carb
import crazyfly.cf_ctrl as cf_ctrl
import ros2.cf_ros2_bridge as cf_ros2_bridge

FILE_PATH = os.path.join(os.path.dirname(__file__), "cfg")
@hydra.main(config_path=FILE_PATH, config_name="drone", version_base=None)
def run_simulator(cfg):

    # Go2 Environment setup
    cf_env_cfg = QuadcopterEnvCfg()
    cf_env_cfg.scene.num_envs = cfg.num_envs
    cf_env_cfg.decimation = math.ceil(1./cf_env_cfg.sim.dt/cfg.freq)
    cf_env_cfg.sim.render_interval = cf_env_cfg.decimation
    cf_ctrl.init_base_vel_cmd(cfg.num_envs)
    # env, policy = go2_ctrl.get_rsl_flat_policy(go2_env_cfg)
    env= cf_ctrl.get_envs(cf_env_cfg)

    # Simulation environment
    if (cfg.env_name == "obstacle-dense"):
        sim_env.create_obstacle_dense_env() # obstacles dense
    elif (cfg.env_name == "obstacle-medium"):
        sim_env.create_obstacle_medium_env() # obstacles medium
    elif (cfg.env_name == "obstacle-sparse"):
        sim_env.create_obstacle_sparse_env() # obstacles sparse
    elif (cfg.env_name == "warehouse"):
        sim_env.create_warehouse_env() # warehouse
    elif (cfg.env_name == "warehouse-forklifts"):
        sim_env.create_warehouse_forklifts_env() # warehouse forklifts
    elif (cfg.env_name == "warehouse-shelves"):
        sim_env.create_warehouse_shelves_env() # warehouse shelves
    elif (cfg.env_name == "full-warehouse"):
        sim_env.create_full_warehouse_env() # full warehouse

    # Sensor setup
    sm = cf_sensors.SensorManager(cfg.num_envs)
    lidar_annotators = sm.add_rtx_lidar()
    cameras = sm.add_camera(10)

    # Keyboard control
    system_input = carb.input.acquire_input_interface()
    system_input.subscribe_to_keyboard_events(
        omni.appwindow.get_default_app_window().get_keyboard(), cf_ctrl.sub_keyboard_event)
    
    # ROS2 Bridge
    rclpy.init()
    dm = cf_ros2_bridge.RobotDataManager(env, lidar_annotators, cameras, cfg)
    # draw = _debug_draw.acquire_debug_draw_interface()
    # robots_action_list = [[]] * cfg.num_envs
    # Run simulation
    sim_step_dt = float(cf_env_cfg.sim.dt * cf_env_cfg.decimation)
    obs, _ = env.reset()
    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():            
            # control joints
            actions = cf_ctrl.base_vel_cmd(env)

            # step the environment
            obs, _, _, _ = env.step(actions)

            # # ROS2 data
            dm.pub_ros2_data()
            rclpy.spin_once(dm)

            ### record the traj
            # data = env.unwrapped.scene.articulations["robot"].data
            # for idx in range(cfg.num_envs):
            #     if len(robots_action_list[idx]) > 100000:
            #         robots_action_list[idx].pop(0)
            #     robots_action_list[idx].append(data.root_state_w[idx, :7])
            # debug_draw(env.unwrapped.draw, robots_action_list)
            # Camera follow
            if (cfg.camera_follow):
                camera_follow(env)
            # limit loop time
            elapsed_time = time.time() - start_time
            if elapsed_time < sim_step_dt:
                sleep_duration = sim_step_dt - elapsed_time
                time.sleep(sleep_duration)


        ## visualize the traj

        # debug_draw(draw, robots_action_list)
        actual_loop_time = time.time() - start_time
        rtf = min(1.0, sim_step_dt/elapsed_time)
        print(f"\rStep time: {actual_loop_time*1000:.2f}ms, Real Time Factor: {rtf:.2f}", end='', flush=True)
    
    dm.destroy_node()
    rclpy.shutdown()
    simulation_app.close()




if __name__ == "__main__":
    run_simulator()
    