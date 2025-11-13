from isaaclab.scene import InteractiveSceneCfg
# from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG
from isaaclab_assets.robots.quadcopter import CRAZYFLIE_CFG
from isaaclab.sensors import RayCasterCfg, patterns, ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg, AssetBaseCfg,Articulation
import isaaclab.sim as sim_utils
import torch
from isaaclab.sim import SimulationCfg
from isaaclab.envs import ManagerBasedRLEnvCfg,DirectRLEnvCfg,DirectRLEnv
from isaacsim.core.utils.viewports import set_camera_view
import numpy as np
from scipy.spatial.transform import Rotation as R
# import cf_ctrl as cf_ctrl
from isaaclab.utils.math import quat_apply
from isaacsim.util.debug_draw import _debug_draw

@configclass
class CFSimCfg(InteractiveSceneCfg):
    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(color=(0.1, 0.1, 0.1), size=(300.0, 300.0)),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0, 0, 1e-4)
        )
    )

    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


    # Crazy_fly Robot
    Crazy_fly: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="{ENV_REGEX_NS}/CF")
    # Go2 foot contact sensor
    # contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/CF/.*", history_length=2, track_air_time=False)

@configclass
class QuadcopterEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 100
    decimation = 1
    action_space = 3
    observation_space = 20
    state_space = 0
    debug_vis = True

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        gravity= (0.0, 0.0, 0.0),
        # disable_contact_processing=True,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # scene
    scene: InteractiveSceneCfg = CFSimCfg(num_envs=2, env_spacing=2.5)


    # reward scales
    lin_vel_reward_scale = -0.05
    ang_vel_reward_scale = -0.01
    distance_to_goal_reward_scale = 15.0


class QuadcopterEnv(DirectRLEnv):
    cfg: QuadcopterEnvCfg

    def __init__(self, cfg: QuadcopterEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Total thrust and moment applied to the base of the quadcopter
        self._actions = torch.zeros(self.num_envs, 3, device=self.device)
        # Goal position
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        # Get specific body indices
        self._body_id = self._robot.find_bodies("body")[0]
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()
        self.draw = _debug_draw.acquire_debug_draw_interface()
        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)

    def _setup_scene(self):
        self._robot = Articulation(self.scene.cfg.Crazy_fly)
        self.scene.articulations["robot"] = self._robot
        # self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        # self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        # self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # add lights
        # light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        # light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        vel_xyz = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        vel_xyz[:, [0,2]] = actions[:, [0,1]]
        world_vel = quat_apply(self._robot.data.root_state_w[:, 3:7], vel_xyz)
        self.raw_actions_drone = actions
        world_vel_yaw = torch.zeros_like(world_vel)
        world_vel_yaw[:, 2] = actions[:,2]
        self.actions = torch.cat((world_vel, world_vel_yaw), dim=-1)


    def _apply_action(self):
        self._robot.write_root_velocity_to_sim(self.actions)

    def _get_observations(self) -> dict:
        obs = torch.cat(
            [
                self._robot.data.root_lin_vel_b,
                self._robot.data.root_ang_vel_b,
                self._robot.data.projected_gravity_b,
            ],
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        return None

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # time_out = self.episode_length_buf >= self.max_episode_length - 1
        # died = torch.logical_or(self._robot.data.root_pos_w[:, 2] < 0.1, self._robot.data.root_pos_w[:, 2] > 10.0)
        return False, False

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0
        # Sample new commands
        self._desired_pos_w[env_ids, :2] = torch.zeros_like(self._desired_pos_w[env_ids, :2]).uniform_(-2.0, 2.0)
        # self._desired_pos_w[env_ids, :2] += self._terrain.env_origins[env_ids, :2]
        self._desired_pos_w[env_ids, 2] = torch.zeros_like(self._desired_pos_w[env_ids, 2]).uniform_(0.5, 1.5)
        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        # default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    # def _set_debug_vis_impl(self, debug_vis: bool):
    #     # create markers if necessary for the first tome
    #     if debug_vis:
    #         if not hasattr(self, "goal_pos_visualizer"):
    #             marker_cfg = CUBOID_MARKER_CFG.copy()
    #             marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
    #             # -- goal pose
    #             marker_cfg.prim_path = "/Visuals/Command/goal_position"
    #             self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
    #         # set their visibility to true
    #         self.goal_pos_visualizer.set_visibility(True)
    #     else:
    #         if hasattr(self, "goal_pos_visualizer"):
    #             self.goal_pos_visualizer.set_visibility(False)
    #
    # def _debug_vis_callback(self, event):
    #     # update the markers
    #     self.goal_pos_visualizer.visualize(self._desired_pos_w)

def camera_follow(env):
    if (env.unwrapped.scene.num_envs == 1):
        robot_position = env.unwrapped.scene.articulations["robot"].data.root_state_w[0, :3].cpu().numpy()
        robot_orientation = env.unwrapped.scene.articulations["robot"].data.root_state_w[0, 3:7].cpu().numpy()
        rotation = R.from_quat([robot_orientation[1], robot_orientation[2], 
                                robot_orientation[3], robot_orientation[0]])
        yaw = rotation.as_euler('zyx')[0]
        yaw_rotation = R.from_euler('z', yaw).as_matrix()
        set_camera_view(
            yaw_rotation.dot(np.asarray([-4.0, 0.0, 5.0])) + robot_position,
            robot_position
        )
def debug_draw(draw, paths: torch.Tensor):
    draw.clear_lines()
    draw.clear_points()

    def draw_single_traj(traj, color, size):
        traj[:, 2] = torch.mean(traj[:, 2])
        draw.draw_lines(traj[:-1].tolist(), traj[1:].tolist(), color * len(traj[1:]),
                        size * len(traj[1:]))
    for idx, path in enumerate(paths):
        curr_path = torch.stack(paths[idx], dim=0)
        draw_single_traj(curr_path, [(1.0, 0.4, 0.1, 1.0)], [10])
        # self.draw.draw_points(goal.tolist(), self.color_path * len(goal), self.size * len(goal))
