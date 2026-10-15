# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""O6 template version of the official Franka cube lift environment."""

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg

from . import actions as o6_actions
from . import events as o6_events
from . import rewards as o6_rewards
from .object_assets import O6_OBJECT_DYNAMIC_FRICTION, O6_OBJECT_STATIC_FRICTION, make_lift_object_cfg
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip


O6_ROBOT_USD = "/home/space2/songwei/rl_ws/IsaacLab/assets/o6_robot/o6_robot_standalone.usd"

O6_ARM_JOINTS = [
    "R_shoulder1_joint",
    "R_shoulder2_joint",
    "R_shoulder3_joint",
    "R_shoulder4_joint",
    "R_shoulder5_joint",
    "R_shoulder6_joint",
    "R_shoulder7_joint",
]
O6_EE_BODY_NAME = "rh_hand_base_link"
O6_BASE_BODY_NAME = "base_link"
O6_ARTICULATION_ROOT_PRIM_PATH = "/root_joint"

O6_GRIPPER_JOINTS = [
    "rh_index_mcp_pitch",
    "rh_index_dip",
    "rh_middle_mcp_pitch",
    "rh_middle_dip",
    "rh_pinky_mcp_pitch",
    "rh_pinky_dip",
    "rh_ring_mcp_pitch",
    "rh_ring_dip",
    "rh_thumb_cmc_yaw",
    "rh_thumb_cmc_pitch",
    "rh_thumb_ip",
]
O6_FINGER_PITCH_JOINTS = [
    "rh_index_mcp_pitch",
    "rh_middle_mcp_pitch",
    "rh_pinky_mcp_pitch",
    "rh_ring_mcp_pitch",
]
O6_FINGER_DIP_JOINTS = [
    "rh_index_dip",
    "rh_middle_dip",
    "rh_pinky_dip",
    "rh_ring_dip",
]
O6_GRASP_BLOCKED_JOINTS = O6_FINGER_PITCH_JOINTS + O6_FINGER_DIP_JOINTS + [
    "rh_thumb_cmc_pitch",
    "rh_thumb_ip",
]
O6_THUMB_TIP_BODY = "rh_thumb_distal"
O6_MIDDLE_TIP_BODY = "rh_middle_distal"
O6_OTHER_FINGER_TIP_BODIES = [
    "rh_index_distal",
    "rh_middle_distal",
    "rh_ring_distal",
    "rh_pinky_distal",
]
O6_GRASP_PLANE_TIP_BODIES = [O6_THUMB_TIP_BODY] + O6_OTHER_FINGER_TIP_BODIES
O6_FINGERTIP_NEAR_THRESHOLD = 0.125
O6_PALM_NEAR_THRESHOLD = 0.25
O6_GRASP_POINT_NEAR_THRESHOLD = 0.13
O6_PALM_FACING_AXIS = (1.0, 0.0, 0.0)
O6_BINARY_GRIPPER_CLOSE_THRESHOLD = 0.65
O6_GRIPPER_CLOSE_REWARD_TRANSITION_WIDTH = 0.85
O6_GRIPPER_OPEN_JOINT_POS = {
    "rh_index_mcp_pitch": 0.0,
    "rh_index_dip": 0.0,
    "rh_middle_mcp_pitch": 0.0,
    "rh_middle_dip": 0.0,
    "rh_pinky_mcp_pitch": 0.0,
    "rh_pinky_dip": 0.0,
    "rh_ring_mcp_pitch": 0.0,
    "rh_ring_dip": 0.0,
    "rh_thumb_cmc_yaw": 1.22173,
    "rh_thumb_cmc_pitch": 0.0,
    "rh_thumb_ip": 0.0,
}
O6_GRIPPER_CLOSE_JOINT_POS = {
    "rh_index_mcp_pitch": 1.57080,
    "rh_index_dip": 1.39626,
    "rh_middle_mcp_pitch": 1.57080,
    "rh_middle_dip": 1.39626,
    "rh_pinky_mcp_pitch": 1.57080,
    "rh_pinky_dip": 1.39626,
    "rh_ring_mcp_pitch": 1.57080,
    "rh_ring_dip": 1.39626,
    "rh_thumb_cmc_yaw": 1.22173,
    "rh_thumb_cmc_pitch": 0.29671,
    "rh_thumb_ip": 1.04720,
}

O6_ROBOT_GROUND_POS = (0.0, -0.75, -1.05)
O6_ROBOT_GROUND_ROT = (0.70710678, 0.0, 0.0, 0.70710678)


@configclass
class O6CubeLiftEnvCfg(LiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=UsdFileCfg(
                usd_path=O6_ROBOT_USD,
                activate_contact_sensors=True,
                rigid_props=RigidBodyPropertiesCfg(disable_gravity=False, max_depenetration_velocity=5.0),
            ),
            init_state=ArticulationCfg.InitialStateCfg(pos=O6_ROBOT_GROUND_POS, rot=O6_ROBOT_GROUND_ROT),
            articulation_root_prim_path=O6_ARTICULATION_ROOT_PRIM_PATH,
            actuators={
                "o6_arm": ImplicitActuatorCfg(
                    joint_names_expr=O6_ARM_JOINTS,
                    effort_limit_sim=140.0,
                    velocity_limit_sim=20.0,
                    stiffness=80.0,
                    damping=4.0,
                ),
                "o6_gripper": ImplicitActuatorCfg(
                    joint_names_expr=O6_GRIPPER_JOINTS,
                    effort_limit_sim=80.0,
                    velocity_limit_sim=20.0,
                    stiffness=80.0,
                    damping=2.0,
                ),
            },
        )

        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=O6_ARM_JOINTS,
            scale=0.5,
            use_default_offset=True,
        )
        self.actions.gripper_action = o6_actions.O6BinaryMimicGripperActionCfg(
            asset_name="robot",
            joint_names=O6_GRIPPER_JOINTS,
            thumb_yaw_joint="rh_thumb_cmc_yaw",
            thumb_pitch_joint="rh_thumb_cmc_pitch",
            thumb_ip_joint="rh_thumb_ip",
            finger_pitch_joints=O6_FINGER_PITCH_JOINTS,
            finger_dip_joints=O6_FINGER_DIP_JOINTS,
            close_threshold=O6_BINARY_GRIPPER_CLOSE_THRESHOLD,
        )

        self.commands.object_pose.body_name = O6_EE_BODY_NAME
        self.commands.object_pose.ranges.pos_x = (0.65, 0.85)
        self.commands.object_pose.ranges.pos_y = (-0.65, -0.35)
        self.commands.object_pose.ranges.pos_z = (1.20, 1.45)

        self.rewards.palm_to_object_coarse = RewTerm(
            func=o6_rewards.body_to_object,
            weight=0.1,
            params={
                "std": 0.7,
                "body_cfg": SceneEntityCfg("robot", body_names=O6_EE_BODY_NAME),
            },
        )
        self.rewards.reaching_object = RewTerm(
            func=o6_rewards.virtual_grasp_point_to_object,
            weight=2.0,
            params={
                "std": 0.25,
                "thumb_cfg": SceneEntityCfg("robot", body_names=O6_THUMB_TIP_BODY),
                "middle_cfg": SceneEntityCfg("robot", body_names=O6_MIDDLE_TIP_BODY),
            },
        )
        self.rewards.lifting_object = RewTerm(
            func=o6_rewards.object_lifted_with_blocked_grasp,
            weight=15.0,
            params={
                "minimal_height": 0.04,
                "close_joint_pos": O6_GRIPPER_CLOSE_JOINT_POS,
                "blocked_joint_names": O6_GRASP_BLOCKED_JOINTS,
                "blocked_error_threshold": 0.08,
                "fingertip_near_threshold": O6_FINGERTIP_NEAR_THRESHOLD,
                "close_threshold": O6_BINARY_GRIPPER_CLOSE_THRESHOLD,
                "gripper_action_index": -1,
                "robot_cfg": SceneEntityCfg("robot"),
                "thumb_cfg": SceneEntityCfg("robot", body_names=O6_THUMB_TIP_BODY),
                "finger_cfg": SceneEntityCfg("robot", body_names=O6_OTHER_FINGER_TIP_BODIES),
            },
        )
        self.rewards.object_goal_tracking = RewTerm(
            func=o6_rewards.object_goal_distance_with_fingers_near,
            weight=16.0,
            params={
                "std": 0.3,
                "minimal_height": 0.04,
                "command_name": "object_pose",
                "fingertip_near_threshold": O6_FINGERTIP_NEAR_THRESHOLD,
                "robot_cfg": SceneEntityCfg("robot"),
                "thumb_cfg": SceneEntityCfg("robot", body_names=O6_THUMB_TIP_BODY),
                "finger_cfg": SceneEntityCfg("robot", body_names=O6_OTHER_FINGER_TIP_BODIES),
            },
        )
        self.rewards.object_goal_tracking_fine_grained = RewTerm(
            func=o6_rewards.object_goal_distance_with_fingers_near,
            weight=5.0,
            params={
                "std": 0.05,
                "minimal_height": 0.04,
                "command_name": "object_pose",
                "fingertip_near_threshold": O6_FINGERTIP_NEAR_THRESHOLD,
                "robot_cfg": SceneEntityCfg("robot"),
                "thumb_cfg": SceneEntityCfg("robot", body_names=O6_THUMB_TIP_BODY),
                "finger_cfg": SceneEntityCfg("robot", body_names=O6_OTHER_FINGER_TIP_BODIES),
            },
        )
        self.rewards.action_rate.weight = -1.0e-4
        self.rewards.joint_vel.weight = -1.0e-4
        self.rewards.joint_vel.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=O6_ARM_JOINTS + O6_GRIPPER_JOINTS
        )
        self.rewards.gripper_close_when_palm_near = RewTerm(
            func=o6_rewards.gripper_close_when_palm_near,
            weight=2.0,
            params={
                "grasp_point_near_threshold": O6_GRASP_POINT_NEAR_THRESHOLD,
                "close_threshold": O6_BINARY_GRIPPER_CLOSE_THRESHOLD,
                "transition_width": O6_GRIPPER_CLOSE_REWARD_TRANSITION_WIDTH,
                "gripper_action_index": -1,
                "thumb_cfg": SceneEntityCfg("robot", body_names=O6_THUMB_TIP_BODY),
                "middle_cfg": SceneEntityCfg("robot", body_names=O6_MIDDLE_TIP_BODY),
            },
        )
        self.rewards.gripper_close_before_palm_near = RewTerm(
            func=o6_rewards.gripper_close_before_palm_near,
            weight=-0.5,
            params={
                "grasp_point_near_threshold": O6_GRASP_POINT_NEAR_THRESHOLD,
                "close_threshold": O6_BINARY_GRIPPER_CLOSE_THRESHOLD,
                "transition_width": O6_GRIPPER_CLOSE_REWARD_TRANSITION_WIDTH,
                "gripper_action_index": -1,
                "thumb_cfg": SceneEntityCfg("robot", body_names=O6_THUMB_TIP_BODY),
                "middle_cfg": SceneEntityCfg("robot", body_names=O6_MIDDLE_TIP_BODY),
            },
        )
        self.rewards.good_finger_contact = RewTerm(
            func=o6_rewards.good_finger_contact,
            weight=12.0,
            params={
                "min_force": 0.05,
                "thumb_sensor_name": "thumb_cube_contact",
                "finger_sensor_names": (
                    "index_cube_contact",
                    "middle_cube_contact",
                    "ring_cube_contact",
                    "pinky_cube_contact",
                ),
            },
        )
        self.rewards.object_lift_progress = RewTerm(
            func=o6_rewards.object_lift_progress_with_contact,
            weight=8.0,
            params={
                # The scaled cube settles with its center at z ~= 0.021 m.
                "start_height": 0.021,
                "target_height": 0.15,
                "min_force": 0.05,
                "close_threshold": O6_BINARY_GRIPPER_CLOSE_THRESHOLD,
                "gripper_action_index": -1,
                "thumb_sensor_name": "thumb_cube_contact",
                "finger_sensor_names": (
                    "index_cube_contact",
                    "middle_cube_contact",
                    "ring_cube_contact",
                    "pinky_cube_contact",
                ),
            },
        )
        self.curriculum.action_rate.params["weight"] = -6.0e-4
        self.curriculum.joint_vel.params["weight"] = -6.0e-4

        self.rewards.palm_lift_to_table = RewTerm(
            func=o6_rewards.palm_lift_to_table,
            weight=0.1,
            params={
                "min_height": -0.45,
                "target_height": 0.16,
                "robot_cfg": SceneEntityCfg("robot", body_names=O6_EE_BODY_NAME),
            },
        )
        self.rewards.palm_facing_down = RewTerm(
            func=o6_rewards.fingertip_plane_parallel_table,
            weight=0.65,
            params={
                "fingertip_cfg": SceneEntityCfg("robot", body_names=O6_GRASP_PLANE_TIP_BODIES),
                "palm_cfg": SceneEntityCfg("robot", body_names=O6_EE_BODY_NAME),
                "palm_axis": O6_PALM_FACING_AXIS,
            },
        )

        # Apply the same high-friction material to either selectable object.
        # This startup override also works for the referenced DexCube USD.
        self.events.object_physics_material = EventTerm(
            func=mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("object"),
                "static_friction_range": (
                    O6_OBJECT_STATIC_FRICTION,
                    O6_OBJECT_STATIC_FRICTION,
                ),
                "dynamic_friction_range": (
                    O6_OBJECT_DYNAMIC_FRICTION,
                    O6_OBJECT_DYNAMIC_FRICTION,
                ),
                "restitution_range": (0.0, 0.0),
                "num_buckets": 1,
                "make_consistent": True,
            },
        )

        self.events.reset_object_position.params["pose_range"] = {
            "x": (-0.08, 0.08),
            "y": (-0.28, -0.17),
            "z": (0.0, 0.0),
        }
        self.events.reset_gripper_random_aperture = EventTerm(
            func=o6_events.reset_gripper_open_or_closed,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=O6_GRIPPER_JOINTS),
                "open_joint_pos": O6_GRIPPER_OPEN_JOINT_POS,
                "close_joint_pos": O6_GRIPPER_OPEN_JOINT_POS,
            },
        )

        self.scene.thumb_cube_contact = ContactSensorCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Robot/{O6_THUMB_TIP_BODY}",
            update_period=0.0,
            history_length=16,
            track_air_time=True,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        )
        self.scene.index_cube_contact = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/rh_index_distal",
            update_period=0.0,
            history_length=16,
            track_air_time=True,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        )
        self.scene.middle_cube_contact = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/rh_middle_distal",
            update_period=0.0,
            history_length=16,
            track_air_time=True,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        )
        self.scene.ring_cube_contact = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/rh_ring_distal",
            update_period=0.0,
            history_length=16,
            track_air_time=True,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        )
        self.scene.pinky_cube_contact = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/rh_pinky_distal",
            update_period=0.0,
            history_length=16,
            track_air_time=True,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        )

        self.scene.object = make_lift_object_cfg("cube")

        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Robot/{O6_BASE_BODY_NAME}",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/Robot/{O6_EE_BODY_NAME}",
                    name="end_effector",
                    offset=OffsetCfg(pos=[0.0, 0.0, 0.0]),
                ),
            ],
        )


@configclass
class O6CubeLiftEnvCfg_PLAY(O6CubeLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class O6CylinderLiftEnvCfg(O6CubeLiftEnvCfg):
    """O6 lift task using a vertical cylinder instead of the DexCube."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.object = make_lift_object_cfg("cylinder")

        # The horizontal cylinder rests on its curved side with its center
        # around one radius above the table (z ~= 0.035 m). Require roughly
        # 2 cm of real lift before enabling binary/goal rewards.
        self.rewards.lifting_object.params["minimal_height"] = 0.055
        self.rewards.object_goal_tracking.params["minimal_height"] = 0.055
        self.rewards.object_goal_tracking_fine_grained.params["minimal_height"] = 0.055
        self.rewards.object_lift_progress.params["start_height"] = 0.036
        self.rewards.object_lift_progress.params["target_height"] = 0.16


@configclass
class O6CylinderLiftEnvCfg_PLAY(O6CylinderLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False