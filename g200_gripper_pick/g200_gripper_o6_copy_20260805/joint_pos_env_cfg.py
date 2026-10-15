"""Independent G200 parallel-gripper copy of the O6 cube-lift task."""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.converters.urdf_converter_cfg import UrdfConverterCfg
from isaaclab.sim.schemas.schemas_cfg import (
    ArticulationRootPropertiesCfg,
    RigidBodyPropertiesCfg,
)
from isaaclab.sim.spawners.from_files.from_files_cfg import UrdfFileCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg

from . import actions as gripper_actions
from . import events as gripper_events
from . import rewards as gripper_rewards


# The robot is spawned from the exact URDF requested by the user. Isaac Lab
# converts it to a cached USD at runtime; the source URDF is not modified.
G200_ROBOT_URDF = (
    "/home/space2/resources/g200_robot.SLDASM/urdf/"
    "g200_robot.SLDASM_0722_1.urdf"
)
G200_CONVERTED_USD_DIR = (
    "/home/space2/songwei/rl_ws/IsaacLab/assets/"
    "g200_gripper_o6_copy_20260805"
)

G200_RIGHT_ARM_JOINTS = [f"right_arm_joint{i}" for i in range(1, 8)]
G200_RIGHT_ELBOW_JOINT = "right_arm_joint4"
G200_RIGHT_ELBOW_LIMIT = (-2.62, 0.0)
G200_RIGHT_GRIPPER_JOINTS = [
    "right_hand_narrow1_joint",
    "right_hand_narrow2_joint",
    "right_hand_narrow3_joint",
    "right_hand_narrow_loop_joint",
    "right_hand_wide1_joint",
    "right_hand_wide2_joint",
    "right_hand_wide3_joint",
    "right_hand_wide_loop_joint",
]
G200_WAIST_AND_BELOW_JOINTS = [
    "lift_down_joint",
    "lift_middle_joint",
    "lift_top_joint",
    "waist_joint",
]
G200_LOCKED_JOINTS = [
    *G200_WAIST_AND_BELOW_JOINTS,
    "head_down_joint",
    "head_top_joint",
    *[f"left_arm_joint{i}" for i in range(1, 8)],
    "hand_narrow1_joint",
    "hand_narrow2_joint",
    "hand_narrow3_joint",
    "hand_narrow_loop_joint",
    "hand_wide1_joint",
    "hand_wide2_joint",
    "hand_wide3_joint",
    "hand_wide_loop_joint",
]

G200_EE_BODY_NAME = "right_hand_link"
G200_BASE_BODY_NAME = "base_link"
G200_LEFT_JAW_TIP = "right_hand_narrow3_Link"
G200_RIGHT_JAW_TIP = "right_hand_wide3_Link"

G200_GRASP_POINT_NEAR_THRESHOLD = 0.1
G200_CLOSE_THRESHOLD = 0.45
G200_CLOSE_TRANSITION_WIDTH = 0.65
G200_MIN_CONTACT_FORCE = 0.05

G200_GRIPPER_OPEN_JOINT_POS = {
    "right_hand_narrow1_joint": -1.0,
    "right_hand_narrow2_joint": 1.0,
    "right_hand_narrow3_joint": 0.0,
    "right_hand_narrow_loop_joint": -1.0,
    "right_hand_wide1_joint": 1.0,
    "right_hand_wide2_joint": -1.0,
    "right_hand_wide3_joint": 0.0,
    "right_hand_wide_loop_joint": 1.0,
}
G200_GRIPPER_CLOSE_JOINT_POS = {
    "right_hand_narrow1_joint": 0.55,
    "right_hand_narrow2_joint": -0.55,
    "right_hand_narrow3_joint": 0.0,
    "right_hand_narrow_loop_joint": 0.55,
    "right_hand_wide1_joint": -0.55,
    "right_hand_wide2_joint": 0.55,
    "right_hand_wide3_joint": 0.0,
    "right_hand_wide_loop_joint": -0.55,
}

# Use the true mechanically open pose from the very first simulation frame.
# Explicit names avoid a catch-all zero rule overriding the gripper targets.
G200_INITIAL_JOINT_POS = {
    name: 0.0 for name in G200_RIGHT_ARM_JOINTS + G200_LOCKED_JOINTS
}
G200_INITIAL_JOINT_POS.update(G200_GRIPPER_OPEN_JOINT_POS)

# Center the robot on the midpoint of the table's long edge (x=0.5).
G200_ROBOT_GROUND_POS = (0.5, -0.75, -1.05)
G200_ROBOT_GROUND_ROT = (0.70710678, 0.0, 0.0, 0.70710678)

G200_TABLE_BLOCK_SIZE = (1.2, 1.0, 1.05)
G200_TABLE_BLOCK_POS = (0.5, 0.0, -0.525)


@configclass
class G200GripperO6CopyCubeLiftEnvCfg(LiftEnvCfg):
    """O6 scene and PPO strategy with the right hand replaced by G200 jaws."""

    def __post_init__(self):
        super().__post_init__()

        # Requested training scale. This remains overrideable with
        # ``--num_envs`` on the Isaac Lab command line.
        self.scene.num_envs = 4096

        # Replace only this copied task's Seattle lab table with a solid static
        # block.  The top face remains at z=0 and the bottom reaches the ground
        # plane at z=-1.05, preserving all existing cube poses and rewards.
        self.scene.table = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Table",
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=G200_TABLE_BLOCK_POS,
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
            spawn=sim_utils.CuboidCfg(
                size=G200_TABLE_BLOCK_SIZE,
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.32, 0.34, 0.38)
                ),
            ),
        )

        self.scene.robot = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=UrdfFileCfg(
                asset_path=G200_ROBOT_URDF,
                usd_dir=G200_CONVERTED_USD_DIR,
                usd_file_name="g200_gripper_o6_copy.usd",
                fix_base=True,
                merge_fixed_joints=False,
                make_instanceable=True,
                collider_type="convex_hull",
                self_collision=False,
                joint_drive=UrdfConverterCfg.JointDriveCfg(
                    target_type="none",
                    gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                        stiffness=0.0, damping=0.0
                    ),
                ),
                activate_contact_sensors=True,
                rigid_props=RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    max_linear_velocity=4.0,
                    max_angular_velocity=30.0,
                    max_depenetration_velocity=2.0,
                ),
                articulation_props=ArticulationRootPropertiesCfg(
                    fix_root_link=True,
                    enabled_self_collisions=False,
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=G200_ROBOT_GROUND_POS,
                rot=G200_ROBOT_GROUND_ROT,
                joint_pos=G200_INITIAL_JOINT_POS,
            ),
            articulation_root_prim_path="/base_link",
            actuators={
                "right_arm": ImplicitActuatorCfg(
                    joint_names_expr=G200_RIGHT_ARM_JOINTS,
                    effort_limit_sim=120.0,
                    velocity_limit_sim=12.0,
                    stiffness=80.0,
                    damping=6.0,
                ),
                "right_gripper": ImplicitActuatorCfg(
                    joint_names_expr=G200_RIGHT_GRIPPER_JOINTS,
                    effort_limit_sim=70.0,
                    velocity_limit_sim=8.0,
                    stiffness=80.0,
                    damping=3.0,
                ),
                "locked_body_and_left_side": ImplicitActuatorCfg(
                    joint_names_expr=G200_LOCKED_JOINTS,
                    effort_limit_sim=140.0,
                    velocity_limit_sim=10.0,
                    stiffness=200.0,
                    damping=20.0,
                ),
            },
        )

        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=G200_RIGHT_ARM_JOINTS,
            scale=0.5,
            use_default_offset=True,
            # Clip the processed position target as well as the PhysX joint
            # below, so the policy never commands elbow hyperextension.
            clip={G200_RIGHT_ELBOW_JOINT: G200_RIGHT_ELBOW_LIMIT},
        )
        self.actions.gripper_action = gripper_actions.PositiveBinaryGripperActionCfg(
            asset_name="robot",
            joint_names=G200_RIGHT_GRIPPER_JOINTS,
            open_joint_pos=G200_GRIPPER_OPEN_JOINT_POS,
            close_joint_pos=G200_GRIPPER_CLOSE_JOINT_POS,
            close_threshold=G200_CLOSE_THRESHOLD,
        )

        movable_joints = G200_RIGHT_ARM_JOINTS + G200_RIGHT_GRIPPER_JOINTS
        self.observations.policy.joint_pos.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=movable_joints
        )
        self.observations.policy.joint_vel.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=movable_joints
        )

        self.commands.object_pose.body_name = G200_EE_BODY_NAME
        self.commands.object_pose.ranges.pos_x = (0.47, 0.58)
        self.commands.object_pose.ranges.pos_y = (-0.58, -0.42)
        self.commands.object_pose.ranges.pos_z = (1.30, 1.40)

        gripper_cfg = SceneEntityCfg("robot", body_names=G200_EE_BODY_NAME)
        left_jaw_cfg = SceneEntityCfg("robot", body_names=G200_LEFT_JAW_TIP)
        right_jaw_cfg = SceneEntityCfg("robot", body_names=G200_RIGHT_JAW_TIP)

        # Same twelve-term strategy as the desktop O6 bundle, with all
        # dexterous-hand geometry/contact gates replaced by two-jaw gates.
        self.rewards.gripper_to_object_coarse = RewTerm(
            func=gripper_rewards.gripper_center_to_object,
            weight=0.1,
            params={
                "std": 0.7,
                "left_jaw_cfg": left_jaw_cfg,
                "right_jaw_cfg": right_jaw_cfg,
            },
        )
        self.rewards.reaching_object = RewTerm(
            func=gripper_rewards.gripper_center_to_object,
            weight=2.0,
            params={
                "std": 0.25,
                "left_jaw_cfg": left_jaw_cfg,
                "right_jaw_cfg": right_jaw_cfg,
            },
        )
        contact_params = {
            "min_force": G200_MIN_CONTACT_FORCE,
            "left_sensor_name": "left_jaw_cube_contact",
            "right_sensor_name": "right_jaw_cube_contact",
        }
        self.rewards.lifting_object = RewTerm(
            func=gripper_rewards.object_lifted_with_two_jaw_grasp,
            weight=15.0,
            params={
                "minimal_relative_height": 0.04,
                "close_threshold": G200_CLOSE_THRESHOLD,
                "gripper_action_index": -1,
                **contact_params,
            },
        )
        goal_params = {
            "minimal_relative_height": 0.04,
            "command_name": "object_pose",
            "robot_cfg": SceneEntityCfg("robot"),
            **contact_params,
        }
        self.rewards.object_goal_tracking = RewTerm(
            func=gripper_rewards.object_goal_distance_with_two_jaw_grasp,
            weight=16.0,
            params={"std": 0.3, **goal_params},
        )
        self.rewards.object_goal_tracking_fine_grained = RewTerm(
            func=gripper_rewards.object_goal_distance_with_two_jaw_grasp,
            weight=5.0,
            params={"std": 0.05, **goal_params},
        )
        self.rewards.action_rate.weight = -1.0e-4
        self.rewards.joint_vel.weight = -1.0e-4
        self.rewards.joint_vel.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=movable_joints
        )
        close_params = {
            "grasp_point_near_threshold": G200_GRASP_POINT_NEAR_THRESHOLD,
            "close_threshold": G200_CLOSE_THRESHOLD,
            "transition_width": G200_CLOSE_TRANSITION_WIDTH,
            "gripper_action_index": -1,
            "left_jaw_cfg": left_jaw_cfg,
            "right_jaw_cfg": right_jaw_cfg,
        }
        self.rewards.gripper_close_when_near = RewTerm(
            func=gripper_rewards.gripper_close_when_near,
            weight=2.0,
            params=close_params,
        )
        self.rewards.gripper_close_before_near = RewTerm(
            func=gripper_rewards.gripper_close_before_near,
            weight=-1.0,
            params={**close_params},
        )
        self.rewards.two_jaw_contact = RewTerm(
            func=gripper_rewards.two_jaw_contact,
            weight=15.0,
            params=contact_params,
        )
        self.rewards.gripper_lift_to_table = RewTerm(
            func=gripper_rewards.gripper_lift_to_table,
            weight=0.3,
            params={
                "min_relative_height": -0.505,
                "target_relative_height": -0.02,
                "left_jaw_cfg": left_jaw_cfg,
                "right_jaw_cfg": right_jaw_cfg,
            },
        )
        self.rewards.gripper_alignment = RewTerm(
            func=gripper_rewards.gripper_pregrasp_alignment,
            weight=0.8,
            params={
                "gripper_cfg": gripper_cfg,
                "left_jaw_cfg": left_jaw_cfg,
                "right_jaw_cfg": right_jaw_cfg,
            },
        )
        self.curriculum.action_rate.params["weight"] = -1.0e-3
        self.curriculum.joint_vel.params["weight"] = -1.0e-3

        self.events.reset_object_position.params["pose_range"] = {
            "x": (-0.08, 0.08),
            "y": (-0.28, -0.17),
            "z": (0.0, 0.0),
        }
        self.events.set_right_elbow_joint_limit = EventTerm(
            func=gripper_events.set_joint_position_limits,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=[G200_RIGHT_ELBOW_JOINT]
                ),
                "lower": G200_RIGHT_ELBOW_LIMIT[0],
                "upper": G200_RIGHT_ELBOW_LIMIT[1],
            },
        )
        self.events.lock_waist_and_lower_body_joints = EventTerm(
            func=gripper_events.set_joint_position_limits,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=G200_WAIST_AND_BELOW_JOINTS
                ),
                "lower": 0.0,
                "upper": 0.0,
            },
        )
        self.events.reset_gripper_random_aperture = EventTerm(
            func=gripper_events.reset_gripper_open_or_closed,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=G200_RIGHT_GRIPPER_JOINTS
                ),
                "open_joint_pos": G200_GRIPPER_OPEN_JOINT_POS,
                "close_joint_pos": G200_GRIPPER_OPEN_JOINT_POS,
            },
        )

        self.scene.left_jaw_cube_contact = ContactSensorCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Robot/{G200_LEFT_JAW_TIP}",
            update_period=0.0,
            history_length=16,
            track_air_time=True,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        )
        self.scene.right_jaw_cube_contact = ContactSensorCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Robot/{G200_RIGHT_JAW_TIP}",
            update_period=0.0,
            history_length=16,
            track_air_time=True,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        )

        # Keep the original O6 scene's cube, size, pose, and physics settings.
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[0.5, 0.0, 0.055], rot=[1.0, 0.0, 0.0, 0.0]
            ),
            spawn=UsdFileCfg(
                usd_path=(
                    f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/"
                    "dex_cube_instanceable.usd"
                ),
                scale=(0.7, 0.7, 0.7),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )

        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Robot/{G200_BASE_BODY_NAME}",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/Robot/{G200_EE_BODY_NAME}",
                    name="end_effector",
                    offset=OffsetCfg(pos=[0.05, 0.0, 0.0]),
                )
            ],
        )


@configclass
class G200GripperO6CopyCubeLiftEnvCfg_PLAY(G200GripperO6CopyCubeLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Reproduce the scene and reward configuration recorded in
        # logs/rsl_rl/g200_gripper_o6_copy/2026-08-05_17-07-51/params/env.yaml.
        # These overrides are playback-only; the current training task remains
        # on the solid table with the newer alignment and joint protections.
        self.scene.robot.init_state.pos = (0.0, -0.75, -1.05)
        self.scene.table = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Table",
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(0.5, 0.0, 0.0),
                rot=(0.707, 0.0, 0.0, 0.707),
            ),
            spawn=UsdFileCfg(
                usd_path=(
                    f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/"
                    "table_instanceable.usd"
                )
            ),
        )

        # Keep the archived elbow action behavior, but retain the requested
        # hard lock for the waist and all three lift joints to prevent shaking.
        self.actions.arm_action.clip = None
        self.events.set_right_elbow_joint_limit = None

        # Both object-goal rewards consume this same command. Raising the
        # command also keeps the policy observation and target marker aligned
        # with the reward target.
        self.commands.object_pose.ranges.pos_z = (1.40, 1.50)

        # Reward geometry from the archived
        # ``joint_pos_env_cfg.py.bak_before_midpoint_rewards_20260805``
        # strategy: the coarse term measures from the gripper base, while the
        # fine reaching term inherited above measures from the jaw midpoint.
        gripper_cfg = SceneEntityCfg("robot", body_names=G200_EE_BODY_NAME)
        self.rewards.gripper_to_object_coarse = RewTerm(
            func=gripper_rewards.gripper_base_to_object,
            weight=0.1,
            params={"std": 0.7, "gripper_cfg": gripper_cfg},
        )

        left_jaw_cfg = SceneEntityCfg("robot", body_names=G200_LEFT_JAW_TIP)
        right_jaw_cfg = SceneEntityCfg("robot", body_names=G200_RIGHT_JAW_TIP)
        self.rewards.gripper_alignment = RewTerm(
            func=gripper_rewards.gripper_pregrasp_alignment_20260805,
            weight=0.1,
            params={
                "left_jaw_cfg": left_jaw_cfg,
                "right_jaw_cfg": right_jaw_cfg,
            },
        )

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
