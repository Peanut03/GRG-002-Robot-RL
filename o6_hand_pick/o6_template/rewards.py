from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils import math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def palm_lift_to_table(
    env: ManagerBasedRLEnv,
    min_height: float,
    target_height: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="rh_hand_base_link"),
) -> torch.Tensor:
    """Reward the palm rising from the low start pose to the tabletop workspace."""

    robot: Articulation = env.scene[robot_cfg.name]
    palm_body_id = robot_cfg.body_ids[0]
    palm_z = robot.data.body_pos_w[:, palm_body_id, 2]
    return torch.clamp((palm_z - min_height) / (target_height - min_height), min=0.0, max=1.0)


def body_to_object(
    env: ManagerBasedRLEnv,
    std: float,
    body_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="rh_thumb_distal"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward a selected robot body moving close to the object."""

    robot: Articulation = env.scene[body_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]

    body_pos_w = robot.data.body_pos_w[:, body_cfg.body_ids[0], :3]
    object_pos_w = obj.data.root_pos_w[:, :3]
    dist = torch.norm(body_pos_w - object_pos_w, dim=-1)
    return 1.0 - torch.tanh(dist / std)


def virtual_grasp_point_to_object(
    env: ManagerBasedRLEnv,
    std: float,
    thumb_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="rh_thumb_distal"),
    middle_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="rh_middle_distal"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward a virtual grasp point moving close to the object center.

    The point is one third of the way from thumb tip to middle fingertip.
    """

    robot: Articulation = env.scene[thumb_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]

    thumb_pos_w = robot.data.body_pos_w[:, thumb_cfg.body_ids[0], :3]
    middle_pos_w = robot.data.body_pos_w[:, middle_cfg.body_ids[0], :3]
    object_pos_w = obj.data.root_pos_w[:, :3]

    point_p_w = thumb_pos_w + (middle_pos_w - thumb_pos_w) / 3.0
    dist = torch.norm(point_p_w - object_pos_w, dim=-1)
    return 1.0 - torch.tanh(dist / std)


def object_lifted_with_blocked_grasp(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    close_joint_pos: Mapping[str, float],
    blocked_joint_names: Sequence[str],
    blocked_error_threshold: float,
    fingertip_near_threshold: float,
    close_threshold: float,
    gripper_action_index: int = -1,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    thumb_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="rh_thumb_distal"),
    finger_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward lifting only when nearby fingers grasp the object and are blocked before full closure."""

    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]

    joint_ids, joint_names = robot.find_joints(blocked_joint_names, preserve_order=True)
    close_targets = torch.tensor(
        [close_joint_pos[joint_name] for joint_name in joint_names],
        device=env.device,
        dtype=robot.data.joint_pos.dtype,
    )
    joint_pos = robot.data.joint_pos[:, joint_ids]
    close_error = torch.abs(close_targets.unsqueeze(0) - joint_pos).mean(dim=-1)

    object_lifted = obj.data.root_pos_w[:, 2] > minimal_height
    commanded_close = env.action_manager.action[:, gripper_action_index] > close_threshold
    fingers_blocked = close_error > blocked_error_threshold
    fingers_near = _thumb_and_finger_near_object(env, fingertip_near_threshold, thumb_cfg, finger_cfg, object_cfg)

    return (object_lifted & commanded_close & fingers_blocked & fingers_near).float()


def gripper_close_when_palm_near(
    env: ManagerBasedRLEnv,
    grasp_point_near_threshold: float,
    close_threshold: float,
    transition_width: float,
    gripper_action_index: int = -1,
    thumb_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="rh_thumb_distal"),
    middle_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="rh_middle_distal"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward increasing the close command once the virtual grasp point is near the object."""

    close_score = _gripper_close_score(env, close_threshold, transition_width, gripper_action_index)
    grasp_point_near = _virtual_grasp_point_near_object(
        env, grasp_point_near_threshold, thumb_cfg, middle_cfg, object_cfg
    )
    return grasp_point_near.float() * close_score


def gripper_close_before_palm_near(
    env: ManagerBasedRLEnv,
    grasp_point_near_threshold: float,
    close_threshold: float,
    transition_width: float,
    gripper_action_index: int = -1,
    thumb_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="rh_thumb_distal"),
    middle_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="rh_middle_distal"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Penalize increasing the close command before the virtual grasp point is near the object."""

    close_score = _gripper_close_score(env, close_threshold, transition_width, gripper_action_index)
    grasp_point_near = _virtual_grasp_point_near_object(
        env, grasp_point_near_threshold, thumb_cfg, middle_cfg, object_cfg
    )
    return (~grasp_point_near).float() * close_score


def good_finger_contact(
    env: ManagerBasedRLEnv,
    min_force: float,
    thumb_sensor_name: str = "thumb_cube_contact",
    finger_sensor_names: Sequence[str] = (
        "index_cube_contact",
        "middle_cube_contact",
        "ring_cube_contact",
        "pinky_cube_contact",
    ),
) -> torch.Tensor:
    """Bool-style contact reward following Dexsuite: thumb and any finger contact the object."""

    thumb_sensor: ContactSensor = env.scene.sensors[thumb_sensor_name]
    finger_sensors = [env.scene.sensors[name] for name in finger_sensor_names]

    thumb_force = torch.norm(thumb_sensor.data.force_matrix_w.view(env.num_envs, -1, 3), dim=-1).amax(dim=-1)
    finger_forces = [
        torch.norm(sensor.data.force_matrix_w.view(env.num_envs, -1, 3), dim=-1).amax(dim=-1)
        for sensor in finger_sensors
    ]

    thumb_contact = thumb_force > min_force
    finger_contact = torch.stack([force > min_force for force in finger_forces], dim=1)
    any_finger_contact = finger_contact.any(dim=1)

    simultaneous_contact = thumb_contact & any_finger_contact
    return simultaneous_contact.float()


def object_lift_progress_with_contact(
    env: ManagerBasedRLEnv,
    start_height: float,
    target_height: float,
    min_force: float,
    close_threshold: float,
    gripper_action_index: int = -1,
    thumb_sensor_name: str = "thumb_cube_contact",
    finger_sensor_names: Sequence[str] = (
        "index_cube_contact",
        "middle_cube_contact",
        "ring_cube_contact",
        "pinky_cube_contact",
    ),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward continuous cube height only after a real, commanded grasp.

    This fills the sparse gap between first contact and the existing binary
    ``lifting_object`` bonus without changing the learned reach/grasp sequence.
    """

    obj: RigidObject = env.scene[object_cfg.name]
    lift_progress = torch.clamp(
        (obj.data.root_pos_w[:, 2] - start_height) / (target_height - start_height),
        min=0.0,
        max=1.0,
    )
    simultaneous_contact = good_finger_contact(
        env,
        min_force=min_force,
        thumb_sensor_name=thumb_sensor_name,
        finger_sensor_names=finger_sensor_names,
    )
    commanded_close = (
        env.action_manager.action[:, gripper_action_index] > close_threshold
    ).float()
    return lift_progress * simultaneous_contact * commanded_close


def object_goal_distance_with_fingers_near(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    fingertip_near_threshold: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    thumb_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="rh_thumb_distal"),
    finger_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward object goal tracking only while fingertips remain near the object."""

    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)

    des_pos_b = command[:, :3]
    des_pos_w, _ = math_utils.combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b)
    distance = torch.norm(des_pos_w - obj.data.root_pos_w, dim=1)

    object_lifted = obj.data.root_pos_w[:, 2] > minimal_height
    fingers_near = _thumb_and_finger_near_object(env, fingertip_near_threshold, thumb_cfg, finger_cfg, object_cfg)

    return (object_lifted & fingers_near).float() * (1.0 - torch.tanh(distance / std))


def _thumb_and_finger_near_object(
    env: ManagerBasedRLEnv,
    threshold: float,
    thumb_cfg: SceneEntityCfg,
    finger_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
) -> torch.Tensor:
    robot: Articulation = env.scene[thumb_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]

    object_pos_w = obj.data.root_pos_w[:, :3]
    thumb_pos_w = robot.data.body_pos_w[:, thumb_cfg.body_ids[0], :3]
    finger_pos_w = robot.data.body_pos_w[:, finger_cfg.body_ids, :3]

    thumb_near = torch.norm(thumb_pos_w - object_pos_w, dim=-1) < threshold
    finger_near = torch.norm(finger_pos_w - object_pos_w.unsqueeze(1), dim=-1).min(dim=-1).values < threshold
    return thumb_near & finger_near


def _body_near_object(
    env: ManagerBasedRLEnv,
    threshold: float,
    body_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
) -> torch.Tensor:
    robot: Articulation = env.scene[body_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]

    body_pos_w = robot.data.body_pos_w[:, body_cfg.body_ids[0], :3]
    object_pos_w = obj.data.root_pos_w[:, :3]
    return torch.norm(body_pos_w - object_pos_w, dim=-1) < threshold


def _virtual_grasp_point_near_object(
    env: ManagerBasedRLEnv,
    threshold: float,
    thumb_cfg: SceneEntityCfg,
    middle_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
) -> torch.Tensor:
    robot: Articulation = env.scene[thumb_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]

    thumb_pos_w = robot.data.body_pos_w[:, thumb_cfg.body_ids[0], :3]
    middle_pos_w = robot.data.body_pos_w[:, middle_cfg.body_ids[0], :3]
    point_p_w = thumb_pos_w + (middle_pos_w - thumb_pos_w) / 2.0
    object_pos_w = obj.data.root_pos_w[:, :3]
    return torch.norm(point_p_w - object_pos_w, dim=-1) < threshold


def _gripper_close_score(
    env: ManagerBasedRLEnv,
    close_threshold: float,
    transition_width: float,
    gripper_action_index: int,
) -> torch.Tensor:
    """Shape closing while strongly favoring commands that really close the gripper."""

    action = env.action_manager.action[:, gripper_action_index]
    # Keep a weak dense signal below the binary threshold so PPO can discover
    # the closing direction.  Cap that pre-close reward at 0.25, then give a
    # clear jump to 0.75 once the action really triggers the binary gripper.
    # This prevents the old failure mode where an open hand could receive a
    # large sigmoid score, without making the reward completely sparse.
    del transition_width  # Retained in the public signature for config compatibility.
    pre_span = max(close_threshold + 1.0, 1.0e-6)
    post_span = max(1.0 - close_threshold, 1.0e-6)
    pre_close = 0.25 * torch.clamp((action + 1.0) / pre_span, min=0.0, max=1.0)
    post_close = 0.75 + 0.25 * torch.clamp((action - close_threshold) / post_span, min=0.0, max=1.0)
    return torch.where(action > close_threshold, post_close, pre_close)


def fingertip_plane_parallel_table(
    env: ManagerBasedRLEnv,
    fingertip_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    palm_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="rh_hand_base_link"),
    palm_axis: Sequence[float] = (1.0, 0.0, 0.0),
) -> torch.Tensor:
    """Reward the fingertip plane being parallel to the tabletop with palm facing down."""

    return _fingertip_plane_parallel_and_palm_down_score(env, fingertip_cfg, palm_cfg, palm_axis)


def _fingertip_plane_parallel_and_palm_down_score(
    env: ManagerBasedRLEnv,
    fingertip_cfg: SceneEntityCfg,
    palm_cfg: SceneEntityCfg,
    palm_axis: Sequence[float],
) -> torch.Tensor:
    robot: Articulation = env.scene[fingertip_cfg.name]

    plane_normal_w = _fingertip_plane_normal(env, fingertip_cfg)
    world_down_w = torch.tensor((0.0, 0.0, -1.0), device=env.device, dtype=torch.float32).view(1, 3)
    world_up_w = torch.tensor((0.0, 0.0, 1.0), device=env.device, dtype=torch.float32).view(1, 3)

    plane_parallel = torch.abs(torch.sum(plane_normal_w * world_up_w, dim=-1))

    palm_axis_b = torch.tensor(palm_axis, device=env.device, dtype=torch.float32).view(1, 3)
    palm_axis_b = palm_axis_b / torch.clamp(torch.norm(palm_axis_b, dim=-1, keepdim=True), min=1.0e-6)
    palm_axis_w = math_utils.quat_apply(robot.data.body_quat_w[:, palm_cfg.body_ids[0], :], palm_axis_b.expand(env.num_envs, -1))
    palm_facing_down = torch.sum(palm_axis_w * world_down_w, dim=-1)

    return torch.clamp(plane_parallel, min=0.0, max=1.0) * torch.clamp(palm_facing_down, min=0.0, max=1.0)


def _fingertip_plane_normal(env: ManagerBasedRLEnv, fingertip_cfg: SceneEntityCfg) -> torch.Tensor:
    robot: Articulation = env.scene[fingertip_cfg.name]

    fingertip_pos_w = robot.data.body_pos_w[:, fingertip_cfg.body_ids, :3]
    centered_pos_w = fingertip_pos_w - fingertip_pos_w.mean(dim=1, keepdim=True)
    covariance = torch.matmul(centered_pos_w.transpose(1, 2), centered_pos_w)

    _, eigenvectors = torch.linalg.eigh(covariance)
    plane_normal_w = eigenvectors[:, :, 0]
    return plane_normal_w / torch.clamp(torch.norm(plane_normal_w, dim=-1, keepdim=True), min=1.0e-6)