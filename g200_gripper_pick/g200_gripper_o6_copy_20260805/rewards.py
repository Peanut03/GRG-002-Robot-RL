"""Two-jaw reward adaptations of ``o6_fresh_train_bundle_20260710``."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils import math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _jaw_positions(
    env: ManagerBasedRLEnv,
    left_jaw_cfg: SceneEntityCfg,
    right_jaw_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor]:
    robot: Articulation = env.scene[left_jaw_cfg.name]
    return (
        robot.data.body_pos_w[:, left_jaw_cfg.body_ids[0], :3],
        robot.data.body_pos_w[:, right_jaw_cfg.body_ids[0], :3],
    )


def _jaw_center(
    env: ManagerBasedRLEnv,
    left_jaw_cfg: SceneEntityCfg,
    right_jaw_cfg: SceneEntityCfg,
) -> torch.Tensor:
    left_pos, right_pos = _jaw_positions(env, left_jaw_cfg, right_jaw_cfg)
    return 0.5 * (left_pos + right_pos)


def gripper_base_to_object(
    env: ManagerBasedRLEnv,
    std: float,
    gripper_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Coarse approach reward from the gripper base to the cube."""

    robot: Articulation = env.scene[gripper_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    base_pos = robot.data.body_pos_w[:, gripper_cfg.body_ids[0], :3]
    return 1.0 - torch.tanh(
        torch.norm(base_pos - obj.data.root_pos_w[:, :3], dim=-1) / std
    )


def gripper_center_to_object(
    env: ManagerBasedRLEnv,
    std: float,
    left_jaw_cfg: SceneEntityCfg,
    right_jaw_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Fine approach reward from the midpoint of the opposing jaw tips."""

    obj: RigidObject = env.scene[object_cfg.name]
    center = _jaw_center(env, left_jaw_cfg, right_jaw_cfg)
    return 1.0 - torch.tanh(
        torch.norm(center - obj.data.root_pos_w[:, :3], dim=-1) / std
    )


def _two_jaw_contact_mask(
    env: ManagerBasedRLEnv,
    min_force: float,
    left_sensor_name: str,
    right_sensor_name: str,
) -> torch.Tensor:
    left_sensor: ContactSensor = env.scene.sensors[left_sensor_name]
    right_sensor: ContactSensor = env.scene.sensors[right_sensor_name]
    left_force = torch.norm(
        left_sensor.data.force_matrix_w.view(env.num_envs, -1, 3), dim=-1
    ).amax(dim=-1)
    right_force = torch.norm(
        right_sensor.data.force_matrix_w.view(env.num_envs, -1, 3), dim=-1
    ).amax(dim=-1)
    return (left_force > min_force) & (right_force > min_force)


def two_jaw_contact(
    env: ManagerBasedRLEnv,
    min_force: float,
    left_sensor_name: str,
    right_sensor_name: str,
) -> torch.Tensor:
    """Reward simultaneous physical contact by both opposing jaws."""

    return _two_jaw_contact_mask(
        env, min_force, left_sensor_name, right_sensor_name
    ).float()


def object_lifted_with_two_jaw_grasp(
    env: ManagerBasedRLEnv,
    minimal_relative_height: float,
    min_force: float,
    close_threshold: float,
    gripper_action_index: int,
    left_sensor_name: str,
    right_sensor_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward a real lift only when both jaws contact and close is commanded."""

    obj: RigidObject = env.scene[object_cfg.name]
    lift_height = obj.data.root_pos_w[:, 2] - obj.data.default_root_state[:, 2]
    lifted = lift_height > minimal_relative_height
    contacted = _two_jaw_contact_mask(
        env, min_force, left_sensor_name, right_sensor_name
    )
    commanded_close = (
        env.action_manager.action[:, gripper_action_index] > close_threshold
    )
    return (lifted & contacted & commanded_close).float()


def object_goal_distance_with_two_jaw_grasp(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_relative_height: float,
    command_name: str,
    min_force: float,
    left_sensor_name: str,
    right_sensor_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Track the goal only while the lifted cube remains between both jaws."""

    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    desired_pos_w, _ = math_utils.combine_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, command[:, :3]
    )
    distance = torch.norm(desired_pos_w - obj.data.root_pos_w[:, :3], dim=-1)
    lifted = (
        obj.data.root_pos_w[:, 2] - obj.data.default_root_state[:, 2]
        > minimal_relative_height
    )
    contacted = _two_jaw_contact_mask(
        env, min_force, left_sensor_name, right_sensor_name
    )
    return (lifted & contacted).float() * (1.0 - torch.tanh(distance / std))


def _close_score(
    env: ManagerBasedRLEnv,
    close_threshold: float,
    transition_width: float,
    gripper_action_index: int,
) -> torch.Tensor:
    action = env.action_manager.action[:, gripper_action_index]
    return torch.sigmoid((action - close_threshold) / transition_width)


def _jaw_center_near_object(
    env: ManagerBasedRLEnv,
    threshold: float,
    left_jaw_cfg: SceneEntityCfg,
    right_jaw_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
) -> torch.Tensor:
    obj: RigidObject = env.scene[object_cfg.name]
    center = _jaw_center(env, left_jaw_cfg, right_jaw_cfg)
    return torch.norm(center - obj.data.root_pos_w[:, :3], dim=-1) < threshold


def gripper_close_when_near(
    env: ManagerBasedRLEnv,
    grasp_point_near_threshold: float,
    close_threshold: float,
    transition_width: float,
    gripper_action_index: int,
    left_jaw_cfg: SceneEntityCfg,
    right_jaw_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward closing only after the jaw center reaches the cube."""

    near = _jaw_center_near_object(
        env, grasp_point_near_threshold, left_jaw_cfg, right_jaw_cfg, object_cfg
    )
    return near.float() * _close_score(
        env, close_threshold, transition_width, gripper_action_index
    )


def gripper_close_before_near(
    env: ManagerBasedRLEnv,
    grasp_point_near_threshold: float,
    close_threshold: float,
    transition_width: float,
    gripper_action_index: int,
    left_jaw_cfg: SceneEntityCfg,
    right_jaw_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Penalize closing before the jaw center reaches the cube."""

    near = _jaw_center_near_object(
        env, grasp_point_near_threshold, left_jaw_cfg, right_jaw_cfg, object_cfg
    )
    return (~near).float() * _close_score(
        env, close_threshold, transition_width, gripper_action_index
    )


def gripper_lift_to_table(
    env: ManagerBasedRLEnv,
    min_relative_height: float,
    target_relative_height: float,
    left_jaw_cfg: SceneEntityCfg,
    right_jaw_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward raising the lower jaw tip into the tabletop workspace."""

    obj: RigidObject = env.scene[object_cfg.name]
    left_pos, right_pos = _jaw_positions(env, left_jaw_cfg, right_jaw_cfg)
    relative_height = torch.minimum(left_pos[:, 2], right_pos[:, 2]) - obj.data.root_pos_w[:, 2]
    return torch.clamp(
        (relative_height - min_relative_height)
        / (target_relative_height - min_relative_height),
        min=0.0,
        max=1.0,
    )


def gripper_pregrasp_alignment(
    env: ManagerBasedRLEnv,
    gripper_cfg: SceneEntityCfg,
    left_jaw_cfg: SceneEntityCfg,
    right_jaw_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward the base-to-jaw-midpoint vector pointing left along the table."""

    robot: Articulation = env.scene[gripper_cfg.name]
    gripper_base = robot.data.body_pos_w[:, gripper_cfg.body_ids[0], :3]
    left_pos, right_pos = _jaw_positions(env, left_jaw_cfg, right_jaw_cfg)
    jaw_center = 0.5 * (left_pos + right_pos)

    base_to_center = jaw_center - gripper_base
    base_to_center = base_to_center / torch.clamp(
        torch.norm(base_to_center, dim=-1, keepdim=True), min=1.0e-6
    )

    # World -X is the left direction along the table's long edge.  This dot
    # product with (-1, 0, 0) combines both requirements in one continuous
    # score: 1.0 means exactly left and table-parallel, while vertical or
    # right-pointing vectors receive zero.
    return torch.clamp(-base_to_center[:, 0], min=0.0, max=1.0)


def gripper_pregrasp_alignment_20260805(
    env: ManagerBasedRLEnv,
    left_jaw_cfg: SceneEntityCfg,
    right_jaw_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Legacy alignment used by the 2026-08-05_17-07-51 checkpoint."""

    obj: RigidObject = env.scene[object_cfg.name]
    left_pos, right_pos = _jaw_positions(env, left_jaw_cfg, right_jaw_cfg)
    jaw_line = right_pos - left_pos
    jaw_line = jaw_line / torch.clamp(
        torch.norm(jaw_line, dim=-1, keepdim=True), min=1.0e-6
    )
    jaw_span_horizontal = 1.0 - torch.abs(jaw_line[:, 2])

    center = 0.5 * (left_pos + right_pos)
    center_to_object = obj.data.root_pos_w[:, :3] - center
    center_to_object = center_to_object / torch.clamp(
        torch.norm(center_to_object, dim=-1, keepdim=True), min=1.0e-6
    )
    world_down = torch.tensor(
        (0.0, 0.0, -1.0), device=env.device, dtype=center.dtype
    ).view(1, 3)
    object_below = torch.clamp(
        torch.sum(center_to_object * world_down, dim=-1), min=0.0, max=1.0
    )
    return torch.clamp(jaw_span_horizontal, min=0.0, max=1.0) * object_below
