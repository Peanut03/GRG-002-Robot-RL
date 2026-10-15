from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reset_gripper_open_or_closed(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    open_joint_pos: Mapping[str, float],
    close_joint_pos: Mapping[str, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Randomly reset the mimic hand to a continuous aperture between open and closed."""

    robot: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    joint_ids, joint_names = robot.find_joints(asset_cfg.joint_names)
    open_targets = torch.tensor(
        [open_joint_pos[joint_name] for joint_name in joint_names],
        device=env.device,
        dtype=robot.data.joint_pos.dtype,
    )
    close_targets = torch.tensor(
        [close_joint_pos[joint_name] for joint_name in joint_names],
        device=env.device,
        dtype=robot.data.joint_pos.dtype,
    )

    close_amount = torch.rand((len(env_ids), 1), device=env.device, dtype=robot.data.joint_pos.dtype)
    joint_pos = open_targets.unsqueeze(0) + close_amount * (close_targets.unsqueeze(0) - open_targets.unsqueeze(0))
    joint_vel = torch.zeros_like(joint_pos)
    robot.write_joint_state_to_sim(joint_pos, joint_vel, joint_ids=joint_ids, env_ids=env_ids)
    robot.set_joint_position_target(joint_pos, joint_ids=joint_ids, env_ids=env_ids)
