from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass


class O6MimicGripperAction(ActionTerm):
    """Map one continuous close command to linked O6 thumb and finger joint targets."""

    def __init__(self, cfg: O6MimicGripperActionCfg, env):
        super().__init__(cfg, env)

        self._joint_ids, self._joint_names = self._asset.find_joints(self.cfg.joint_names, preserve_order=True)
        self._raw_actions = torch.zeros(self.num_envs, 1, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, len(self._joint_ids), device=self.device)

        close_targets = {
            self.cfg.thumb_yaw_joint: math.radians(self.cfg.thumb_yaw_deg),
            self.cfg.thumb_pitch_joint: math.radians(self.cfg.thumb_pitch_close_deg),
            self.cfg.thumb_ip_joint: math.radians(self.cfg.thumb_ip_close_deg),
        }
        for joint_name in self.cfg.finger_pitch_joints:
            close_targets[joint_name] = math.radians(self.cfg.finger_pitch_close_deg)
        for joint_name in self.cfg.finger_dip_joints:
            close_targets[joint_name] = math.radians(self.cfg.finger_dip_close_deg)

        self._open_targets = torch.zeros(self.num_envs, len(self._joint_ids), device=self.device)
        self._close_targets = torch.zeros_like(self._open_targets)
        for index, joint_name in enumerate(self._joint_names):
            self._close_targets[:, index] = close_targets[joint_name]
        self._open_targets[:, self._joint_names.index(self.cfg.thumb_yaw_joint)] = close_targets[self.cfg.thumb_yaw_joint]

    @property
    def action_dim(self) -> int:
        return 1

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions[:, :1]
        close_amount = torch.clamp(self._raw_actions, min=0.0, max=1.0)
        self._processed_actions = self._open_targets + close_amount * (self._close_targets - self._open_targets)

    def apply_actions(self):
        self._asset.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = self._open_targets[env_ids]


class O6BinaryMimicGripperAction(O6MimicGripperAction):
    """Map one scalar command to either fully open or fully closed mimic targets."""

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions[:, :1]
        close_amount = (self._raw_actions > self.cfg.close_threshold).to(dtype=self._processed_actions.dtype)
        self._processed_actions = self._open_targets + close_amount * (self._close_targets - self._open_targets)


@configclass
class O6MimicGripperActionCfg(ActionTermCfg):
    """One-dimensional mimic-gripper action for the O6 right dexterous hand."""

    class_type: type[ActionTerm] = O6MimicGripperAction
    joint_names: list[str] = None
    thumb_yaw_joint: str = "rh_thumb_cmc_yaw"
    thumb_pitch_joint: str = "rh_thumb_cmc_pitch"
    thumb_ip_joint: str = "rh_thumb_ip"
    finger_pitch_joints: list[str] = None
    finger_dip_joints: list[str] = None
    thumb_yaw_deg: float = 70.0
    thumb_pitch_close_deg: float = 17.0
    thumb_ip_close_deg: float = 60.0
    finger_pitch_close_deg: float = 90.0
    finger_dip_close_deg: float = 80.0


@configclass
class O6BinaryMimicGripperActionCfg(O6MimicGripperActionCfg):
    """One-dimensional binary gripper action for the O6 right dexterous hand."""

    class_type: type[ActionTerm] = O6BinaryMimicGripperAction
    close_threshold: float = 0.5