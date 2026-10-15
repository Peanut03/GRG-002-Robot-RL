"""One-dimensional positive-close action for the G200 parallel gripper."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass


class PositiveBinaryGripperAction(ActionTerm):
    """Map one scalar action to the G200 gripper's synchronized joint targets.

    This deliberately keeps the desktop O6 policy convention: an action above
    ``close_threshold`` closes the gripper; lower values open it.
    """

    cfg: PositiveBinaryGripperActionCfg

    def __init__(self, cfg: PositiveBinaryGripperActionCfg, env):
        super().__init__(cfg, env)
        self._joint_ids, self._joint_names = self._asset.find_joints(
            cfg.joint_names, preserve_order=True
        )
        self._raw_actions = torch.zeros(self.num_envs, 1, device=self.device)
        self._processed_actions = torch.zeros(
            self.num_envs, len(self._joint_ids), device=self.device
        )
        self._open_targets = self._targets_from_mapping(cfg.open_joint_pos)
        self._close_targets = self._targets_from_mapping(cfg.close_joint_pos)
        self._processed_actions[:] = self._open_targets

    def _targets_from_mapping(self, values: Mapping[str, float]) -> torch.Tensor:
        row = torch.tensor(
            [values[name] for name in self._joint_names],
            device=self.device,
            dtype=torch.float32,
        )
        return row.unsqueeze(0).expand(self.num_envs, -1).clone()

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
        requested_close = self._raw_actions > self.cfg.close_threshold
        self._processed_actions = torch.where(
            requested_close, self._close_targets, self._open_targets
        )

    def apply_actions(self):
        self._asset.set_joint_position_target(
            self._processed_actions, joint_ids=self._joint_ids
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = self._open_targets[env_ids]


@configclass
class PositiveBinaryGripperActionCfg(ActionTermCfg):
    """Configuration for :class:`PositiveBinaryGripperAction`."""

    class_type: type[ActionTerm] = PositiveBinaryGripperAction
    joint_names: list[str] = None
    open_joint_pos: dict[str, float] = None
    close_joint_pos: dict[str, float] = None
    close_threshold: float = 0.65
