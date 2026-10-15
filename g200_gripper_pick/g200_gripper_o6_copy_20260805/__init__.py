"""Independent G200 gripper task copied from o6_fresh_train_bundle_20260710."""

import gymnasium as gym

from . import agents


gym.register(
    id="Isaac-Lift-Cube-G200-Gripper-O6-Copy-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.joint_pos_env_cfg:G200GripperO6CopyCubeLiftEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:G200GripperO6CopyPPORunnerCfg"
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Cube-G200-Gripper-O6-Copy-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.joint_pos_env_cfg:G200GripperO6CopyCubeLiftEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:G200GripperO6CopyPPORunnerCfg"
        ),
    },
    disable_env_checker=True,
)
