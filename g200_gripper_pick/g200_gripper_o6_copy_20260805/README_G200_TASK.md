# G200 parallel-gripper cube-lift task

This is an independent copy of the desktop
`o6_fresh_train_bundle_20260710/o6_template` task. It does not import or
modify that task.

## Robot

- Source URDF: `/home/space2/resources/g200_robot.SLDASM/urdf/g200_robot.SLDASM_0722_1.urdf`
- Controlled arm: `right_arm_joint1` through `right_arm_joint7`
- Controlled gripper: eight right-hand linkage joints driven by one scalar
- Action convention: values greater than `0.65` close; all other values open
- Base, torso, head, left arm, and left gripper are held at zero

## Registered tasks

- Training: `Isaac-Lift-Cube-G200-Gripper-O6-Copy-v0`
- Playback: `Isaac-Lift-Cube-G200-Gripper-O6-Copy-Play-v0`

## Reward strategy

| Reward | Weight | G200 adaptation |
|---|---:|---|
| `gripper_to_object_coarse` | 0.1 | Right gripper base approaches cube |
| `reaching_object` | 2.0 | Midpoint of the two jaw tips approaches cube |
| `lifting_object` | 15.0 | Cube rises 4 cm while both jaws contact and close is commanded |
| `object_goal_tracking` | 16.0 | Coarse goal tracking while lifted and held by both jaws |
| `object_goal_tracking_fine_grained` | 5.0 | Fine goal tracking while lifted and held by both jaws |
| `action_rate` | -0.0001 | Original action-rate penalty |
| `joint_vel` | -0.0001 | Right arm and gripper joint-velocity penalty |
| `gripper_close_when_near` | 2.0 | Rewards closing only when jaw midpoint is near cube |
| `gripper_close_before_near` | -0.5 | Penalizes premature closing |
| `two_jaw_contact` | 15.0 | Requires simultaneous filtered contact on both jaw tips |
| `gripper_lift_to_table` | 0.1 | Raises the lower jaw tip into the tabletop workspace |
| `gripper_alignment` | 0.8 | Keeps jaw span horizontal and cube below jaw midpoint |

The PPO network, optimizer, rollout length, noise, and 15,000-iteration limit
are unchanged from the desktop O6 bundle. Only `experiment_name` is changed to
keep logs separate.

## Commands

```bash
cd /home/space2/songwei/rl_ws/IsaacLab
conda activate env_isaaclab
export DISPLAY=:10.0
export XAUTHORITY=/home/songwei/.Xauthority
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Lift-Cube-G200-Gripper-O6-Copy-v0 \
  --num_envs 4096
```

The training configuration also defaults to 4096 environments. Do not add
`--headless` when a visible Isaac Sim window is required.

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Lift-Cube-G200-Gripper-O6-Copy-Play-v0 \
  --checkpoint /absolute/path/to/model.pt
```
