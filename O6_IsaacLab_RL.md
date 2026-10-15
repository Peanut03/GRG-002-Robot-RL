# O6 人形机器人抓取与抬升任务使用说明

> IsaacLab 根目录：`/home/space2/songwei/rl_ws/IsaacLab`

## 1. 项目做什么

当前强化学习任务使用 IsaacLab 的 Manager-Based 环境和 RSL-RL PPO，使 O6 人形机器人的右臂与五指手完成以下行为：

1. 抬高手掌进入桌面工作空间。
2. 让手掌和虚拟抓取点靠近物体。
3. 保持手掌朝下并对准物体。
4. 在合适距离闭合夹爪。
5. 让拇指与至少一根手指接触物体。
6. 抬起物体并靠近目标位置。

当前同时注册了 Cube 和 Cylinder 两类任务。

## 2. 目录结构与含义

```text
/home/space2/songwei/rl_ws/IsaacLab
├── assets/
│   └── o6_robot/
│       └── o6_robot_standalone.usd        # O6 机器人 USD 资产
├── source/isaaclab_tasks/isaaclab_tasks/
│   └── manager_based/manipulation/lift/
│       ├── lift_env_cfg.py                 # IsaacLab 通用 Lift 任务基础配置
│       └── config/o6_template/
│           ├── __init__.py                 # Gym 任务名称注册
│           ├── joint_pos_env_cfg.py        # O6 场景、奖励项、阈值、传感器、动作配置
│           ├── rewards.py                  # 奖励函数的具体数学实现
│           ├── actions.py                  # O6 二值联动夹爪动作实现
│           ├── events.py                   # 环境重置和夹爪初始化逻辑
│           ├── object_assets.py            # Cube/Cylinder 尺寸、姿态、摩擦等资产配置
│           └── agents/
│               └── rsl_rl_ppo_cfg.py       # PPO 网络、学习率、噪声、迭代数等
├── scripts/reinforcement_learning/rsl_rl/
│   ├── train.py                            # IsaacLab 通用 RSL-RL 训练入口
│   ├── play.py                             # IsaacLab 通用 RSL-RL 推理入口
│   ├── train_o6_scaled.py                  # O6 2.6 倍物理场景训练入口
│   └── collect_o6_*.py                     # 数据集采集脚本
└── logs/rsl_rl/
    └── <experiment_name>/
        └── <日期_时间_可选run_name>/
            ├── model_*.pt                  # 训练权重
            ├── events.out.tfevents.*       # TensorBoard 日志
            ├── params/env.yaml             # 本次训练最终展开的环境配置快照
            └── params/agent.yaml           # 本次训练最终展开的 PPO 配置快照
```

### 2.1 最重要的配置文件

#### `joint_pos_env_cfg.py`

完整路径：

```text
/home/space2/songwei/rl_ws/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/config/o6_template/joint_pos_env_cfg.py
```

这里主要配置：

- 机器人 USD 路径和初始姿态。
- 右臂7个关节和夹爪11个关节。
- 关节执行器的力矩、速度、刚度和阻尼。
- 夹爪开合关节角和二值闭合阈值。
- 目标位置采样范围。
- 各奖励项的权重与参数。
- Cube/Cylinder 的任务差异。
- 物体接触传感器。
- 物体重置范围。

#### `rewards.py`

完整路径：

```text
/home/space2/songwei/rl_ws/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/config/o6_template/rewards.py
```

这里实现奖励的数学计算，例如：

- 手掌或虚拟抓取点到物体的距离。
- 夹爪闭合引导。
- 拇指和其他手指是否同时接触物体。
- 物体是否被抬起。
- 物体到目标位置的距离。
- 手掌高度和朝向。

当前回滚版本中的 `_gripper_close_score()` 是两段式闭合得分：闭合动作未越过阈值时保留弱引导，越过真实闭合阈值后给予更高得分。

#### `actions.py`

二值夹爪的真实动作在这里执行。策略动作共8维：

```text
动作 0～6：右臂7个关节目标
动作 7：  右手夹爪开合命令
```

夹爪动作大于 `0.65` 时执行闭合，否则执行张开。

#### `object_assets.py`

这里可以切换和调整抓取物体：

```python
self.scene.object = make_lift_object_cfg("cube")
```

或：

```python
self.scene.object = make_lift_object_cfg("cylinder")
```

当前资产参数：

| 物体 | 参数 | 当前值 |
|---|---|---:|
| Cube | USD scale | `(2.7, 0.7, 0.8)` |
| Cylinder | radius | `0.03 m` |
| Cylinder | height | `0.220 m` |
| Cylinder | axis | `X`，因此侧面接触桌面 |
| 两者 | static friction | `3.0` |
| 两者 | dynamic friction | `3.0` |

注意：使用 `--task_scale 2.6` 后，资产自身 scale 还会再乘2.6。例如 Cube 最终 spawn scale 是 `(7.02, 1.82, 2.08)`。

#### `agents/rsl_rl_ppo_cfg.py`

当前源码中的 PPO 主要参数：

| 参数 | 当前值 | 含义 |
|---|---:|---|
| `num_steps_per_env` | 32 | 每次更新前，每个环境采样的步数 |
| `max_iterations` | 15000 | 默认 PPO 迭代数 |
| `save_interval` | 250 | 每250代保存一次权重 |
| `init_noise_std` | 1.2 | 初始策略动作噪声 |
| `learning_rate` | `2e-4` | PPO 学习率 |
| `entropy_coef` | `0.006` | 探索熵系数 |
| `gamma` | 0.99 | 折扣因子 |
| `lam` | 0.95 | GAE 参数 |
| Actor/Critic | `[512,256,128]` | 隐藏层尺寸 |
| activation | ELU | 激活函数 |

## 3. 已注册的任务名称

| Task ID | 用途 |
|---|---|
| `Isaac-Lift-Cube-O6-Template-v0` | Cube 训练任务 |
| `Isaac-Lift-Cube-O6-Template-Play-v0` | Cube Play 配置 |
| `Isaac-Lift-Cylinder-O6-Template-v0` | Cylinder 训练任务 |
| `Isaac-Lift-Cylinder-O6-Template-Play-v0` | Cylinder Play 配置 |

Cube 训练和推理必须使用相同的 Cube Task ID。Cylinder 权重也必须搭配 Cylinder Task ID。

## 4. 观测和动作

当前策略观测维度为102：

| 观测项 | 维度 | 含义 |
|---|---:|---|
| `joint_pos` | 42 | 机器人关节位置 |
| `joint_vel` | 42 | 机器人关节速度 |
| `object_position` | 3 | 物体相对位置 |
| `target_object_position` | 7 | 目标位置与姿态 |
| `actions` | 8 | 上一步策略动作 |

动作维度为8：右臂7维，加夹爪1维。

## 5. 当前奖励项

当前源码包含13项奖励：

| 奖励名 | 权重 | 含义 |
|---|---:|---|
| `reaching_object` | `2.0` | 虚拟抓取点靠近物体 |
| `lifting_object` | `15.0` | 夹爪真实抓持并抬起物体 |
| `object_goal_tracking` | `16.0` | 物体抬起后靠近目标位置 |
| `object_goal_tracking_fine_grained` | `5.0` | 更精细的目标位置奖励 |
| `action_rate` | `-1e-4` | 惩罚动作变化过快 |
| `joint_vel` | `-1e-4` | 惩罚关节速度过大 |
| `palm_to_object_coarse` | `0.1` | 手掌粗略靠近物体 |
| `gripper_close_when_palm_near` | `2.0` | 手接近后鼓励闭合 |
| `gripper_close_before_palm_near` | `-0.5` | 手未接近时惩罚提前闭合 |
| `good_finger_contact` | `12.0` | 拇指与任一其他手指同时接触 |
| `object_lift_progress` | `8.0` | 有效接触并闭合后按抬升高度连续奖励 |
| `palm_lift_to_table` | `0.1` | 鼓励手从低处抬到桌面工作空间 |
| `palm_facing_down` | `0.5` | 鼓励抓取平面平行桌面且掌心向下 |

训练后期 curriculum 会把 `action_rate` 和 `joint_vel` 的目标惩罚权重逐渐调整到 `-6e-4`。

## 6. 场景启动参数

当前使用的物理任务变换：

```bash
--task_translate -0.8 2.212 -2.008
--task_orient_euler 0 0 -1.1
--task_scale 2.6
--ground_translate 0 0 -4.66
```

### 参数解释

| 参数 | 单位 | 含义 |
|---|---|---|
| `--task_translate X Y Z` | 米 | Robot、Table、Object 整体世界坐标平移 |
| `--task_orient_euler RX RY RZ` | 度 | Robot、Table、Object 的 XYZ 欧拉角旋转 |
| `--task_scale S` | 倍数 | Robot、Table、Object 的统一物理缩放 |
| `--ground_translate X Y Z` | 米 | GroundPlane 独立世界坐标位置 |
| `--num_envs N` | 个 | 并行仿真环境数量 |
| `--device cuda:1` | — | 使用第2块 GPU；编号从0开始 |
| `--max_iterations N` | 代 | 本次训练的 PPO 迭代预算 |
| `--experiment_name NAME` | — | `logs/rsl_rl/` 下的实验目录名 |
| `--run_name NAME` | — | 时间戳目录后附加的说明文字 |
| `--headless` | — | 不显示 Isaac Sim 窗口 |
| `--video` | — | 训练过程中录制视频，同时启用摄像头 |

`train_o6_scaled.py` 会在 PhysX 创建场景前处理缩放，而不是只缩放画面。它会同步处理：

- Robot、Table、Object 的几何尺寸和初始位置。
- 机器人执行器的力矩、刚度和阻尼。
- 奖励中的米制距离阈值。
- 奖励中的绝对高度。
- 物体目标范围和重置范围。
- 环境间距和 PhysX 摩擦相关距离。
- 位置观测的数值尺度。

它不会改变奖励权重。

### 2.6倍后的关键奖励阈值

按照当前脚本的换算方式：

| 原参数 | 原值 | 2.6倍场景中的值 |
|---|---:|---:|
| reaching `std` | 0.25 | 0.65 |
| coarse `std` | 0.70 | 1.82 |
| goal tracking `std` | 0.30 | 0.78 |
| fine tracking `std` | 0.05 | 0.13 |
| fingertip near | 0.125 | 0.325 |
| grasp-point near | 0.13 | 0.338 |
| lifting minimal height | 0.04 | -1.904 |
| palm minimum height | -0.45 | -3.178 |
| palm target height | 0.16 | -1.592 |
| Cube lift-progress start | 0.021 | -1.9534 |
| Cube lift-progress target | 0.15 | -1.618 |

绝对高度使用以下公式：

```text
新高度 = task_translate_z + task_scale × 原高度
```

## 7. 每次启动前的环境初始化

在服务器终端执行：

```bash
cd /home/space2/songwei/rl_ws/IsaacLab

source /home/songwei/miniconda3/etc/profile.d/conda.sh
conda activate /home/space2/songwei/conda_envs/env_isaaclab
TERM=xterm source _isaac_sim/setup_conda_env.sh
export DISPLAY=:10
```

如果使用 `--headless`，通常不依赖可见窗口，但保留 `DISPLAY=:10` 也没有问题。

## 8. 从头启动训练

### 8.1 有画面的 Cube 训练

```bash
python scripts/reinforcement_learning/rsl_rl/train_o6_scaled.py \
  --task Isaac-Lift-Cube-O6-Template-v0 \
  --device cuda:1 \
  --num_envs 1024 \
  --max_iterations 15000 \
  --task_translate -0.8 2.212 -2.008 \
  --task_orient_euler 0 0 -1.1 \
  --task_scale 2.6 \
  --ground_translate 0 0 -4.66 \
  --experiment_name o6_lift_cube
```

不要添加 `--headless`，Isaac Sim 会显示画面。1024个环境同时渲染可能降低训练速度；只为了观察时可以减少为64或128个环境。

### 8.2 无画面的高速 Cube 训练

```bash
python scripts/reinforcement_learning/rsl_rl/train_o6_scaled.py \
  --task Isaac-Lift-Cube-O6-Template-v0 \
  --headless \
  --device cuda:1 \
  --num_envs 4096 \
  --max_iterations 15000 \
  --task_translate -0.8 2.212 -2.008 \
  --task_orient_euler 0 0 -1.1 \
  --task_scale 2.6 \
  --ground_translate 0 0 -4.66 \
  --experiment_name o6_lift_cube
```

如果显存不足，依次把 `4096` 降为 `2048`、`1024` 或 `512`。

### 8.3 Cylinder 训练

将 Task ID 换成 Cylinder：

```bash
python scripts/reinforcement_learning/rsl_rl/train_o6_scaled.py \
  --task Isaac-Lift-Cylinder-O6-Template-v0 \
  --device cuda:1 \
  --num_envs 4096 \
  --max_iterations 15000 \
  --experiment_name o6_lift_cylinder
```

### 8.4 gripper 训练

训练gripper抓取cube

```bash
cd /home/space2/songwei/rl_ws/IsaacLab
conda activate env_isaaclab

export DISPLAY=:10.0
export XAUTHORITY=/home/songwei/.Xauthority

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Lift-Cube-G200-Gripper-O6-Copy-Play-v0 \
  --num_envs 4096 \
  --device cuda:0 \
  --max_iterations 15000
```

## 9. 断点续训

假设要从以下权重继续：

```text
logs/rsl_rl/o6_lift_cube/
└── 2026-07-20_10-39-22/
    └── model_2000.pt
```

命令为：

```bash
python scripts/reinforcement_learning/rsl_rl/train_o6_scaled.py \
  --task Isaac-Lift-Cube-O6-Template-v0 \
  --device cuda:1 \
  --num_envs 4096 \
  --task_translate -0.8 2.212 -2.008 \
  --task_orient_euler 0 0 -1.1 \
  --task_scale 2.6 \
  --ground_translate 0 0 -4.66 \
  --experiment_name o6_lift_cube \
  --resume \
  --load_run 2026-07-20_10-39-22 \
  --checkpoint model_2000.pt
```

续训时必须保持以下内容与原训练一致：

- Task ID。
- 观测维度和动作维度。
- Cube/Cylinder 类型。
- `task_translate`、`task_orient_euler`、`task_scale` 和地面位置。
- 网络隐藏层尺寸与归一化设置。

## 10. 启动强化学习推理

### 10.1 当前已有的完整2.6倍权重

目前服务器上存在：

```text
/home/space2/songwei/rl_ws/IsaacLab/logs/rsl_rl/o6_lift_cube/2026-08-10_16-48-06/model_14000.pt
```

### 10.2 直接加载权重运行推理

先执行第7节的环境初始化，然后运行：

cube

变换后

```bash
python scripts/reinforcement_learning/rsl_rl/play_o6_scaled.py \
  --task Isaac-Lift-Cube-O6-Template-v0 \
  --device cuda:1 \
  --num_envs 1 \
  --checkpoint /home/space2/songwei/rl_ws/IsaacLab/logs/rsl_rl/o6_lift_cube/2026-08-10_16-48-06/model_14000.pt \
  --task_scale 2.6 \
  --task_translate -0.8 2.212 -2.008 \
  --task_orient_euler 0 0 -1.1 \
  --ground_translate 0 0 -4.66 \
  --real-time
```

未变换

```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Lift-Cube-O6-Template-v0 \
  --device cuda:1 \
  --num_envs 1 \
  --checkpoint /home/space2/songwei/rl_ws/IsaacLab/logs/rsl_rl/o6_lift_cube_fresh_tuned/2026-07-14_16-53-59/model_14000.pt \
  --real-time
```

Cylinder

未变换

```bash
python scripts/reinforcement_learning/rsl_rl/play.py   \
	--task Isaac-Lift-Cylinder-O6-Template-v0  \
  --device cuda:1 \
  --num_envs 1  \
  --checkpoint  /home/space2/songwei/rl_ws/IsaacLab/logs/rsl_rl/o6_lift_cylinder_fresh_tuned/2026-07-15_17-06-00/model_10000.pt \
  --real-time
```



gripper

```bash
cd /home/space2/songwei/rl_ws/IsaacLab
conda activate env_isaaclab

export DISPLAY=:10.0
export XAUTHORITY=/home/songwei/.Xauthority

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Lift-Cube-G200-Gripper-O6-Copy-Play-v0 \
  --num_envs 1 \
  --device cuda:0 \
  --real-time \
  --checkpoint /home/space2/songwei/rl_ws/IsaacLab/logs/rsl_rl/g200_gripper_o6_copy/2026-08-07_13-42-57_resume_from_4000/model_7000.pt
```



关键点：

- `--checkpoint` 必须是 `.pt` 文件的绝对路径。
- 该命令只加载任务本身的机器人、桌子和抓取物体，不加载任何外部场景。
- `--num_envs 1` 只创建一个环境，便于观察机器人动作。
- `--real-time` 让推理速度尽量接近真实时间；需要最快运行时可以删掉。

## 11. 查看训练曲线

在服务器另开一个终端：

```bash
cd /home/space2/songwei/rl_ws/IsaacLab
source /home/songwei/miniconda3/etc/profile.d/conda.sh
conda activate /home/space2/songwei/conda_envs/env_isaaclab

tensorboard \
  --logdir /home/space2/songwei/rl_ws/IsaacLab/logs/rsl_rl \
  --port 6006 \
  --bind_all
```

重点观察：

| 指标 | 预期现象 |
|---|---|
| `Episode_Reward/reaching_object` | 手靠近物体后上升 |
| `Episode_Reward/gripper_close_when_palm_near` | 进入抓取范围并增加闭合动作后上升 |
| `Episode_Reward/good_finger_contact` | 拇指和其他手指真实接触后出现 |
| `Episode_Reward/object_lift_progress` | 抓稳并向上抬物体后持续上升 |
| `Episode_Reward/lifting_object` | 达到最小抬升高度并保持抓取后出现 |
| `Episode_Reward/object_goal_tracking` | 物体抬起并靠近目标后上升 |
| `Metrics/object_pose/position_error` | 越低越好 |
| `Episode_Termination/object_dropping` | 越低越好 |

## 12. 配置快照和源码的区别

每次训练都会在运行目录保存：

```text
params/env.yaml
params/agent.yaml
```

它们记录的是“该次训练启动时最终生效的配置”，包括命令行缩放后的数值。它们非常适合排查某个 checkpoint 当时到底使用了什么参数。

但是，续训和推理脚本仍然从当前 Python 源码创建环境，并不会自动把 `params/env.yaml` 重新加载回源码。因此：

- 修改源码后，旧 checkpoint 的网络文件不会自动记录全部环境行为。
- 推理通常不使用奖励，但必须保持观测、动作、场景几何和物体类型兼容。
- 继续训练旧 checkpoint 前，应对比该运行目录的 `params/env.yaml`、`params/agent.yaml` 与当前源码。

当前 `model_14999.pt` 所属实验名称带有 `fresh_bundle`，而服务器源码后来发生过回滚。推理时网络结构和动作维度仍可兼容，但如果要严格复现实验或继续训练，应优先以该运行目录中的两个 params 文件为依据。

## 13. 常见问题

### 启动后没有画面

检查：

```bash
export DISPLAY=:10
echo $DISPLAY
```

并确保训练命令没有 `--headless`。

### CUDA 显存不足

降低：

```bash
--num_envs 4096
```

到 `2048`、`1024`、`512` 或更低。
