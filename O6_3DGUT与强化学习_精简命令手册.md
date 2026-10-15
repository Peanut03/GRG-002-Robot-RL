# O6 3DGUT 与强化学习精简命令手册

> 适用服务器：`songwei@10.1.47.59`  
> 整理日期：2026-07-22  
> 本文只保留目录结构、启动命令和关键参数解释。

---

## 1. 目录结构

### 1.1 IsaacLab

```text
/home/space2/songwei/rl_ws/IsaacLab
├── assets/o6_robot/
│   └── o6_robot_standalone.usd              # O6 机器人资产入口
├── source/isaaclab_tasks/isaaclab_tasks/
│   └── manager_based/manipulation/lift/
│       └── config/o6_template/
│           ├── __init__.py                   # 注册任务名称
│           ├── joint_pos_env_cfg.py          # 机器人、物体、奖励和传感器配置
│           ├── rewards.py                    # 奖励函数
│           ├── actions.py                    # 右臂和灵巧手动作
│           ├── events.py                     # 环境重置逻辑
│           ├── object_assets.py              # Cube/Cylinder 资产配置
│           └── agents/rsl_rl_ppo_cfg.py      # PPO 参数
├── scripts/reinforcement_learning/rsl_rl/
│   ├── train_o6_scaled.py                    # 缩放后的物理任务训练
│   ├── play_o6_3dgut_scaled.py               # 机器人和 3DGUT 对齐/推理
│   └── collect_o6_dual_camera_dataset.py     # 双相机数据集采集
├── logs/rsl_rl/                              # 强化学习日志和权重
└── datasets/                                 # 可选的数据集目录
```

### 1.2 3DGRUT 与当前 3DGUT 场景

```text
/home/songwei/3dgrut                         # 3DGRUT 代码

/home/space2/songwei/run_d6184_3dgut_pipeline.sh
                                              # 视频到 3DGUT 完整流水线

/home/space2/songwei/lab_nurec_d6184_3dgut
├── nerfstudio_data/                          # 抽帧和 COLMAP 结果
├── colmap_input/                             # 原始 COLMAP 格式数据
├── colmap_undistorted/                       # 去畸变训练数据
├── output/                                   # 3DGUT 训练结果
├── logs/pipeline.log                         # 流水线日志
└── transparent_50/
    ├── ckpt_density50.pt                     # 50% density checkpoint
    ├── export_nurec_density50.usdz           # NuRec 高斯场景
    └── 3dgut_density50.usda                  # Isaac Sim 加载入口
```

### 1.3 当前强化学习权重

```text
/home/space2/songwei/rl_ws/IsaacLab/logs/rsl_rl/
/o6_lift_cube/2026-08-10_16-48-06/
├── model_14999.pt                            # GPU 原始权重
├── model_14999_cpu.pt                        # CPU 存储、可映射到任意 GPU
└── params/
    ├── env.yaml                              # 训练环境快照
    └── agent.yaml                            # PPO 配置快照
```

---

## 2. 环境初始化命令

### 2.1 登录服务器

```bash
ssh songwei@10.1.47.59
```

### 2.2 IsaacLab 有窗口模式

```bash
cd /home/space2/songwei/rl_ws/IsaacLab

source /home/songwei/miniconda3/etc/profile.d/conda.sh
conda activate /home/space2/songwei/conda_envs/env_isaaclab
TERM=xterm source _isaac_sim/setup_conda_env.sh
export DISPLAY=:10
```

### 2.3 IsaacLab 无窗口模式

```bash
cd /home/space2/songwei/rl_ws/IsaacLab

source /home/songwei/miniconda3/etc/profile.d/conda.sh
conda activate /home/space2/songwei/conda_envs/env_isaaclab
TERM=xterm source _isaac_sim/setup_conda_env.sh
unset DISPLAY
```

### 2.4 3DGRUT 环境

```bash
source /home/songwei/miniconda3/etc/profile.d/conda.sh
conda activate /home/space2/songwei/conda_envs/3dgrut

export PATH="$CONDA_PREFIX/lib/python3.11/site-packages/slangtorch/bin:$CONDA_PREFIX/bin:$PATH"
export CUDA_HOME="$CONDA_PREFIX"
export CUDA_PATH="$CONDA_PREFIX"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=1

cd /home/songwei/3dgrut
```

---

## 3. 视频训练成 3DGUT

### 3.1 推荐：复制并运行完整流水线

复制已有流水线：

```bash
cp /home/space2/songwei/run_d6184_3dgut_pipeline.sh \
   /home/space2/songwei/run_scene01_3dgut_pipeline.sh
```

在新脚本开头修改：

```bash
VIDEO=/home/space2/d6184fe57137be3539c207c9d7610f79.mp4
PROJECT=/home/space2/songwei/lab_scene01_3dgut
NS_DATA="$PROJECT/nerfstudio_data"
BASE_COLMAP="$PROJECT/colmap_input"
UNDIST="$PROJECT/colmap_undistorted"
OUT="$PROJECT/output"
EXP=scene01_3dgut_mcmc_400k_undistorted
LOG="$PROJECT/logs/pipeline.log"
```

后台启动：

```bash
chmod +x /home/space2/songwei/run_d6184_3dgut_pipeline.sh

tmux new-session -d -s scene01_3dgut \
  "bash -lc '/home/space2/songwei/run_d6184_3dgut_pipeline.sh'"
```

查看日志：

```bash
tail -f /home/space2/songwei/lab_nurec_d6184_3dgut/logs/pipeline.log
```

### 3.2 3DGUT 核心训练命令

```bash
PROJECT_DIR=/home/space2/songwei/lab_scene01_3dgut
UNDISTORTED_DIR="$PROJECT_DIR/colmap_undistorted"
OUTPUT_DIR="$PROJECT_DIR/output"
EXPERIMENT_NAME=scene01_3dgut_mcmc_400k_undistorted

python train.py --config-name apps/colmap_3dgut_mcmc \
  path="$UNDISTORTED_DIR" \
  out_dir="$OUTPUT_DIR" \
  experiment_name="$EXPERIMENT_NAME" \
  n_iterations=10000 \
  test_last=false \
  compute_extra_metrics=false \
  strategy.add.max_n_gaussians=400000 \
  strategy.add.end_iteration=7000 \
  strategy.relocate.end_iteration=7000 \
  strategy.perturb.end_iteration=10000 \
  scheduler.positions.max_steps=10000 \
  checkpoint.iterations='${int_list:[7000,10000]}'
```

### 3.3 导出 NuRec USDZ

```bash
RUN_DIR=/替换为3DGUT运行目录
CHECKPOINT_PATH="$RUN_DIR/ckpt_last.pt"
OUTPUT_USDZ="$RUN_DIR/export_last_nurec_400k.usdz"
UNDISTORTED_DIR=/home/space2/songwei/lab_scene01_3dgut/colmap_undistorted

python -m threedgrut.export.scripts.export_usd \
  -c "$CHECKPOINT_PATH" \
  -o "$OUTPUT_USDZ" \
  --format nurec \
  --dataset "$UNDISTORTED_DIR"
```

训练输出：

```text
<运行目录>/
├── ckpt_last.pt
├── ours_7000/ckpt_7000.pt
├── ours_10000/ckpt_10000.pt
├── export_last_nurec_400k.usdz
└── 3dgut.usda
```

生成50%密度的场景

```bash
bash /home/space2/songwei/run_d6184_density50_export.sh 
```

---

## 4. 机器人与 3DGUT 场景对齐命令

不加载强化学习权重,导入50%密度场景：

```bash
cd /home/space2/songwei/rl_ws/IsaacLab

source /home/songwei/miniconda3/etc/profile.d/conda.sh
conda activate /home/space2/songwei/conda_envs/env_isaaclab
TERM=xterm source _isaac_sim/setup_conda_env.sh
export DISPLAY=:10

python scripts/reinforcement_learning/rsl_rl/play_o6_3dgut_scaled.py \
  --task Isaac-Lift-Cube-O6-Template-v0 \
  --num_envs 1 \
  --no_policy \
  --device cuda:1 \
  --usd_path /home/space2/songwei/lab_nurec_d6184_3dgut/transparent_50/3dgut_density50.usda \
  --task_translate -0.8 2.212 -2.008 \
  --task_orient_euler 0 0 -1.1 \
  --task_scale 2.6 \
  --ground_translate 0 0 -4.66 \
  --scene_pos 0 0 0 \
  --scene_orient_euler -105 7.2 12.2 \
  --scene_scale 1 1 1
```

最终确认物理缩放时，在命令中增加：

```bash
--bake_physics
```

---

## 5. 强化学习训练命令

训练时不加载 3DGUT，只训练缩放后的 Robot、Table、Object 和 GroundPlane。

缩放的目的是为了和场景对齐，不缩放场景的原因是因为会破坏场景

### 5.1 从头训练

```bash
cd /home/space2/songwei/rl_ws/IsaacLab

source /home/songwei/miniconda3/etc/profile.d/conda.sh
conda activate /home/space2/songwei/conda_envs/env_isaaclab
TERM=xterm source _isaac_sim/setup_conda_env.sh

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

显存不足时把 `4096` 降为 `2048`、`1024` 或 `512`。

### 5.2 断点续训

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
  --resume \
  --load_run 运行目录名称 \
  --checkpoint model_XXXX.pt
```

### 5.3 TensorBoard

```bash
tensorboard \
  --logdir /home/space2/songwei/rl_ws/IsaacLab/logs/rsl_rl \
  --port 6006 \
  --bind_all
```

---

## 6. 加载权重和 3DGUT 推理命令

```bash
cd /home/space2/songwei/rl_ws/IsaacLab

source /home/songwei/miniconda3/etc/profile.d/conda.sh
conda activate /home/space2/songwei/conda_envs/env_isaaclab
TERM=xterm source _isaac_sim/setup_conda_env.sh
export DISPLAY=:10

python scripts/reinforcement_learning/rsl_rl/play_o6_3dgut_scaled.py \
  --task Isaac-Lift-Cube-O6-Template-v0 \
  --num_envs 1 \
  --device cuda:1 \
  --checkpoint /home/space2/songwei/rl_ws/IsaacLab/logs/rsl_rl/o6_lift_cube/2026-08-10_16-48-06/model_14999.pt \
  --bake_physics \
  --task_translate -0.8 2.212 -2.008 \
  --task_orient_euler 0 0 -1.1 \
  --task_scale 2.6 \
  --ground_translate 0 0 -4.66 \
  --usd_path /home/space2/songwei/lab_nurec_d6184_3dgut/transparent_50/3dgut_density50.usda \
  --scene_pos 0 0 0 \
  --scene_orient_euler -105 7.2 12.2 \
  --scene_scale 1 1 1 \
  --real-time
```

---

## 7. 相机预览和数据集采集命令

### 7.1 预览相机，不加载权重

```bash
cd /home/space2/songwei/rl_ws/IsaacLab

source /home/songwei/miniconda3/etc/profile.d/conda.sh
conda activate /home/space2/songwei/conda_envs/env_isaaclab
TERM=xterm source _isaac_sim/setup_conda_env.sh
export DISPLAY=:10

python scripts/reinforcement_learning/rsl_rl/collect_o6_dual_camera_dataset.py \
  --preview_only \
  --task Isaac-Lift-Cube-O6-Template-v0 \
  --width 640 \
  --height 480 \
  --head_fov_deg 135 \
  --palm_fov_deg 135 \
  --head_camera_pos 0.12 0 0.04 \
  --head_camera_rot 0.991445 0 0.130526 0 \
  --palm_camera_pos 0.04 0 -0.03 \
  --palm_camera_rot 0.70710678 0 -0.70710678 0 \
  --usd_path /home/space2/songwei/lab_nurec_d6184_3dgut/transparent_50/3dgut_density50.usda \
  --task_translate -0.8 2.212 -2.008 \
  --task_orient_euler 0 0 -1.1 \
  --task_scale 2.6 \
  --ground_translate 0 0 -4.66 \
  --scene_pos 0 0 0 \
  --scene_orient_euler -105 7.2 12.2 \
  --scene_scale 1 1 1 \
  --bake_physics \
  --device cuda:1
```

### 7.2 采集 20 条完整轨迹

```bash
cd /home/space2/songwei/rl_ws/IsaacLab

source /home/songwei/miniconda3/etc/profile.d/conda.sh
conda activate /home/space2/songwei/conda_envs/env_isaaclab
TERM=xterm source _isaac_sim/setup_conda_env.sh
unset DISPLAY

CUDA_VISIBLE_DEVICES=1 python scripts/reinforcement_learning/rsl_rl/collect_o6_dual_camera_dataset.py \
  --task Isaac-Lift-Cube-O6-Template-v0 \
  --num_episodes 20 \
  --capture_every 1 \
  --warmup_steps 5 \
  --output_dir /home/space2/songwei/datasets/o6_rgb135_20episodes_50pct_model14999_001 \
  --width 640 \
  --height 480 \
  --head_fov_deg 135 \
  --palm_fov_deg 135 \
  --head_camera_pos 0.12 0 0.04 \
  --head_camera_rot 0.991445 0 0.130526 0 \
  --palm_camera_pos 0.04 0 -0.03 \
  --palm_camera_rot 0.70710678 0 -0.70710678 0 \
  --checkpoint /home/space2/songwei/rl_ws/IsaacLab/logs/rsl_rl/o6_lift_cube/2026-08-10_16-48-06/model_14999.pt \
  --usd_path /home/space2/songwei/lab_nurec_d6184_3dgut/transparent_50/3dgut_density50.usda \
  --task_translate -0.8 2.212 -2.008 \
  --task_orient_euler 0 0 -1.1 \
  --task_scale 2.6 \
  --ground_translate 0 0 -4.66 \
  --scene_pos 0 0 0 \
  --scene_orient_euler -105 7.2 12.2 \
  --scene_scale 1 1 1 \
  --bake_physics \
  --headless \
  --device cuda:0 \
  agent.device=cuda:0
```

数据集地址：

```text
/home/space2/songwei/datasets/o6_rgb135_20episodes_50pct_model14999_001
```

目录格式：

```text
o6_rgb135_20episodes_50pct_model14999_001/
├── dataset_info.json
├── episodes.json
└── episodes/
    ├── episode_000000/
    │   ├── frames.jsonl
    │   ├── trajectory_info.json
    │   ├── head/rgb/
    │   └── palm/rgb/
    ├── episode_000001/
    └── ...
```

---

## 8. 关键参数解释

### 8.1 物理任务参数

| 参数 | 当前值 | 作用 |
|---|---|---|
| `--task_translate` | `-0.8 2.212 -2.008` | Robot、Table、Object 整体平移 |
| `--task_orient_euler` | `0 0 -1.1` | 物理任务整体 XYZ 欧拉角，单位为度 |
| `--task_scale` | `2.6` | 物理任务统一放大 2.6 倍 |
| `--ground_translate` | `0 0 -4.66` | 物理 GroundPlane 位置 |
| `--bake_physics` | 开关 | 把缩放和变换写入物理配置，推理和采集必须使用 |

`train_o6_scaled.py` 会同步缩放机器人、桌子、物体、执行器参数、奖励距离阈值、重置范围和位置观测。

### 8.2 3DGUT 场景参数

| 参数 | 当前值 | 作用 |
|---|---|---|
| `--usd_path` | `3dgut_density50.usda` | 高斯场景入口 |
| `--scene_pos` | `0 0 0` | 高斯场景平移 |
| `--scene_orient_euler` | `-105 7.2 12.2` | 高斯场景 XYZ 欧拉角，单位为度 |
| `--scene_scale` | `1 1 1` | 高斯场景 XYZ 比例 |

`scene_*` 只影响高斯视觉场景，不改变机器人训练物理。

### 8.3 强化学习训练参数

| 参数 | 当前值 | 作用 |
|---|---:|---|
| `--num_envs` | 4096 | 并行环境数 |
| `--max_iterations` | 15000 | PPO 训练迭代数 |
| `--device` | `cuda:1` | 训练使用 GPU 1 |
| `--experiment_name` | 自定义 | `logs/rsl_rl/` 下的实验目录 |
| `--run_name` | 自定义 | 时间戳运行目录后缀 |
| `--resume` | 开关 | 启用断点续训 |
| `--load_run` | 运行目录 | 选择已有 run |
| `--checkpoint` | 权重名/绝对路径 | 加载 PPO 权重 |

当前动作维度为 8：右臂 7 维，灵巧手开合 1 维。

### 8.4 3DGUT 训练参数

| 参数 | 当前值 | 作用 |
|---|---:|---|
| `n_iterations` | 10000 | 3DGUT 总训练步数 |
| `max_n_gaussians` | 400000 | Gaussian 数量上限 |
| `add.end_iteration` | 7000 | 停止新增 Gaussian 的步数 |
| `relocate.end_iteration` | 7000 | 停止 relocate 的步数 |
| `checkpoint.iterations` | 7000、10000 | checkpoint 保存节点 |

### 8.5 相机与数据集参数

| 参数 | 当前值 | 作用 |
|---|---|---|
| `--num_episodes` | 20 | 采集 20 条完整重置轨迹 |
| `--capture_every` | 1 | 每个策略步保存一次 |
| `--width/--height` | `640/480` | RGB 分辨率 |
| `--head_fov_deg` | 135 | 头部相机水平 FOV |
| `--palm_fov_deg` | 135 | 手部相机水平 FOV |
| `--warmup_steps` | 5 | 正式记录前预热步数 |
| `--output_dir` | 新目录 | 数据集输出地址，必须为空或不存在 |

正式数据集只保存 RGB。IsaacLab 地面和桌面视觉默认隐藏，但碰撞保留；目标箭头和末端坐标箭头关闭。

只记录 18 个关节：右臂 7 个、右手灵巧手 11 个。具体名称和顺序保存在：

```text
dataset_info.json -> recorded_joint_names
```

### 8.6 GPU 映射参数

```bash
CUDA_VISIBLE_DEVICES=1 ... --device cuda:0 agent.device=cuda:0
```

含义：只让进程看到物理 GPU 1，该 GPU 在进程内编号为逻辑 `cuda:0`。数据集采集配合 `model_14999_cpu.pt` 使用。

---

## 9. 最终参数备忘

```text
Task：Isaac-Lift-Cube-O6-Template-v0

物理任务：
translate = -0.8 2.212 -2.008
orient = 0 0 -1.1 degrees
scale = 2.6
ground = 0 0 -4.66

3DGUT：
position = 0 0 0
orient = -105 7.2 12.2 degrees
scale = 1 1 1
density = 50%

相机：
head FOV = 135 degrees
palm FOV = 135 degrees
head position = 0.12 0 0.04
head WXYZ = 0.991445 0 0.130526 0
palm position = 0.04 0 -0.03
palm WXYZ = 0.70710678 0 -0.70710678 0

数据集：
20 complete episodes
RGB only
18 joints
```
