# O6 + Cosmos3 双相机 VLA 数据工作流

这个目录只放新流程脚本，不修改原来的 3DGUT 采集程序。流程是：

1. Isaac Sim 加载 PPO 权重，保存双相机 RGB、深度、语义/实例分割、动作和状态。
2. 把连续轨迹切成 Cosmos3 支持的 `8n+1` 帧短片。
3. 可用真实图片或视频作为外观参考，Isaac 深度与分割约束结构。
4. Cosmos3 生成真实感画面。
5. 用 Isaac 实例真值把机械臂和方块回贴，避免生成模型改变抓取关系。
6. 检查帧数/分辨率并导出对齐的双相机 VLA 数据。

## 0. 固定路径

```bash
export ISAAC_ROOT=/home/space2/songwei/rl_ws/IsaacLab
export PIPE_ROOT=$ISAAC_ROOT/scripts/vla_data_cosmos3
export RAW_ROOT=/home/space2/songwei/o6_vla_raw
export COSMOS_ROOT=/home/space2/songwei/cosmos3
export JOB_ROOT=$COSMOS_ROOT/o6_vla
export CKPT=/home/space2/songwei/rl_ws/IsaacLab/logs/rsl_rl/o6_lift_cube/2026-08-10_16-48-06/model_14999.pt
```

## 1. 准备真实图片或视频（可选）

真实参考最好来自同一工作台，头部和手掌摄像头各一段。静态图片也可以，打包器会自动循环成视频。

在服务器：

```bash
mkdir -p /home/space2/songwei/real_refs
```

在本地 Mac：

```bash
scp ~/Desktop/head_reference.mp4 songwei@10.1.47.59:/home/space2/songwei/real_refs/
scp ~/Desktop/palm_reference.mp4 songwei@10.1.47.59:/home/space2/songwei/real_refs/
```

## 2. 采集 Isaac 结构真值

Cosmos3 和 Isaac Sim 不要同时占用 GPU。先关闭 Cosmos3 服务，然后：

```bash
cd "$ISAAC_ROOT"
source /home/songwei/miniconda3/etc/profile.d/conda.sh
conda activate /home/space2/songwei/conda_envs/env_isaaclab
TERM=xterm source _isaac_sim/setup_conda_env.sh
unset DISPLAY

rm -rf "$RAW_ROOT/smoke"
CUDA_VISIBLE_DEVICES=1 python "$PIPE_ROOT/collect_o6_vla_base_dataset.py" \
  --task Isaac-Lift-Cube-O6-Template-v0 \
  --checkpoint "$CKPT" \
  --num_episodes 1 \
  --capture_every 1 \
  --output_dir "$RAW_ROOT/smoke" \
  --width 640 --height 480 \
  --head_body head2_link \
  --head_camera_pos 0.12 0 0.04 \
  --head_camera_rot 0.991445 0 0.130526 0 \
  --head_fov_deg 135 \
  --palm_body rh_hand_base_link \
  --palm_camera_pos 0.04 0 -0.03 \
  --palm_camera_rot 0.70710678 0 -0.70710678 0 \
  --palm_fov_deg 135 \
  --task_translate -0.8 2.212 -2.008 \
  --task_orient_euler 0 0 -1.1 \
  --task_scale 2.6 \
  --ground_translate 0 0 -4.66 \
  --bake_physics \
  --device cuda:0 \
  agent.device=cuda:0 \
  --headless
```

检查小样：

```bash
find "$RAW_ROOT/smoke/episodes" -type f | head
du -sh "$RAW_ROOT/smoke"
```

确认摄像机位置和分割正确后，正式采集。输出目录必须为空：

```bash
rm -rf "$RAW_ROOT/train"
CUDA_VISIBLE_DEVICES=1 python "$PIPE_ROOT/collect_o6_vla_base_dataset.py" \
  --task Isaac-Lift-Cube-O6-Template-v0 \
  --checkpoint "$CKPT" \
  --num_episodes 10 \
  --capture_every 1 \
  --output_dir "$RAW_ROOT/train" \
  --width 640 --height 480 \
  --head_body head2_link \
  --head_camera_pos 0.12 0 0.04 \
  --head_camera_rot 0.991445 0 0.130526 0 \
  --head_fov_deg 135 \
  --palm_body rh_hand_base_link \
  --palm_camera_pos 0.04 0 -0.03 \
  --palm_camera_rot 0.70710678 0 -0.70710678 0 \
  --palm_fov_deg 135 \
  --task_translate -0.8 2.212 -2.008 \
  --task_orient_euler 0 0 -1.1 \
  --task_scale 2.6 \
  --ground_translate 0 0 -4.66 \
  --bake_physics \
  --device cuda:0 \
  agent.device=cuda:0 \
  --headless
```

## 3. 打包 Cosmos3 控制视频

以下命令每 5 个策略步取一帧，以 10 FPS 输出；每个 clip 最多 49 帧，并自动保证帧数为 `8n+1`。

```bash
mkdir -p "$JOB_ROOT"
rm -rf "$JOB_ROOT/controls_smoke"
python "$PIPE_ROOT/pack_o6_cosmos_controls.py" \
  --dataset_dir "$RAW_ROOT/smoke" \
  --output_dir "$JOB_ROOT/controls_smoke" \
  --stride 5 \
  --max_frames 49 \
  --fps 10 \
  --width 640 --height 480 \
  --depth_near 0.05 --depth_far 5.0 \
  --head_reference /home/space2/songwei/real_refs/head_reference.mp4 \
  --palm_reference /home/space2/songwei/real_refs/palm_reference.mp4
```

如果暂时没有真实参考，删除最后两行 `--head_reference` 和 `--palm_reference`；Cosmos3 会使用 Isaac RGB 作为输入参考。

查看控制视频：

```bash
find "$JOB_ROOT/controls_smoke" -name manifest.json -print
ffprobe -v error -show_entries stream=width,height,nb_frames,avg_frame_rate \
  "$JOB_ROOT/controls_smoke/episode_000000/head/clip_0000/input_reference.mp4"find "$JOB_ROOT/controls_smoke" -name manifest.json -print
ffprobe -v error -show_entries stream=width,height,nb_frames,avg_frame_rate \
  "$JOB_ROOT/controls_smoke/episode_000000/head/clip_0000/input_reference.mp4"
```

## 4. 启动 Cosmos3-Nano

退出 Isaac 环境，确认两张 4090 空闲：

```bash
conda deactivate || true
nvidia-smi
```

在终端 A 启动服务：

```bash
cd /home/space2/songwei/cosmos3
source /home/space2/songwei/cosmos3/env.sh
source /home/space2/songwei/cosmos3/runtime/.venv/bin/activate
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export COSMOS_MODEL=/home/space2/songwei/cosmos3/models/Cosmos3-Nano

vllm-omni serve "$COSMOS_MODEL" \
  --omni \
  --model-class-name Cosmos3OmniDiffusersPipeline \
  --served-model-name nvidia/Cosmos3-Nano \
  --host 127.0.0.1 \
  --port 8000 \
  --allowed-local-media-path /home/space2/songwei/cosmos3 \
  --tensor-parallel-size 2 \
  --enable-layerwise-offload \
  --no-guardrails \
  --stage-init-timeout 1800 \
  --init-timeout 1800
```

`Multi-thread loading shards: 0/7` 是在从磁盘读权重，不是卡死。第一次可能需要数分钟；另开终端查看 GPU 显存和磁盘读取即可，不要重复启动服务。

终端 B 验证：

```bash
source /home/space2/songwei/cosmos3/env.sh
source /home/space2/songwei/cosmos3/runtime/.venv/bin/activate
curl -sS http://127.0.0.1:8000/v1/models | python -m json.tool
python -c "import requests; print(requests.__version__)"
```

如果最后一条缺包：

```bash
python -m pip install requests
```

## 5. 先跑一个 Cosmos3 小样

```bash
rm -rf "$JOB_ROOT/cosmos_smoke"
python "$PIPE_ROOT/run_cosmos3_transfer.py" \
  --controls_dir "$JOB_ROOT/controls_smoke" \
  --output_dir "$JOB_ROOT/cosmos_smoke" \
  --controls depth seg \
  --num_inference_steps 20 \
  --limit 1
```

生成位置：

```bash
find "$JOB_ROOT/cosmos_smoke" -name cosmos.mp4 -print
```

如果双控制导致显存不够，先只用深度：

```bash
rm -rf "$JOB_ROOT/cosmos_smoke"
python "$PIPE_ROOT/run_cosmos3_transfer.py" \
  --controls_dir "$JOB_ROOT/controls_smoke" \
  --output_dir "$JOB_ROOT/cosmos_smoke" \
  --controls depth \
  --num_inference_steps 20 \
  --limit 1
```

## 6. 回贴机械臂与方块真值并质检

```bash
rm -rf "$JOB_ROOT/composite_smoke"
python "$PIPE_ROOT/composite_o6_foreground.py" \
  --controls_dir "$JOB_ROOT/controls_smoke" \
  --cosmos_dir "$JOB_ROOT/cosmos_smoke" \
  --output_dir "$JOB_ROOT/composite_smoke" \
  --mask_blur 2 \
  --limit 2

python "$PIPE_ROOT/validate_cosmos3_outputs.py" \
  --controls_dir "$JOB_ROOT/controls_smoke" \
  --video_dir "$JOB_ROOT/composite_smoke" \
  --video_name composite.mp4 \
  --report "$JOB_ROOT/composite_smoke/validation.json" \
  --limit 2
```

还需要人工观看三个视频：`input_reference.mp4`、`cosmos.mp4`、`composite.mp4`。只有方块、夹爪接触和相机运动都正确的 clip 才能进入 VLA 训练。

## 7. 正式批处理

先打包全部原始轨迹：

```bash
rm -rf "$JOB_ROOT/controls_train"
python "$PIPE_ROOT/pack_o6_cosmos_controls.py" \
  --dataset_dir "$RAW_ROOT/train" \
  --output_dir "$JOB_ROOT/controls_train" \
  --stride 5 \
  --max_frames 49 \
  --fps 10 \
  --width 640 --height 480 \
  --depth_near 0.05 --depth_far 5.0 \
  --head_reference /home/space2/songwei/real_refs/head_reference.mp4 \
  --palm_reference /home/space2/songwei/real_refs/palm_reference.mp4
```

批量生成、合成、检查：

```bash
mkdir -p "$JOB_ROOT/cosmos_train"
python "$PIPE_ROOT/run_cosmos3_transfer.py" \
  --controls_dir "$JOB_ROOT/controls_train" \
  --output_dir "$JOB_ROOT/cosmos_train" \
  --controls depth seg \
  --num_inference_steps 35

rm -rf "$JOB_ROOT/composite_train"
python "$PIPE_ROOT/composite_o6_foreground.py" \
  --controls_dir "$JOB_ROOT/controls_train" \
  --cosmos_dir "$JOB_ROOT/cosmos_train" \
  --output_dir "$JOB_ROOT/composite_train" \
  --mask_blur 2

python "$PIPE_ROOT/validate_cosmos3_outputs.py" \
  --controls_dir "$JOB_ROOT/controls_train" \
  --video_dir "$JOB_ROOT/composite_train" \
  --video_name composite.mp4 \
  --report "$JOB_ROOT/composite_train/validation.json"
```

`run_cosmos3_transfer.py` 会跳过已经存在的 `cosmos.mp4`，所以中断后可直接执行同一条命令续跑。

## 8. 导出 VLA 训练目录

```bash
rm -rf "$JOB_ROOT/vla_train"
python "$PIPE_ROOT/export_o6_vla_dataset.py" \
  --controls_dir "$JOB_ROOT/controls_train" \
  --composite_dir "$JOB_ROOT/composite_train" \
  --output_dir "$JOB_ROOT/vla_train"

find "$JOB_ROOT/vla_train/episodes" -name frames.jsonl | wc -l
du -sh "$JOB_ROOT/vla_train"
```

每个导出 episode 包含：

```text
episodes/episode_XXXXXX/
  head/rgb/*.png
  palm/rgb/*.png
  frames.jsonl
  trajectory_info.json
```

`frames.jsonl` 保留 PPO 动作、关节位置/速度、方块姿态、目标姿态、相机姿态，并更新为 Cosmos3 双相机图像路径。

## 9. 最重要的限制

- Cosmos3 生成画面不能当物理真值；动作、状态、接触关系一律使用 Isaac 数据。
- 真实视频最好来自相同工作台和近似相机视角；完全不同的场景会与深度控制冲突。
- 当前 135° 视场角非常宽。正式采集前应使用真实相机标定出的内参/FOV 替换它。
- 小样通过后再跑 100 个 episode；先批量生成后发现相机或蒙版错误会浪费大量算力。
