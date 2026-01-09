#!/bin/bash
# ======================
# Piper双臂PI0.5微调启动脚本
# ======================
set -e  # 出错立即停止

# 1. 环境配置（PI0.5显存优化）
export LEROBOT_HOME="/root/autodl-tmp/RoboParty_pi/openpi/hf_lerobot_home"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9  # 限制XLA显存占用
export XLA_PYTHON_CLIENT_ALLOCATOR=platform  # 优化显存分配
export PYTHONPATH="${PYTHONPATH}:/root/autodl-tmp/RoboParty_pi/openpi"  # 项目根路径
export TF_ENABLE_ONEDNN_OPTS=0  # 禁用不必要的优化（避免冲突）

# 2. 计算Piper数据集归一化统计量（首次运行必须）
echo -e "\n========== 计算Piper数据集统计量 =========="
uv run scripts/compute_norm_stats.py --config-name pi0_5_dual_piper_finetune

# 3. 启动PI0.5微调训练
echo -e "\n========== 启动Piper双臂PI0.5微调 =========="
uv run scripts/train.py pi0_5_dual_piper_finetune \
  --exp-name=dual_piper_pi0_5_finetune \
  --overwrite \
  --log-dir=./logs/dual_piper_pi0_5 \
  --workdir=./outputs/dual_piper_pi0_5 \
  --resume=False  # 从头微调（如需续训改为True）

# 4. 训练完成提示
echo -e "\n========== Piper双臂PI0.5微调完成 =========="
echo "模型保存路径: ./outputs/dual_piper_pi0_5"
echo "日志路径: ./logs/dual_piper_pi0_5"