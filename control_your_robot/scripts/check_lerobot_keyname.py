import pandas as pd
import numpy as np
from pathlib import Path

# 替换为你的数据集路径（根据你的 tree 结构，data 目录下的 parquet 文件）
data_path = Path("/root/autodl-tmp/RoboParty_pi/openpi/hf_lerobot_home/stack_blocks_three/meta/tasks.parquet")

# 读取 Parquet 文件
df = pd.read_parquet(data_path)

# 1. 输出所有字段（键名）
print("=== 数据集所有键名 ===")
print(df.columns.tolist())
print("\n")

# 2. 检查动作字段（通常是 "action" 或类似名称）
action_keys = [col for col in df.columns if "action" in col.lower()]
print("=== 动作相关键名 ===")
print(action_keys)
if action_keys:
    # 取第一个动作字段的示例数据，查看维度
    action_sample = df[action_keys[0]].iloc[0]
    action_array = np.array(action_sample)
    print(f"动作示例维度: {action_array.shape}")  # 确认动作维度
    print(f"动作示例数据: {action_array[:5]}...")  # 预览前5个值
print("\n")

# 3. 检查状态字段（通常是 "state"、"joint" 等）
state_keys = [col for col in df.columns if "state" in col.lower() or "joint" in col.lower()]
print("=== 状态相关键名 ===")
print(state_keys)
if state_keys:
    state_sample = df[state_keys[0]].iloc[0]
    state_array = np.array(state_sample)
    print(f"状态示例维度: {state_array.shape}")
print("\n")

# 4. 检查图像字段（根据你的目录，可能包含 "cam_high"、"cam_left_wrist" 等）
image_keys = [col for col in df.columns if "image" in col.lower() or "cam" in col.lower()]
print("=== 图像相关键名 ===")
print(image_keys)
if image_keys:
    # 图像通常存储为路径或数组，这里查看格式
    image_sample = df[image_keys[0]].iloc[0]
    print(f"图像示例格式: {type(image_sample)}")  # 若为数组则显示 shape
    if isinstance(image_sample, (np.ndarray, list)):
        print(f"图像形状: {np.array(image_sample).shape}")
print("\n")

# 5. 检查提示文本字段（通常是 "prompt"、"instruction"、"task" 等）
prompt_keys = [col for col in df.columns if "prompt" in col.lower() or "instruction" in col.lower() or "task" in col.lower()]
print("=== 提示文本相关键名 ===")
print(prompt_keys)
if prompt_keys:
    prompt_sample = df[prompt_keys[0]].iloc[0]
    print(f"提示示例: {prompt_sample}")