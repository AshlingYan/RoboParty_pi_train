# 在运行之前，请务必完成这 2 步，否则代码会报错：

# 步骤 A：创建指令文件
# 去你的数据文件夹 /root/autodl-tmp/RoboParty_pi/ygx_data/stack_blocks_three_1/。

# 查看里面的 config.json，找到 task_name 的值。假设它的值是 "stack_blocks"。

# 回到脚本所在的目录，新建文件夹 task_instructions。

# 在里面新建文件 stack_blocks.json (名字必须和 task_name 一样)。

# 写入内容：

# JSON

# {
#     "instruction": "Please stack the blocks on the table.",
#     "language": "english"
# }
# 步骤 B：运行转换
# 使用以下命令运行脚本（假设脚本名为 convert_to_lerobot.py）：

# Bash

# python convert_to_lerobot.py /root/autodl-tmp/RoboParty_pi/ygx_data/stack_blocks_three_1 ygx/stack_blocks_test_v1
# 参数1: 你的数据文件夹路径。

# 参数2: 你给这个数据集起的名字（随意起，比如 ygx/test01）。

import os
import sys

# ==============================================================================
# 1. 【核心修改】设置保存路径到代码的上级目录
# ==============================================================================
# 获取当前脚本所在的绝对路径
current_script_path = os.path.abspath(__file__)
# 获取当前脚本所在的目录
current_dir = os.path.dirname(current_script_path)
# 获取上级目录的 'data_lerobot' 路径
target_data_dir = os.path.abspath(os.path.join(current_dir, "..", "data_lerobot"))

# 创建这个文件夹（如果不存在）
os.makedirs(target_data_dir, exist_ok=True)

# 强制设置 HF_HOME 环境变量
# 注意：这必须在 import MyLerobotDataset 之前执行才有效！
print(f"[Info] 将数据保存路径设置为: {target_data_dir}")
os.environ["HF_HOME"] = target_data_dir
# ==============================================================================

# 添加项目路径 (保持你之前的设置)
sys.path.append("/root/autodl-tmp/RoboParty_pi/control_your_robot")

from data.generate_lerobot import MyLerobotDataset
from utils.data_handler import *
import argparse
import json

if __name__== '__main__':
    # ... (以下内容保持不变，直接复制之前的逻辑即可) ...
    
    features={
        "observation.images.cam_high": {
            "dtype": "video",   # <--- 修改这里：从 image 改为 video
            "shape": (3, 480, 640),
            "names": ["channels", "height", "width"],
        },
        "observation.images.cam_left_wrist": {
            "dtype": "video",   # <--- 修改这里
            "shape": (3, 480, 640),
            "names": ["channels", "height", "width"],
        },
        "observation.images.cam_right_wrist": {
            "dtype": "video",   # <--- 修改这里
            "shape": (3, 480, 640),
            "names": ["channels", "height", "width"],
        },
        "observation.state": { 
            "dtype": "float64",
            "shape": (14,),
            "names": ["l1,l2,l3,l4,l5,l6,gl,r1,r2,r3,r4,r5,r6,gr"],
        },
        "action": {
            "dtype": "float64",
            "shape": (14,),
            "names": ["l1,l2,l3,l4,l5,l6,gl,r1,r2,r3,r4,r5,r6,gr"],
        }
    }

    feature_map = {
            # 1. 映射三个摄像头视图 (把 / 改成 .)
            "observation.images.cam_high": "cam_head.color",        # 之前是 cam_head/color
            "observation.images.cam_left_wrist": "cam_left_wrist.color", # 之前是 /
            "observation.images.cam_right_wrist": "cam_right_wrist.color", # 之前是 /
            
            # 2. 映射状态 (State)
            # 注意：这里的列表里的内容也要改！
            "observation.state": ["left_arm.joint", "left_arm.gripper", "right_arm.joint", "right_arm.gripper"],
            
            # 3. 映射动作 (Action)
            "action": ["left_arm.joint", "left_arm.gripper", "right_arm.joint", "right_arm.gripper"]
        }
    
    parser = argparse.ArgumentParser(description='Transform datasets to LeRobot format.')
    parser.add_argument('data_path', type=str, help="raw data path containing .hdf5 files")
    parser.add_argument('repo_id', type=str, help='HuggingFace repo_id')
    parser.add_argument('--multi', action='store_true', default=False, help="Enable for multi-task folder structure")
    args = parser.parse_args()
    
    data_path = args.data_path
    repo_id = args.repo_id
    multi = args.multi
    
    # ================= 修复开始 =================
    print(f"正在路径中查找文件: {data_path}")
    
    # 尝试查找 .hdf5
    hdf5_paths = get_files(data_path, "*.hdf5")
    
    # 如果没找到，尝试查找 .h5 (防止后缀写错)
    if not hdf5_paths:
        print("未找到 .hdf5 文件，尝试查找 .h5 文件...")
        hdf5_paths = get_files(data_path, "*.h5")

    # 如果还是没找到，尝试递归查找 (假设 get_files 支持递归，或者我们手动用 glob)
    if not hdf5_paths:
        import glob
        print("尝试递归查找子目录...")
        # 这里的 /**/*.hdf5 需要 glob 配合 recursive=True
        hdf5_paths = glob.glob(os.path.join(data_path, "**", "*.hdf5"), recursive=True)

    print(f"--> 共找到 {len(hdf5_paths)} 个数据文件")

    if len(hdf5_paths) == 0:
        print("【错误】: 找不到任何数据文件！程序终止。")
        print("请检查：")
        print("1. 文件夹路径是否正确？")
        print("2. 文件夹里真的有 .hdf5 文件吗？还是在子文件夹里？")
        sys.exit(1) # 强制退出
    # ================= 修复结束 =================

    # ... (接你原来的 config 加载和 lerobot 初始化逻辑) ...
    
    # 注意：确保这一步逻辑是通的
    if not multi:
        config_path = os.path.join(data_path, "config.json")
        if os.path.exists(config_path):
            data_config = json.load(open(config_path))
            inst_path = f"./task_instructions/{data_config['task_name']}.json"
        else:
            print("【警告】根目录没有 config.json，无法自动加载指令！")
            inst_path = None
    else:
        inst_path = None

    lerobot = MyLerobotDataset(repo_id, "piper", 30, features, feature_map, inst_path)

    for hdf5_path in hdf5_paths:
        print(f"Processing: {hdf5_path}")
        try:
            data = hdf5_groups_to_dict(hdf5_path)
            
            if multi:
                # ... (multi 的逻辑) ...
                pass # 你的代码
            else:
                lerobot.write(data, inst_path)
        except Exception as e:
            print(f"读取文件失败 {hdf5_path}: {e}")

    # ==========================================
    # 关键补充：只有 MyLerobotDataset 类里有 .save() 方法时才需要。
    # 如果它是实时写入磁盘的（根据名字推测可能是），这步可能不需要。
    # 但通常 HuggingFace 数据集需要一个 push 或者 save_to_disk。
    # 假设你的类会在 __del__ 或者 write 内部处理，那就算了。
    # ==========================================

    print("转换完成！")