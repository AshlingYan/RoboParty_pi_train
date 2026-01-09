import os
# 设置环境变量，确保数据集保存在期望的位置
# os.environ["HF_LEROBOT_HOME"] = "/home/ygx/control_your_robot/datasets/lerobot_dataset_02"

from pathlib import Path

import dataclasses
import shutil
from typing import Literal

import h5py
from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import torch
import tqdm
import tyro
import json
import fnmatch
import re
import cv2


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = False
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()


def create_empty_dataset(
    repo_id: str,
    robot_type: str,
    mode: Literal["video", "image"] = "image",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> LeRobotDataset:
    # 状态和动作的维度名称保持不变 (14 DoF for Aloha)
    motors = [
        "left_waist", "left_shoulder", "left_elbow", "left_forearm_roll", "left_wrist_angle", "left_wrist_rotate", "left_gripper",
        "right_waist", "right_shoulder", "right_elbow", "right_forearm_roll", "right_wrist_angle", "right_wrist_rotate", "right_gripper",
    ]

    # LeRobotDataset 的期望相机名称
    cameras = ["head_camera", "left_camera", "right_camera"]

    features = {
        "observation.state": {"dtype": "float32", "shape": (len(motors),), "names": [motors]},
        "action": {"dtype": "float32", "shape": (len(motors),), "names": [motors]},
    }

    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": "image", "shape": (3, 480, 640), "names": ["channels", "height", "width"],
        }

    if Path(HF_LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(HF_LEROBOT_HOME / repo_id)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=30,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )


def get_cameras(hdf5_file: Path) -> list[str]:
    """从 HDF5 文件的根目录查找以 'cam_' 开头的组作为相机列表。"""
    with h5py.File(hdf5_file, "r") as ep:
        return [key for key in ep.keys() if key.startswith("cam_")]


def load_raw_images_per_camera(ep: h5py.File, cameras: list[str]) -> dict[str, np.ndarray]:
    """修改图像加载路径并添加相机名称映射。"""
    imgs_per_cam = {}
    
    camera_name_map = {
        "cam_head": "head_camera", "cam_left_wrist": "left_camera", "cam_right_wrist": "right_camera",
    }
    
    for camera_hdf5_name in cameras:
        dataset_path = f"/{camera_hdf5_name}/color"
        if dataset_path not in ep:
            print(f"Warning: Dataset path '{dataset_path}' not found in HDF5 file.")
            continue
            
        uncompressed = ep[dataset_path].ndim == 4

        if uncompressed:
            imgs_array = ep[dataset_path][:]
        else:
            imgs_array = []
            for data in ep[dataset_path]:
                data = np.frombuffer(data, np.uint8)
                img = cv2.imdecode(data, cv2.IMREAD_COLOR)
                imgs_array.append(img)
            imgs_array = np.stack(imgs_array, axis=0)

        if imgs_array.shape[1:3] != (480, 640):
            resized_imgs = [cv2.resize(img, (640, 480)) for img in imgs_array]
            imgs_array = np.stack(resized_imgs, axis=0)

        imgs_array = np.transpose(imgs_array, (0, 3, 1, 2))
        
        lerobot_camera_name = camera_name_map.get(camera_hdf5_name)
        if lerobot_camera_name:
            imgs_per_cam[lerobot_camera_name] = imgs_array
        else:
            print(f"Warning: Camera '{camera_hdf5_name}' not in name map, skipping.")
            
    return imgs_per_cam


def load_raw_episode_data(ep_path: Path) -> tuple[dict[str, np.ndarray], torch.Tensor, torch.Tensor]:
    """从 `left_arm` 和 `right_arm` 组中读取数据并拼接成 state 和 action。"""
    with h5py.File(ep_path, "r") as ep:
        left_qpos = ep["/left_arm/qpos"][:]
        left_gripper = ep["/left_arm/gripper"][:]
        right_qpos = ep["/right_arm/qpos"][:]
        right_gripper = ep["/right_arm/gripper"][:]

        left_gripper = left_gripper[:, np.newaxis]
        right_gripper = right_gripper[:, np.newaxis]

        state_np = np.concatenate([left_qpos, left_gripper, right_qpos, right_gripper], axis=1)
        
        state = torch.from_numpy(state_np).float()
        action = state.clone()

        imgs_per_cam = load_raw_images_per_camera(ep, get_cameras(ep_path))

    return imgs_per_cam, state, action


def populate_dataset(
    dataset: "LeRobotDataset",
    hdf5_files: list[Path],
    instruction_dir: Path,
    task: str, # `task` 参数现在用于备用或分类，实际指令来自json
) -> "LeRobotDataset":
    """
    通过遍历 HDF5 和对应的 JSON 指令文件来填充 LeRobotDataset。
    """
    for ep_path in tqdm.tqdm(hdf5_files, desc="Processing episodes"):
        match = re.search(r"(\d+)\.hdf5", os.path.basename(ep_path))
        if not match:
            print(f"Warning: Skipping file with unexpected name format: {os.path.basename(ep_path)}")
            continue
        
        ep_idx = int(match.group(1))

        # --- INSTRUCTION LOADING LOGIC ---
        # 查找与HDF5文件编号匹配的json文件
        json_path = instruction_dir / f"{ep_idx}.json"
        if not json_path.exists():
            print(f"Warning: Instruction file not found for episode {ep_idx}, skipping. Path: {json_path}")
            continue
            
        try:
            with open(json_path, 'r') as f_instr:
                instruction_dict = json.load(f_instr)
                # 假设指令存储在 'seen' 键下
                instructions = instruction_dict.get('seen', [])
                if not instructions:
                    print(f"Warning: No instructions found in 'seen' key for {json_path}, skipping episode.")
                    continue
                # 随机选择一条指令用于整个episode
                instruction = np.random.choice(instructions)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error reading or parsing instruction file {json_path}: {e}. Skipping episode.")
            continue
        # --- END INSTRUCTION LOGIC ---

        imgs_per_cam, state, action = load_raw_episode_data(ep_path)
        num_frames = state.shape[0]
        
        for i in range(num_frames):
            frame = {
                "observation.state": state[i],
                "action": action[i],
            }
            for camera_lerobot_name, img_array in imgs_per_cam.items():
                frame[f"observation.images.{camera_lerobot_name}"] = img_array[i]
            
            # 使用从json文件中加载的指令
            dataset.add_frame(frame, task=instruction)
            
        dataset.save_episode()

    return dataset


def port_aloha(
    raw_dir: Path,
    instruction_dir: Path, # 添加 instruction_dir 参数
    repo_id: str,
    raw_repo_id: str | None = None,
    task: str = "DEBUG",
    *,
    push_to_hub: bool = False,
    is_mobile: bool = False,
    mode: Literal["video", "image"] = "image",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
):
    repo_id = f"{repo_id}/{task}"
    if (HF_LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(HF_LEROBOT_HOME / repo_id)

    if not raw_dir.exists():
        if raw_repo_id is None:
            raise ValueError("raw_repo_id must be provided if raw_dir does not exist")
        # download_raw(raw_dir, repo_id=raw_repo_id)
        
    hdf5_files = []
    for root, _, files in os.walk(raw_dir):
        for filename in fnmatch.filter(files, '*.hdf5'):
            file_path = os.path.join(root, filename)
            hdf5_files.append(Path(file_path))
            
    if not hdf5_files:
        print(f"Error: No HDF5 files found in the specified raw_dir: {raw_dir}")
        return

    hdf5_files = sorted(hdf5_files, key=lambda x: int(re.search(r"(\d+)\.hdf5", x.name).group(1)))
    
    dataset = create_empty_dataset(
        repo_id,
        robot_type="mobile_aloha" if is_mobile else "aloha_agilex",
        mode=mode,
        dataset_config=dataset_config,
    )
    
    dataset = populate_dataset(
        dataset,
        hdf5_files,
        instruction_dir, # 传递 instruction_dir
        task=task,
    )

    if push_to_hub:
        dataset.push_to_hub()


if __name__ == "__main__":
    # 使用这个脚本时，请确保 instruction_dir 指向包含 0.json, 1.json ... 的目录
    port_aloha(
        raw_dir=Path("/root/autodl-tmp/RoboParty_pi/company_data/stack_blocks_three_real/stack_blocks_three"),
        instruction_dir=Path("/root/autodl-tmp/RoboParty_pi/control_your_robot/task_instructions"), # <--- 修改这里
        repo_id="piper",
        task="small"
    )