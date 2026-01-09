import sys
sys.path.append("./")

from lerobot.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import json
from typing import Dict  # 补充类型注解，避免报错

from utils.data_handler import *

class MyLerobotDataset:
    def __init__(self, repo_id: str, robot_type: str, fps: int, features: dict, map: dict, intruction_path: str):
        self.dataset = LeRobotDataset.create(
            repo_id=repo_id,
            robot_type=robot_type,
            fps=fps,
            features=features,
            image_writer_threads=10,
            image_writer_processes=5,   
        )
        self.map = map
        self.intruction_path = intruction_path  # 注：此处拼写错误，建议改为 instruction_path

    def get_random_intruction(self, path=None):  # 注：此处拼写错误，建议改为 get_random_instruction
        if path is None:
            with open(self.intruction_path, 'r') as f_instr:
                instruction_dict = json.load(f_instr)
                instructions = instruction_dict['instructions']
                instruction = np.random.choice(instructions)
                return instruction
        else:
            with open(path, 'r') as f_instr:
                instruction_dict = json.load(f_instr)
                instructions = instruction_dict['instructions']
                instruction = np.random.choice(instructions)
                return instruction

    def write(self, data: Dict, path=None):
        base_frame = {}
        if self.intruction_path is None:
            # multi task need multi instruction
            instruction = self.get_random_intruction(path) 
        else:
            # single task, read default instruction path
            instruction = self.get_random_intruction() 
        for key, value in self.map.items():
            base_frame[key] = np.array(get_item(data, value))
        episode_length = base_frame[list(base_frame.keys())[0]].shape[0]
        for i in range(episode_length):
            frame = {}
            for key, value in base_frame.items():
                frame[key] = value[i]   
            # 【核心修改】删除 frame["task"] = instruction，改为传入 task 参数
            self.dataset.add_frame(frame, task=instruction)  # 补充 task 参数
        self.dataset.save_episode()
        # for lerobot 1.8 -> openpi
        # self.dataset.save_episode(task=instruction)
    
    # only for lerobot 1.8
    def consolidate(self):
        self.dataset.consolidate()