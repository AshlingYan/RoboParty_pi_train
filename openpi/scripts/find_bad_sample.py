#!/usr/bin/env python3
import traceback
from openpi.training import config as _config
from openpi.training.data_loader import create_torch_dataset, transform_dataset

cfg = _config.get_config("pi05_ygx")
data_cfg = cfg.data.create(cfg.assets_dirs, cfg.model)

raw_dataset = create_torch_dataset(data_cfg, cfg.model.action_horizon, cfg.model)
dataset = transform_dataset(raw_dataset, data_cfg, skip_norm_stats=True)


def find_strings(obj, prefix=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            find_strings(v, f"{prefix}{k}.")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            find_strings(v, f"{prefix}{i}.")
    elif isinstance(obj, str):
        if "/" in obj or obj.endswith((".mp4", ".avi", ".mkv")):
            print(f"{prefix[:-1]} -> {obj}")


for i in range(len(dataset)):
    try:
        _ = dataset[i]
    except Exception as e:
        print("Error at dataset index:", i)
        traceback.print_exc()
        try:
            raw = raw_dataset[i]
            print("Raw sample string-like fields (possible paths):")
            find_strings(raw)
        except Exception as re:
            print("Also failed reading raw dataset index:", i, re)
        break
    if i % 1000 == 0:
        print("checked", i)
