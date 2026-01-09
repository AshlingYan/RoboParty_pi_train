#!/usr/bin/env python3
"""
直接从第8000个样本开始测试，验证修复效果
"""

import numpy as np
import tqdm
import tyro

import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
    start_from_sample: int = 0,  # 新增参数：从哪个样本开始
) -> tuple[_data_loader.Dataset, int]:
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    if data_config.repo_id == "fake":
        return _data_loader.FakeDataset(model_config, num_samples=1024), 0

    # Check if repo_id is a local path
    is_local = data_config.repo_id.startswith("/") or data_config.repo_id.startswith(".")

    # 创建数据集
    dataset = _data_loader.create_torch_dataset(data_config, action_horizon, model_config)

    # 创建包装器，只加载从指定样本开始的数据
    class SubsetDataset:
        def __init__(self, dataset, start_idx):
            self.dataset = dataset
            self.start_idx = start_idx
            self.original_length = len(dataset)
            self.new_length = self.original_length - start_idx

        def __len__(self):
            return max(0, self.new_length)

        def __getitem__(self, idx):
            # 映射到原始数据集的索引
            original_idx = self.start_idx + idx
            if original_idx >= self.original_length:
                raise IndexError(f"Index {original_idx} out of range for dataset of length {self.original_length}")
            return self.dataset[original_idx]

    # 应用子集
    if start_from_sample > 0:
        print(f"🔍 直接从第 {start_from_sample} 个样本开始测试")
        print(f"📊 跳过前 {start_from_sample} 个样本")
        print(f"📈 将测试 {len(dataset) - start_from_sample} 个样本")
        dataset = SubsetDataset(dataset, start_from_sample)

    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            RemoveStrings(),
        ],
    )

    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False

    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def main(config_name: str, start_from_sample: int = 8000, max_frames: int | None = None):
    print("=" * 60)
    print(f"🧪 测试修复效果 - 从第 {start_from_sample} 个样本开始")
    print("=" * 60)

    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)

    if data_config.rlds_data_dir is not None:
        data_loader, num_batches = create_rlds_dataloader(
            data_config, config.model.action_horizon, config.batch_size, max_frames
        )
    else:
        data_loader, num_batches = create_torch_dataloader(
            data_config, config.model.action_horizon, config.batch_size, config.model, config.num_workers, max_frames, start_from_sample
        )

    # 初始化运行时统计器
    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    # 计数器
    processed_batches = 0
    skipped_batches = 0
    error_samples = 0

    print(f"📋 准备处理 {num_batches} 个批次")
    print(f"⚡ 开始从样本 {start_from_sample} 处理...")

    # 遍历数据集计算统计量
    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Testing from sample 8000"):
        try:
            for key in keys:
                if key in batch:
                    stats[key].update(np.asarray(batch[key]))
            processed_batches += 1

            # 每处理100个批次报告一次
            if processed_batches % 100 == 0:
                print(f"✅ 已处理 {processed_batches} 个批次")

        except Exception as e:
            skipped_batches += 1
            error_samples += 1
            print(f"❌ 第 {processed_batches + skipped_batches} 批次出错: {str(e)[:100]}...")
            continue

    print("\n" + "=" * 60)
    print("🎉 测试完成!")
    print(f"✅ 成功处理批次: {processed_batches}")
    print(f"❌ 跳过批次: {skipped_batches}")
    print(f"📈 成功率: {(processed_batches / (processed_batches + skipped_batches)) * 100:.1f}%")

    if processed_batches > 0:
        print("\n📊 计算的统计量:")
        for key in keys:
            stats_dict = stats[key].get_statistics()
            print(f"  {key}: mean={stats_dict['mean']:.4f}, std={stats_dict['std']:.4f}")

        # 检查是否有足够的样本
        total_samples = processed_batches * config.batch_size
        print(f"\n📊 处理的样本总数: {total_samples}")

        if total_samples > 1000:
            print("✅ 样本数量足够，计算结果可信")
        else:
            print("⚠️  样本数量较少，建议检查是否有问题")

    print("=" * 60)


if __name__ == "__main__":
    tyro.cli(main)