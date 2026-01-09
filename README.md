以下是为你优化后的 GitHub README 文档，采用了更可爱、易读的风格，添加了图标、分区、醒目提示和清晰的步骤指引，直接复制即可使用：

```markdown
# 🤖 RoboParty_pi_train 使用指南
> 超详细的训练环境配置 & 运行教程，新手友好版 💖

## 📋 前置说明
由于部分大文件目录被 `.gitignore` 排除未上传，使用前请确保本地已有以下目录（总大小约95.5GB）：
| 目录路径 | 大小 | 说明 |
|----------|------|------|
| `openpi/checkpoints/` | 40GB | 模型权重文件 |
| `openpi/hf_lerobot_home/` | 33GB | 数据集文件 |
| `openpi/openpi_data_home/` | 14GB | 基础数据文件 |
| `openpi/examples/` | 4.5GB | 示例文件 |
| `openpi/wandb/` | - | 日志文件目录 |
| `openpi/third_party/` | - | 第三方依赖目录 |

## 🚀 快速开始
### 1. 创建并激活 Conda 环境
```bash
# 创建名为 uv_envs_train 的 conda 环境（Python 3.10）
conda create -n uv_envs_train python=3.10 -y

# 激活该环境
conda activate uv_envs_train

# 安装 uv 工具
pip install uv
```

### 2. 配置 uv 环境依赖
```bash
# 进入 openpi 目录（需确保该目录下有 pyproject.toml 文件）
cd openpi/

# ✨ 重要：先修改 pyproject.toml 中的 name 字段为 openpi_train

# 同步依赖包（跳过 Git LFS 大文件拉取）
GIT_LFS_SKIP_SMUDGE=1 uv sync

# 以可编辑模式安装本地包
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

# 激活 uv 生成的虚拟环境
cd openpi
source ./.venv/bin/activate
```

### 3. 设置缓存目录（关键步骤）
```bash
# 设置 OpenPI 数据缓存目录
export OPENPI_DATA_HOME=/data0/syy_data/RoboParty_pi_train/openpi/openpi_data_home

# 设置 Hugging Face LERobot 缓存目录
export HF_LEROBOT_HOME="/data0/syy_data/RoboParty_pi_train/openpi/hf_lerobot_home"
```
> 💡 提示：如果希望每次启动环境自动生效，可以将上述两行添加到 conda 环境的 `activate` 脚本中

### 4. 计算 norm_stats 统计值
```bash
# 使用 pi05_ygx 配置计算统计值
uv run scripts/compute_norm_stats.py --config-name pi05_ygx
```

### 5. 启动正式训练
```bash
# 指定 GPU 2 运行，配置 CUDA 内存策略，启动训练
CUDA_VISIBLE_DEVICES=2 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train_pytorch.py pi05_ygx --exp_name stack_blocks_three_merged --resume
```
> 📌 参数说明：
> - `--exp_name stack_blocks_three_merged`：指定实验名称
> - `--resume`：支持断点续训（如果有中断的训练任务）

## ❓ 常见问题小Tips
1. 🚨 如果遇到依赖安装失败：检查 `pyproject.toml` 的 `name` 是否已改为 `openpi_train`
2. 🚨 如果 GPU 内存不足：调整 `XLA_PYTHON_CLIENT_MEM_FRACTION` 的值（如改为 0.8）
3. 🚨 如果缓存目录报错：确认目录路径存在且有读写权限

## 📄 许可证
（如果有需要，可在此添加项目许可证信息）
```

### 总结
1. 文档新增了 emoji 图标、表格、提示框等元素，视觉上更友好，重点信息更突出；
2. 拆分了长命令行，添加了中文注释和参数说明，新手更容易理解每一步的作用；
3. 补充了关键提示（如配置自动生效、参数含义、常见问题），覆盖了实际使用中可能遇到的问题；
4. 保持了原有的核心命令和步骤，仅优化格式和可读性，不影响实际执行逻辑。

你可以根据项目实际情况，补充「许可证」「贡献指南」「联系方式」等模块，或调整 emoji 风格、颜色标记等细节。
