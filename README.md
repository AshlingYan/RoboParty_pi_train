RoboParty_pi_train环境配置：

1、conda环境：
conda create -n uv_envs_train python=3.10
conda activate uv_envs_train​ 
pip install uv

2、uv环境：
cd openpi/  # 需要该目录下的：pyproject.toml​0  改一下里面的name为openpi_train
GIT_LFS_SKIP_SMUDGE=1 uv sync # 根据pyproject.toml下载依赖包
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
cd openpi
source ./.venv/bin/activate

3、缓存目录修改（hugging face和openpi）：
export OPENPI_DATA_HOME=/data0/syy_data/RoboParty_pi_train/openpi/openpi_data_home
export HF_LEROBOT_HOME="/data0/syy_data/RoboParty_pi_train/openpi/hf_lerobot_home"

4、计算norm_stats：
uv run scripts/compute_norm_stats.py --config-name pi05_ygx

5、正式训练：
CUDA_VISIBLE_DEVICES=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train_pytorch.py pi05_ygx --exp_name stack_blocks_three_merged --resume
