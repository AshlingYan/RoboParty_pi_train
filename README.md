6. RoboParty_pi_train环境配置：
conda create -n uv_envs_train python=3.10
conda activate uv_envs_train​ 
pip install uv
------------------------------------------
cd openpi/  # 需要该目录下的：pyproject.toml​0  改一下里面的name为openpi_train
GIT_LFS_SKIP_SMUDGE=1 uv sync # 根据pyproject.toml下载依赖包
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
cd openpi
source ./.venv/bin/activate
------------------------------------------
export OPENPI_DATA_HOME=/data0/syy_data/RoboParty_pi_train/openpi/openpi_data_home
export HF_LEROBOT_HOME="/data0/syy_data/RoboParty_pi_train/openpi/hf_lerobot_home"
------------------------------------------
uv run scripts/compute_norm_stats.py --config-name pi05_ygx
------------------------------------------
CUDA_VISIBLE_DEVICES=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train_pytorch.py pi05_ygx --exp_name stack_blocks_three_merged --resume
------------------------------------------
cd /data0/syy_data/RoboParty_pi_train/openpi

# 快速测试 RTC 训练
python scripts/train_pytorch.py pi05_ygx_rtc --exp_name my_rtc_run \
    --rtc_enabled \
    --rtc_simulated_delay 5 \
    --rtc_prefix_attention_schedule exp \
    --rtc_max_guidance_weight 5.0 \
    --num_train_steps 100000 \    
    --resume    
------------------------------------------        
python control_your_robot/example/deploy/piper_deploy_pi05_ygx.py \
  --remote-ws 127.0.0.1:8005 \
  --max-step 100 \
  --max-queue-size 50 \
  --task "There are three blocks on the table, the color of the blocks is red, green and blue. Stack blue on green, green on red."    
