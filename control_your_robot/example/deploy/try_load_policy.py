#!/usr/bin/env python3
import sys
import traceback
from pathlib import Path

print('=== try_load_policy.py starting ===')
print('python executable:', sys.executable)
print('python version:', sys.version)

# Ensure openpi src is on sys.path
HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[3]
OPENPI_SRC = REPO_ROOT / 'openpi' / 'src'
print('repo root:', REPO_ROOT)
print('openpi src:', OPENPI_SRC)
sys.path.insert(0, str(OPENPI_SRC))
sys.path.insert(0, str(REPO_ROOT / 'control_your_robot'))

try:
    from openpi.training import config as _config
    from openpi.policies import policy_config
    print('imported openpi modules OK')
except Exception:
    print('Failed to import openpi modules')
    traceback.print_exc()
    sys.exit(2)

cfg_name = 'pi05_ygx'
checkpoint_dir = '/root/autodl-tmp/RoboParty_pi/openpi/checkpoints/pi05_ygx/piper_ygx'

print('Attempting to get config:', cfg_name)
try:
    cfg = _config.get_config(cfg_name)
    print('Config loaded:', cfg.name)
except Exception:
    print('Failed to load config')
    traceback.print_exc()
    sys.exit(3)

print('Attempting to create_trained_policy from checkpoint:', checkpoint_dir)
try:
    policy = policy_config.create_trained_policy(cfg, checkpoint_dir, default_prompt=None)
    print('create_trained_policy returned:', type(policy))
except Exception:
    print('create_trained_policy failed with exception:')
    traceback.print_exc()
    sys.exit(4)

print('SUCCESS: policy loaded')
print('policy repr:', repr(policy))
print('=== try_load_policy.py finished ===')
