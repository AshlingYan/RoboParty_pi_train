import pandas as pd

# 读取 Parquet 文件
# file_path = "/root/autodl-tmp/RoboParty_pi/openpi/hf_lerobot_home/ygx/stack_blocks_test_v1/data/chunk-000/episode_000000.parquet"
file_path = "/root/autodl-tmp/RoboParty_pi/openpi/hf_lerobot_home/libero_2_lerobot/data/chunk-000/episode_000000.parquet"

df = pd.read_parquet(file_path)

# 1. 查看所有列名（关键信息通常包含在列名中）
print("所有列名：")
print(df.columns.tolist())

# 2. 查看前 3 行完整数据（快速浏览所有可能的关键信息）
print("\n前 3 行完整数据：")
print(df.head(3))

# 3. 自动筛选可能的关键训练字段（根据常见关键词匹配）
# 常见关键信息关键词：image、state、action、observation、prompt、task 等
key_keywords = ["image", "state", "action", "observation", "prompt", "task"]
auto_key_fields = [col for col in df.columns if any(kw in col.lower() for kw in key_keywords)]

# 筛选并展示自动识别的关键字段
if auto_key_fields:
    filtered_df = df[auto_key_fields]
    print("\n自动识别的关键训练字段：")
    print(filtered_df.columns.tolist())
    print("\n关键字段前 3 行数据：")
    print(filtered_df.head(3))
else:
    print("\n未识别到明显的关键训练字段")