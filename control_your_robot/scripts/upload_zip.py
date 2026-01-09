import os

# =========================================================
# ⚠️ 必须放在最最前面，在 import cv2 和 numpy 之前执行！
# 强制底层库只使用单线程，防止多进程死锁
# =========================================================
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import glob
import cv2
import h5py
import numpy as np
import time
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# --- 子进程初始化函数 ---
def worker_init():
    # 再次强制当前子进程只用一个线程
    try:
        cv2.setNumThreads(0)
    except:
        pass

# --- 辅助函数 ---
def hdf5_groups_to_dict(hdf5_path):
    result = {}
    with h5py.File(hdf5_path, 'r') as f:
        def visit_handler(name, obj):
            if isinstance(obj, h5py.Dataset):
                parts = name.split('/')
                curr = result
                for part in parts[:-1]:
                    if part not in curr: curr[part] = {}
                    curr = curr[part]
                curr[parts[-1]] = obj[()]
        f.visititems(visit_handler)
    return result

def images_decoding(encoded_data, valid_len=None):
    imgs = []
    for data in encoded_data:
        if valid_len is not None:
            data = data[:valid_len]
        if isinstance(data, bytes):
            data = data.rstrip(b'\0')
        nparr = np.frombuffer(data, np.uint8)
        try:
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None: raise ValueError("Decode return None")
        except Exception:
            # 容错：给一个黑色图片
            img = np.zeros((480, 640, 3), dtype=np.uint8)
        imgs.append(img)
    return imgs

def images_encoding(imgs):
    encode_data = []
    max_len = 0
    for img in imgs:
        success, encoded_image = cv2.imencode(".jpg", img)
        jpeg_data = encoded_image.tobytes()
        encode_data.append(jpeg_data)
        max_len = max(max_len, len(jpeg_data))
    padded_data = [d.ljust(max_len, b"\0") for d in encode_data]
    return padded_data, max_len

# --- 核心处理任务 ---
def process_single_file(args_pack):
    hdf5_file, output_path, do_encode = args_pack
    dataset_name = os.path.basename(hdf5_file)
    target_file = os.path.join(output_path, dataset_name)

    try:
        # 读数据
        ep = hdf5_groups_to_dict(hdf5_file)
        
        with h5py.File(target_file, "w") as root:
            if do_encode:
                # 压缩模式逻辑 (根据你的需求简写了，如需完整请参考之前版本)
                pass 
            else:
                # ================= 解压逻辑 =================
                # 1. 尝试定位 observations 组
                source = ep.get("observations", ep)

                # 2. 处理图像
                img_mapping = {
                    "cam_high": "cam_head",
                    "cam_left_wrist": "cam_left_wrist",
                    "cam_right_wrist": "cam_right_wrist"
                }
                
                # 图像写入
                for src_k, target_grp in img_mapping.items():
                    # 容错查找：先找 cam_high，找不到找 cam_head ...
                    data = source.get(src_k)
                    if data is None and src_k == "cam_high": data = source.get("cam_head")
                    
                    if data is not None:
                        # 解码
                        decoded = images_decoding(data)
                        decoded_np = np.array(decoded)
                        
                        grp = root.create_group(target_grp)
                        grp.create_dataset("color", data=decoded_np)

                # 3. 处理机械臂状态 (Joints, qpos 等)
                for arm in ["left_arm", "right_arm"]:
                    arm_data = source.get(arm, ep.get(arm)) # 双重查找
                    if arm_data:
                        arm_grp = root.create_group(arm)
                        for k, v in arm_data.items():
                            # 只存非图像的数组
                            if isinstance(v, np.ndarray) and v.ndim < 3:
                                arm_grp.create_dataset(k, data=v)

        return None # 成功

    except Exception as e:
        if os.path.exists(target_file): os.remove(target_file)
        return f"Error {dataset_name}: {str(e)}"

# --- 主程序 ---
def main(args):
    hdf5_paths = args.input
    if args.encode:
        output_path = hdf5_paths + "_zip"
    else:
        output_path = hdf5_paths.replace("_zip", "")
    
    if output_path == hdf5_paths:
        output_path = hdf5_paths + "_unzipped"

    os.makedirs(output_path, exist_ok=True)
    hdf5_files = glob.glob(f"{hdf5_paths}/*.hdf5")
    
    # ----------------------------------------------------
    # ⚠️ 安全模式：先用 8 个核心跑，确保不死锁
    # ----------------------------------------------------
    max_workers = 16
    
    print(f"检测到 CPU 核心数: {os.cpu_count()}")
    print(f"输入: {hdf5_paths}")
    print(f"输出: {output_path}")
    print(f">>> 安全模式启动: 使用 {max_workers} 个进程 (防止OpenCV死锁)")
    print(">>> 正在初始化进程池... (如卡住请等待10秒)")

    task_args = [(f, output_path, args.encode) for f in hdf5_files]
    errors = []

    # 这里的 initializer 非常关键
    with ProcessPoolExecutor(max_workers=max_workers, initializer=worker_init) as executor:
        results = list(tqdm(executor.map(process_single_file, task_args), total=len(hdf5_files)))
        
        for res in results:
            if res is not None:
                errors.append(res)
    
    print("-" * 30)
    if errors:
        print(f"❌ 完成，但有 {len(errors)} 个文件失败！前3个错误:")
        for e in errors[:3]: print(e)
    else:
        print(f"✅ 全部成功！输出在: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str)
    parser.add_argument("--encode", action="store_true")
    args = parser.parse_args()
    main(args)