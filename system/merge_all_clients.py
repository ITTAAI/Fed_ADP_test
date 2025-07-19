#!/usr/bin/env python3
# merge_all_clients.py

import os
import numpy as np

def merge_clients_in_dir(folder_path, prefix, num_clients=10):
    """
    在 folder_path 下查找 prefix0_.npz ... prefix{num_clients-1}_.npz，
    将它们的 'x','y' concat 到一起，保存为 prefix{num_clients}_.npz。
    """
    xs, ys = [], []
    for k in range(num_clients):
        fn = f"{prefix}{k}_.npz"
        path = os.path.join(folder_path, fn)
        if not os.path.isfile(path):
            print(f"[WARN] 找不到 {path}，跳过")
            continue
        with np.load(path, allow_pickle=True) as arc:
            # 之前保存时用 np.savez(..., data={'x':X,'y':Y})
            data_dict = arc["data"].item()
        xs.append(data_dict["x"])
        ys.append(data_dict["y"])
    if not xs:
        print(f"[WARN] 在 {folder_path} 中未找到任何 {prefix}0~{prefix}{num_clients-1} 文件")
        return
    X = np.concatenate(xs, axis=0)
    Y = np.concatenate(ys, axis=0)
    out_fn = f"{prefix}{num_clients}_.npz"
    out_path = os.path.join(folder_path, out_fn)
    np.savez(out_path, data={"x": X, "y": Y})
    print(f"[OK ] 合并并保存 {out_path}，X shape={X.shape}, Y shape={Y.shape}")

if __name__ == "__main__":
    base_dir = "../dataset"  # 根目录，里面是 0.5, 0.7, 0.8, 1, 10 ...
    client_sets = ["cifar-10-dp", "cifar-10-normal", "cifar-10-shadow"]
    splits = ["train", "test"]
    # 对每个 ε 目录
    for eps in os.listdir(base_dir):
        eps_path = os.path.join(base_dir, eps)
        if not os.path.isdir(eps_path):
            continue
        print(f"\n=== 处理 ε={eps} ===")
        # 对每个数据集类型
        for cs in client_sets:
            for sp in splits:
                folder = os.path.join(eps_path, cs, sp)
                if not os.path.isdir(folder):
                    print(f"[WARN] 子目录不存在：{folder}")
                    continue
                print(f"-- 合并 {cs}/{sp} 客户端 0~9 → 10")
                merge_clients_in_dir(folder, prefix=sp)
