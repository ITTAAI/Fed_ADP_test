# =========================
# whitebox_mia_pipeline.py
# =========================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Subset, ConcatDataset
from model import GradientMIA
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mia_attack_utils import get_model_outputs_labels_and_grads, prepare_attack_model_inputs
from data_utils import read_client_data, filter_by_label


# ============================
# 2. 攻击训练接口
# ============================
def train_attack_model(attack_model, dataloader, epochs, lr, device):
    optimizer = torch.optim.Adam(attack_model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()
    for epoch in range(epochs):
        attack_model.train()
        total_loss, correct = 0, 0
        for g1, g2, g3, g4, softmax, labels in dataloader:
            g1, g2, g3, g4, softmax = g1.to(device), g2.to(device), g3.to(device), g4.to(device), softmax.to(device)
            labels = labels.float().unsqueeze(1).to(device)
            preds = attack_model(g1, g2, g3, g4, softmax)
            loss = loss_fn(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
            correct += ((preds > 0.5).int() == labels.int()).sum().item()
        acc = correct / len(dataloader.dataset)
        avg_loss = total_loss / len(dataloader.dataset)
        print(f"[Epoch {epoch + 1}/{epochs}] Loss: {avg_loss:.4f}, Acc: {acc:.4f}")

# ============================
# 3. 白盒攻击评估接口
# ============================
def whitebox_membership_inference_attack_pipeline(client_files, target_model, target_label, BATCH_SIZE, DEVICE, attack_model,
                                                  num_clients=5):
    """
    白盒 MIA 攻击评估流程：利用目标模型在训练集与 holdout 集上的梯度 + softmax 构造攻击样本，送入攻击模型判断是否为成员。
    """

    train_datasets = read_client_data(is_train=True, is_shadow=False, num_clients=num_clients)
    holdout_datasets = read_client_data(is_train=False, is_shadow=False, num_clients=num_clients)

    train_datasets_filtered = [filter_by_label(ds, target_label) for ds in train_datasets]
    holdout_datasets_filtered = [filter_by_label(ds, target_label) for ds in holdout_datasets]

    train_datasets_filtered = [
        Subset(train_dataset, list(range(len(holdout_dataset))))
        for train_dataset, holdout_dataset in zip(train_datasets_filtered, holdout_datasets_filtered)
    ]

    train_loaders = [DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False) for ds in train_datasets_filtered]
    holdout_loaders = [DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False) for ds in holdout_datasets_filtered]

    client_results = []

    for model_file, train_loader, holdout_loader in zip(client_files, train_loaders, holdout_loaders):
        if len(train_loader.dataset) == 0 or len(holdout_loader.dataset) == 0:
            print(f"[DEBUG] Empty dataset for client {model_file}. Skipping.")
            continue
        target_model.load_state_dict(torch.load(model_file, map_location=DEVICE))
        target_model.to(DEVICE)

        # ========== 成员样本 ==========
        outputs, _, head_grads, feat_grads = get_model_outputs_labels_and_grads(target_model, train_loader, DEVICE)
        g1, g2, g3, g4, softmax = prepare_attack_model_inputs(outputs, head_grads, feat_grads)
        label_in = torch.ones(g1.size(0)).long()
        data_in = TensorDataset(g1, g2, g3, g4, softmax, label_in)

        # ========== 非成员样本 ==========
        outputs, _, head_grads, feat_grads = get_model_outputs_labels_and_grads(target_model, holdout_loader, DEVICE)
        g1, g2, g3, g4, softmax = prepare_attack_model_inputs(outputs, head_grads, feat_grads)
        label_out = torch.zeros(g1.size(0)).long()
        data_out = TensorDataset(g1, g2, g3, g4, softmax, label_out)

        # ========== 构造 DataLoader ==========
        data_all = DataLoader(ConcatDataset([data_in, data_out]), batch_size=BATCH_SIZE, shuffle=True)
        data_tps = DataLoader(data_in, batch_size=BATCH_SIZE, shuffle=True)
        data_fps = DataLoader(data_out, batch_size=BATCH_SIZE, shuffle=True)

        # ========== 模型评估 ==========
        def eval_loader(loader):
            attack_model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for g1, g2, g3, g4, softmax, labels in loader:
                    g1, g2, g3, g4, softmax = g1.to(DEVICE), g2.to(DEVICE), g3.to(DEVICE), g4.to(DEVICE), softmax.to(DEVICE)
                    labels = labels.to(DEVICE).float().unsqueeze(1)
                    preds = attack_model(g1, g2, g3, g4, softmax)
                    preds_label = (preds > 0.5).float()
                    all_preds.append(preds_label.cpu())
                    all_labels.append(labels.cpu())
            return (torch.cat(all_preds) == torch.cat(all_labels)).float().mean().item()

        attack_acc = eval_loader(data_all)
        TPS_acc = eval_loader(data_tps)
        FPS_acc = eval_loader(data_fps)
        FPS_error = 1 - FPS_acc

        client_results.append({
            'client': model_file,
            'attack_acc': attack_acc,
            'TPS_acc': TPS_acc,
            'FPS_acc': FPS_acc,
            'FPS_error': FPS_error,
        })

        print(f"[Client: {model_file}] Attack Acc: {attack_acc:.4f} | TPS: {TPS_acc:.4f} | FPS Error: {FPS_error:.4f}")

    return client_results


# ============================
# 4. 可视化模块
# ============================
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import matplotlib
matplotlib.use('Agg')                 # 服务器 / 无 GUI
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

def plot_attack_results_per_client(results_by_part, part_names,
                                   save_root='view'):
    """
    results_by_part: list (part) -> list (client) -> list (label) -> dict
    """
    save_root = Path(save_root)
    save_root.mkdir(exist_ok=True)

    for part_idx, (part_results, part_name) in enumerate(zip(results_by_part, part_names)):
        for client_idx, label_results in enumerate(part_results):

            # --------- 收集 10 个 label 的数值 ---------
            attack_vals = [r['attack_acc'] for r in label_results]
            tps_vals    = [r['TPS_acc']    for r in label_results]
            fps_vals    = [r['FPS_error']  for r in label_results]
            labels_x    = list(range(len(label_results)))      # 0…9

            # --------- 一张图绘三条折线 ---------
            fig, ax = plt.subplots(figsize=(5, 3.5))
            ax.plot(labels_x, attack_vals, marker='o', linestyle='-',
                    label='Attack Acc')
            ax.plot(labels_x, tps_vals,    marker='s', linestyle='--',
                    label='TPS Acc')
            ax.plot(labels_x, fps_vals,    marker='^', linestyle='-.',
                    label='FPS Error')

            ax.set_xticks(labels_x)
            ax.set_xlabel("Target Label")
            ax.set_ylim(0, 1.05)
            ax.set_ylabel("Accuracy / Error")
            ax.set_title(f"{part_name}  |  Client {client_idx}")
            ax.grid(True)
            ax.legend(loc='best')
            fig.tight_layout()

            out = save_root / f"{part_name}_c{client_idx}.png"
            fig.savefig(out, dpi=200)
            plt.close(fig)

