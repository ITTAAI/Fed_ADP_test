# =========================
# whitebox_mia_pipeline.py
# =========================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Subset, ConcatDataset
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
def whitebox_membership_inference_attack_pipeline(
    client_files,
    target_model,
    target_label,
    BATCH_SIZE,
    DEVICE,
    attack_model,
    num_clients=5,
    alpha=1,
):
    """
    白盒 MIA 攻击评估流程：利用目标模型在训练集与 holdout 集上的梯度 + softmax 构造攻击样本，
    并返回目标模型在这两个子集上的分类准确率，以及对它们的攻击成功率。
    """

    # 读入原始数据（未过滤）
    train_datasets = read_client_data(is_train=True, is_shadow=False, num_clients=num_clients,alpha=alpha)
    holdout_datasets = read_client_data(is_train=False, is_shadow=False, num_clients=num_clients,alpha=alpha)

    # 仅保留指定 label 的子集
    train_datasets_filtered = [filter_by_label(ds, target_label) for ds in train_datasets]
    holdout_datasets_filtered = [filter_by_label(ds, target_label) for ds in holdout_datasets]

    # 确保 train 与 holdout 样本数一致（可选）
    train_datasets_filtered = [
        Subset(train_ds, list(range(min(len(train_ds), len(holdout_ds)))))
        for train_ds, holdout_ds in zip(train_datasets_filtered, holdout_datasets_filtered)
    ]
    holdout_datasets_filtered = [
        Subset(holdout_ds, list(range(min(len(train_ds), len(holdout_ds)))))
        for train_ds, holdout_ds in zip(train_datasets_filtered, holdout_datasets_filtered)
    ]

    train_loaders = [DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False) for ds in train_datasets_filtered]
    holdout_loaders = [DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False) for ds in holdout_datasets_filtered]

    def eval_classification_acc(model, loader):
        """返回 model 在 loader 上的分类准确率"""
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits = model(x)
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        return correct / total if total > 0 else 0.0

    client_results = []

    for model_file, train_loader, holdout_loader in zip(client_files, train_loaders, holdout_loaders):
        if len(train_loader.dataset) == 0 or len(holdout_loader.dataset) == 0:
            print(f"[DEBUG] Empty dataset for client {model_file}. Skipping.")
            continue

        # 加载目标模型
        target_model.load_state_dict(torch.load(model_file, map_location=DEVICE))
        target_model.to(DEVICE)

        # —— 1) 先评估目标模型的分类准确率 ——
        train_acc = eval_classification_acc(target_model, train_loader)
        holdout_acc = eval_classification_acc(target_model, holdout_loader)

        # —— 2) 构造成员样本 ——
        outputs, _, head_grads, feat_grads = get_model_outputs_labels_and_grads(
            target_model, train_loader, DEVICE
        )
        g1, g2, g3, g4, sm_in = prepare_attack_model_inputs(outputs, head_grads, feat_grads)
        data_in = TensorDataset(g1, g2, g3, g4, sm_in, torch.ones(g1.size(0)).long())

        # —— 3) 构造非成员样本 ——
        outputs, _, head_grads, feat_grads = get_model_outputs_labels_and_grads(
            target_model, holdout_loader, DEVICE
        )
        g1, g2, g3, g4, sm_out = prepare_attack_model_inputs(outputs, head_grads, feat_grads)
        data_out = TensorDataset(g1, g2, g3, g4, sm_out, torch.zeros(g1.size(0)).long())

        # —— 4) 构造攻击模型输入的 DataLoaders ——
        data_all = DataLoader(ConcatDataset([data_in, data_out]), batch_size=BATCH_SIZE, shuffle=True)
        data_tps = DataLoader(data_in, batch_size=BATCH_SIZE, shuffle=True)
        data_fps = DataLoader(data_out, batch_size=BATCH_SIZE, shuffle=True)

        # —— 5) 评估攻击模型 ——
        def eval_attack_acc(loader):
            attack_model.eval()
            preds_list, labels_list = [], []
            with torch.no_grad():
                for a, b, c, d, sm, lbl in loader:
                    a,b,c,d,sm = [t.to(DEVICE) for t in (a,b,c,d,sm)]
                    lbl = lbl.to(DEVICE).float().unsqueeze(1)
                    out = attack_model(a, b, c, d, sm)
                    preds_list.append((out > 0.5).float().cpu())
                    labels_list.append(lbl.cpu())
            p = torch.cat(preds_list); l = torch.cat(labels_list)
            return (p == l).float().mean().item()

        attack_acc = eval_attack_acc(data_all)
        tps_acc    = eval_attack_acc(data_tps)
        fps_acc    = eval_attack_acc(data_fps)
        fps_err    = 1 - fps_acc

        # —— 6) 将所有指标附加到结果中 ——
        client_results.append({
            'client': model_file,
            'train_acc':      (train_acc+holdout_acc)/2,
            'holdout_acc':    holdout_acc,
            'attack_acc':     attack_acc,
            'TPS_acc':        tps_acc,
            'FPS_acc':        fps_acc,
            'FPS_error':      fps_err,
        })

        print(
            f"[Client: {model_file}] "
            f"Train Acc: {train_acc:.4f} | Holdout Acc: {holdout_acc:.4f} | "
            f"Attack Acc: {attack_acc:.4f} | TPS: {tps_acc:.4f} | FPS Err: {fps_err:.4f}"
        )

    return client_results


# ============================
# 4. 可视化模块
# ============================


def plot_attack_results_per_client(results_by_part, part_names,
                                   save_root='view'):
    import matplotlib
    matplotlib.use('Agg')  # 服务器 / 无 GUI
    import matplotlib.pyplot as plt
    from pathlib import Path
    """
    results_by_part: list (part) -> list (client) -> list (label) -> dict
    每个 dict 现在包含:
      - 'train_acc'
      - 'holdout_acc'
      - 'attack_acc'
      - 'TPS_acc'
      - 'FPS_error'
    """
    save_root = Path(save_root)
    save_root.mkdir(exist_ok=True)

    for part_idx, (part_results, part_name) in enumerate(zip(results_by_part, part_names)):
        for client_idx, label_results in enumerate(part_results):

            # --------- 收集 10 个 label 的数值 ---------
            train_vals  = [r['train_acc']     for r in label_results]
            attack_vals = [r['attack_acc']    for r in label_results]
            tps_vals    = [r['TPS_acc']       for r in label_results]
            fps_vals    = [r['FPS_error']     for r in label_results]
            labels_x    = list(range(len(label_results)))  # 0…9

            # --------- 打印 train_acc 到控制台 ---------
            print(f"[Plot] {part_name} | Client {client_idx} | Train Acc per label: {train_vals}")

            # --------- 一张图绘四条折线 ---------
            fig, ax = plt.subplots(figsize=(5, 3.5))
            ax.plot(labels_x, train_vals,  marker='d', linestyle=':',  label='Train Acc')
            ax.plot(labels_x, attack_vals, marker='o', linestyle='-',  label='Attack Acc')
            ax.plot(labels_x, tps_vals,    marker='s', linestyle='--', label='TPS Acc')
            ax.plot(labels_x, fps_vals,    marker='^', linestyle='-.', label='FPS Error')

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

def plot_attack_results_avg_per_client(results_by_part, part_names):
    """
    Plot average per-label metrics across clients, per part.

    results_by_part: list (part) -> list (client) -> list (label) -> dict
    Each dict contains:
        - 'train_acc'
        - 'holdout_acc'
        - 'attack_acc'
        - 'TPS_acc'
        - 'FPS_error'
    """
    import matplotlib.pyplot as plt

    for part_idx, (part_results, part_name) in enumerate(zip(results_by_part, part_names)):
        num_clients = len(part_results)
        num_labels = len(part_results[0])

        # Accumulate values
        train_vals_all  = np.zeros(num_labels)
        attack_vals_all = np.zeros(num_labels)
        tps_vals_all    = np.zeros(num_labels)
        fps_vals_all    = np.zeros(num_labels)

        for client_results in part_results:
            for label_idx, metrics in enumerate(client_results):
                train_vals_all[label_idx]  += metrics['train_acc']
                attack_vals_all[label_idx] += metrics['attack_acc']
                tps_vals_all[label_idx]    += metrics['TPS_acc']
                fps_vals_all[label_idx]    += metrics['FPS_error']

        # Compute averages
        train_avg  = train_vals_all  / num_clients
        attack_avg = attack_vals_all / num_clients
        tps_avg    = tps_vals_all    / num_clients
        fps_avg    = fps_vals_all    / num_clients

        labels_x = list(range(num_labels))

        # Plot
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(labels_x, train_avg,  marker='d', linestyle=':',  label='Avg Train Acc')
        ax.plot(labels_x, attack_avg, marker='o', linestyle='-',  label='Avg Attack Acc')
        ax.plot(labels_x, tps_avg,    marker='s', linestyle='--', label='Avg TPS Acc')
        ax.plot(labels_x, fps_avg,    marker='^', linestyle='-.', label='Avg FPS Error')

        ax.set_xticks(labels_x)
        ax.set_xlabel("Target Label")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Accuracy / Error")
        ax.set_title(f"{part_name}  |  Avg Across Clients")
        ax.grid(True)
        ax.legend(loc='best')
        fig.tight_layout()
        plt.show()