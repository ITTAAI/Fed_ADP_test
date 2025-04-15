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
        data_all = DataLoader(ConcatDataset([data_in, data_out]), batch_size=BATCH_SIZE, shuffle=False)
        data_tps = DataLoader(data_in, batch_size=BATCH_SIZE, shuffle=False)
        data_fps = DataLoader(data_out, batch_size=BATCH_SIZE, shuffle=False)

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
def plot_attack_results_per_client(results_by_part, part_names):
    """
    每个 client 单独绘图。
    :param results_by_part: List of list，每一部分模型的所有 client 结果列表
    :param part_names: 与 results_by_part 对应的模型部分名称列表
    """
    for part_idx, (results, part_name) in enumerate(zip(results_by_part, part_names)):
        for client_idx, client_result in enumerate(results):
            attack_acc = client_result['attack_acc']
            TPS_acc = client_result['TPS_acc']
            FPS_error = client_result['FPS_error']

            plt.figure(figsize=(6, 4))
            x_axis = ['Attack Acc', 'TPS Acc', 'FPS Error']
            y_values = [attack_acc, TPS_acc, FPS_error]
            markers = ['o', 's', '^']
            linestyles = ['-', '--', '-.']

            for i in range(3):
                plt.plot([i], [y_values[i]],
                         marker=markers[i],
                         linestyle=linestyles[i],
                         color='black',
                         label=x_axis[i])

            plt.xticks(range(3), x_axis)
            plt.ylim(0.0, 1.05)
            plt.ylabel("Accuracy / Error")
            plt.title(f"{part_name} - Client {client_idx}")
            plt.grid(True)

            # 图例
            line_handles = [Line2D([0], [0], color='black', lw=2, linestyle=linestyles[i], marker=markers[i])
                            for i in range(3)]
            plt.legend(line_handles, ['Attack Accuracy', 'TPS Accuracy', 'FPS Error'], loc='best')

            plt.tight_layout()
            plt.show()

