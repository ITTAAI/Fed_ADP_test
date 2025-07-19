import torch

from torch.utils.data import DataLoader, TensorDataset, Subset, ConcatDataset
from data_utils import read_client_data, filter_by_label

def evaluate_all_clients_accuracy(client_model_files, get_model_fn, device, batch_size, alpha, num_clients):
    """
    评估每个客户端模型在 holdout 数据集上的准确率。

    Args:
        client_model_files: list[str]，每个客户端模型文件路径
        get_model_fn: 返回初始化模型的函数
        device: torch.device
        batch_size: batch size
        alpha: 数据划分控制参数（用于读取 holdout 数据）
        num_clients: 客户端数量

    Returns:
        list[float]，每个客户端模型的准确率
    """
    # 加载 holdout 数据
    holdout_datasets = read_client_data(is_train=False, is_shadow=False, num_clients=num_clients, alpha=alpha)

    accuracies = []

    for i, model_path in enumerate(client_model_files):
        model = get_model_fn().to(device)
        model.load_state_dict(torch.load(model_path))
        model.eval()

        dataset_i = holdout_datasets[i]
        x = torch.stack([sample[0] for sample in dataset_i])
        y = torch.tensor([sample[1] for sample in dataset_i])
        loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)

        # 准确率评估
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb).argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)

        acc = correct / total if total > 0 else 0.0
        print(f"Client {i} holdout accuracy: {acc:.4f}")
        accuracies.append(acc)

    return accuracies
