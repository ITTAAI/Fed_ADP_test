# ============================
# data_utils.py
# ============================

import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, Subset


def read_data(is_train=True, is_shadow=True, num_clients=5,alpha=1):
    data_list = []
    for i in range(num_clients):
        if is_shadow:
            file_name = f"dataset/{alpha}/{'cifar-10-shadow/train/train' if is_train else 'cifar-10-shadow/test/test'}{i}_.npz"
        else:
            file_name = f"dataset/{alpha}/{'cifar-10-normal/train/train' if is_train else 'cifar-10-normal/test/test'}{i}_.npz"
        if not os.path.exists(file_name):
            raise FileNotFoundError(f"File {file_name} not found.")
        with open(file_name, 'rb') as f:
            single_data = np.load(f, allow_pickle=True)['data'].tolist()
        data_list.append(single_data)
    return data_list


def read_client_data(is_train=True, is_shadow=True, num_clients=5,alpha=1):
    data_list = read_data(is_train=is_train, is_shadow=is_shadow, num_clients=num_clients,alpha=alpha)
    client_datasets = []
    for client_data in data_list:
        X = torch.Tensor(client_data['x']).float()
        y = torch.Tensor(client_data['y']).long()
        client_datasets.append(TensorDataset(X, y))
    return client_datasets


def filter_by_label(dataset, target_label):
    indices = [i for i, (_, label) in enumerate(dataset) if label == target_label]
    return Subset(dataset, indices)
