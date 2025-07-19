import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# CIFAR-10 常用均值和标准差（3通道）
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

# 设置 Dirichlet 分布的 alpha 值
alpha = 10
# 正常训练的客户端数量
num_clients_normal = 10
# 影子（用于成员攻击）训练的客户端数量
num_clients_shadow = 10


def load_cifar10(data_dir):
    """
    加载 CIFAR-10 数据集，返回：
    train_data, train_labels, test_data, test_labels
    其中：
      - train_data.shape = (50000, 32, 32, 3)
      - train_labels.shape = (50000,)
      - test_data.shape  = (10000, 32, 32, 3)
      - test_labels.shape = (10000,)
    """
    # Create directory if it doesn't exist, to prevent errors with torchvision dataset creation
    os.makedirs(data_dir, exist_ok=True)
    # Using ToTensor just to trigger download and standard format, but we use .data
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)

    train_data = train_dataset.data  # shape: (50000, 32, 32, 3)
    train_labels = np.array(train_dataset.targets)  # shape: (50000,)

    test_data = test_dataset.data  # shape: (10000, 32, 32, 3)
    test_labels = np.array(test_dataset.targets)  # shape: (10000,)

    return train_data, train_labels, test_data, test_labels


def normalize_images_cifar10(images):
    """
    对 CIFAR-10 图像进行归一化：
    - 将图像转换到 [0,1]
    - 对每个通道分别减去均值再除以标准差
    images: numpy 数组，形状为 (N, H, W, 3)
    """
    images = images.astype(np.float32) / 255.0
    mean = np.array(CIFAR10_MEAN, dtype=np.float32).reshape(1, 1, 1, 3)
    std = np.array(CIFAR10_STD, dtype=np.float32).reshape(1, 1, 1, 3)
    images = (images - mean) / std
    return images


def gen_class_proportions_for_clients(num_classes, num_clients, alpha):
    """
    使用 Dirichlet 为每个类别生成对各个客户端的分配比例。
    返回形状为 (num_classes, num_clients) 的数组 proportions[c, k] 表示
    类 c 分配到客户端 k 的比例（所有 k 之和为 1）。
    """
    proportions = np.zeros((num_classes, num_clients), dtype=np.float32)
    for c in range(num_classes):
        p = np.random.dirichlet([alpha] * num_clients)
        proportions[c] = p
    return proportions


def split_data_with_given_proportions(data, labels, proportions):
    """
    根据已有的 proportions 数组 (num_classes, num_clients),
    把 data/labels 分配到各个客户端。

    data: shape=(N, H, W, C)
    labels: shape=(N,)
    proportions: shape=(num_classes, num_clients)
    返回: client_data (字典)
        client_data[k] = {'data': [...], 'labels': [...]}
    """
    num_clients = proportions.shape[1]
    num_classes_total = proportions.shape[0]
    client_data = {i: {'data': [], 'labels': []} for i in range(num_clients)}
    unique_classes = np.unique(labels)

    for c in unique_classes:
        if c >= num_classes_total:
            print(
                f"Warning: Class label {c} is out of bounds for proportions array (max index {num_classes_total - 1}). Skipping this class.")
            continue

        idx_c = np.where(labels == c)[0]
        # Data for class c is already part of a shuffled dataset.
        # If further shuffling of class c samples before distribution is needed, uncomment:
        # np.random.shuffle(idx_c)
        n_c = len(idx_c)

        if n_c == 0:
            continue

        current_proportions_for_class_c = proportions[c]
        counts = np.floor(current_proportions_for_class_c * n_c).astype(int)
        sum_counts = np.sum(counts)
        remainder = n_c - sum_counts

        if remainder > 0:
            # Distribute remainder randomly among clients, allowing a client to receive more than one remainder item
            inc_indices = np.random.choice(num_clients, remainder, replace=True)
            for inc_k in inc_indices:
                counts[inc_k] += 1

        current_sum_counts = np.sum(counts)
        diff = n_c - current_sum_counts
        if diff != 0:
            if diff > 0:  # need to add more
                adjust_indices = np.random.choice(num_clients, diff, replace=True)
                for idx_adjust in adjust_indices: counts[idx_adjust] += 1
            elif diff < 0:  # need to remove some
                for _ in range(int(abs(diff))):
                    # Remove from clients that have samples, could prioritize those with more
                    eligible_clients_to_reduce = np.where(counts > 0)[0]
                    if not eligible_clients_to_reduce.size: break
                    client_to_reduce = np.random.choice(eligible_clients_to_reduce)
                    counts[client_to_reduce] -= 1
        counts = np.maximum(0, counts)  # Ensure no negative counts
        final_diff_check = n_c - np.sum(counts)  # Final adjustment if any mismatch
        if final_diff_check != 0:
            counts[0] += final_diff_check

        start = 0
        for k in range(num_clients):
            num_k = counts[k]
            if num_k > 0:
                client_data[k]['data'].extend(data[idx_c[start:start + num_k]])
                client_data[k]['labels'].extend(labels[idx_c[start:start + num_k]])
                start += num_k
            if start > n_c: break
            # if start != n_c:
        # print(f"Warning: For class {c}, not all samples were distributed. Distributed {start}/{n_c}.")

    return client_data


def shuffle_client_data(client_data):
    """
    对每个客户端的数据进行随机打乱。
    """
    for client_id, content in client_data.items():
        if len(content['data']) == 0:
            continue
        data_arr = np.array(content['data'])
        labels_arr = np.array(content['labels'])

        indices = np.arange(len(data_arr))
        np.random.shuffle(indices)

        client_data[client_id]['data'] = data_arr[indices].tolist()
        client_data[client_id]['labels'] = labels_arr[indices].tolist()


def save_client_data_cifar10(client_data, output_dir, prefix):
    """
    保存CIFAR-10客户端数据到文件，数据已归一化并转换为 (N, C, H, W) 格式（C=3）。
    """
    os.makedirs(output_dir, exist_ok=True)
    for client_id, content in tqdm(client_data.items(), desc=f"Saving {prefix} CIFAR-10 client data"):
        if len(content['data']) == 0:
            # print(f"Skipping client {client_id} for {prefix} (CIFAR-10) due to no data.")
            continue

        X = np.array(content['data'], dtype=np.float32)  # shape: (N, 32, 32, 3)
        Y = np.array(content['labels'], dtype=np.int64)

        X = normalize_images_cifar10(X)  # shape: (N, 32, 32, 3)
        X = np.transpose(X, (0, 3, 1, 2))  # Convert to (N, 3, 32, 32)

        client_path = os.path.join(output_dir, f"{prefix}{client_id}_.npz")
        np.savez(client_path, data={'x': X, 'y': Y})


def save_public_data_cifar10(data, labels, output_dir, filename_prefix):
    """
    保存公共CIFAR-10数据集到文件，数据已归一化并转置为 (N, C, H, W) 格式。
    """
    os.makedirs(output_dir, exist_ok=True)
    if len(data) == 0:
        print(f"Skipping saving public CIFAR-10 data {filename_prefix} due to no data.")
        return

    X = np.array(data, dtype=np.float32)  # Shape (N, 32, 32, 3)
    Y = np.array(labels, dtype=np.int64)

    X = normalize_images_cifar10(X)  # Shape (N, 32, 32, 3)
    X = np.transpose(X, (0, 3, 1, 2))  # Shape (N, 3, 32, 32)

    file_path = os.path.join(output_dir, f"{filename_prefix}.npz")
    np.savez(file_path, data={'x': X, 'y': Y})
    print(f"Public CIFAR-10 data {filename_prefix} saved: X_shape={X.shape}, Y_shape={Y.shape} to {file_path}")


def create_per_client_attack_dataset_cifar10(
        train_clients,
        test_clients,
        output_dir="attack_data_per_client_cifar10",
        prefix="client"
):
    """
    为每个CIFAR-10客户端分别抽取「成员」(member) 与「非成员」(non-member) 攻击数据集。
    """
    os.makedirs(output_dir, exist_ok=True)

    for cid in tqdm(train_clients.keys(), desc=f"Generating CIFAR-10 attack data for {prefix}"):
        if cid not in test_clients:
            # print(f"[Client {cid}] not found in test_clients (CIFAR-10), skipping attack dataset generation for {prefix}{cid}.")
            continue

        train_data_list = train_clients[cid]['data']
        train_labels_list = train_clients[cid]['labels']
        test_data_list = test_clients[cid]['data']
        test_labels_list = test_clients[cid]['labels']

        if len(train_data_list) == 0 or len(test_data_list) == 0:
            # print(f"[Client {cid}] has no data in train or test split for {prefix} (CIFAR-10), skip attack data generation.")
            continue

        train_data_arr = np.array(train_data_list)
        train_labels_arr = np.array(train_labels_list)
        test_data_arr = np.array(test_data_list)
        test_labels_arr = np.array(test_labels_list)

        M = min(len(train_data_arr), len(test_data_arr), 500)
        if M == 0:
            # print(f"[Client {cid}] Not enough data for attack set (M=0) for {prefix} (CIFAR-10), skip.")
            continue

        idx_train = np.random.permutation(len(train_data_arr))
        idx_test = np.random.permutation(len(test_data_arr))

        member_x = train_data_arr[idx_train[:M]]
        member_y = train_labels_arr[idx_train[:M]]
        nonmember_x = test_data_arr[idx_test[:M]]
        nonmember_y = test_labels_arr[idx_test[:M]]

        member_x = normalize_images_cifar10(member_x)
        member_x = np.transpose(member_x, (0, 3, 1, 2))
        nonmember_x = normalize_images_cifar10(nonmember_x)
        nonmember_x = np.transpose(nonmember_x, (0, 3, 1, 2))

        member_path = os.path.join(output_dir, f"{prefix}_{cid}_member.npz")
        nonmember_path = os.path.join(output_dir, f"{prefix}_{cid}_nonmember.npz")
        np.savez(member_path, data={'x': member_x, 'y': member_y})
        np.savez(nonmember_path, data={'x': nonmember_x, 'y': nonmember_y})
        # print(f"[Client {cid}] CIFAR-10 Attack data saved for {prefix}: member={member_x.shape}, nonmember={nonmember_x.shape}")


if __name__ == "__main__":
    # ========== 1. 加载并打乱 CIFAR-10 数据 ==========
    data_dir_cifar = 'cifar10_data_main'
    original_train_data, original_train_labels, original_test_data, original_test_labels = load_cifar10(data_dir_cifar)
    print(f"Original CIFAR-10 loaded: train={original_train_data.shape}, test={original_test_data.shape}")

    perm_train_orig = np.random.permutation(len(original_train_data))
    original_train_data = original_train_data[perm_train_orig]
    original_train_labels = original_train_labels[perm_train_orig]

    perm_test_orig = np.random.permutation(len(original_test_data))
    original_test_data = original_test_data[perm_test_orig]
    original_test_labels = original_test_labels[perm_test_orig]

    train_size_orig = len(original_train_data)
    test_size_orig = len(original_test_data)

    # ========== 2. 单独拿出3% IID 数据作为公共数据集 ==========
    public_percentage = 0.03
    num_public_train = int(train_size_orig * public_percentage)
    num_public_test = int(test_size_orig * public_percentage)

    public_train_data = original_train_data[:num_public_train]
    public_train_labels = original_train_labels[:num_public_train]
    public_test_data = original_test_data[:num_public_test]
    public_test_labels = original_test_labels[:num_public_test]

    remaining_train_data = original_train_data[num_public_train:]
    remaining_train_labels = original_train_labels[num_public_train:]
    remaining_test_data = original_test_data[num_public_test:]
    remaining_test_labels = original_test_labels[num_public_test:]

    print(f"Public CIFAR-10 train data: {public_train_data.shape}, Public test data: {public_test_data.shape}")
    print(
        f"Remaining CIFAR-10 train data: {remaining_train_data.shape}, Remaining test data: {remaining_test_data.shape}")

    # ========== 3. 保存公共 CIFAR-10 数据集 ==========
    output_public_dir_cifar = 'public_cifar10_data_iid_5percent'
    save_public_data_cifar10(public_train_data, public_train_labels, output_public_dir_cifar, "public_train")
    save_public_data_cifar10(public_test_data, public_test_labels, output_public_dir_cifar, "public_test")
    print("[Info] Public CIFAR-10 dataset saved.")

    # ========== 4. 拆分剩余数据为 normal / shadow 两部分 ==========
    remaining_train_size = len(remaining_train_data)
    remaining_test_size = len(remaining_test_data)

    half_remaining_train = remaining_train_size // 3*2
    half_remaining_test = remaining_test_size // 3*2

    normal_train_data = remaining_train_data[:half_remaining_train]
    normal_train_labels = remaining_train_labels[:half_remaining_train]
    normal_test_data = remaining_test_data[:half_remaining_test]
    normal_test_labels = remaining_test_labels[:half_remaining_test]

    base_shadow_train_data = remaining_train_data[half_remaining_train:]
    base_shadow_train_labels = remaining_train_labels[half_remaining_train:]
    base_shadow_test_data = remaining_test_data[half_remaining_test:]
    base_shadow_test_labels = remaining_test_labels[half_remaining_test:]

    print(f"Normal CIFAR-10 train data (from remaining): {normal_train_data.shape}")
    print(f"Base shadow CIFAR-10 train data (from remaining): {base_shadow_train_data.shape}")

    # ------------------------------------------------
    # Shadow augmentation: add 10% of *original* CIFAR-10 data (this will overlap with public data)
    # ------------------------------------------------
    ten_percent_train_orig = int(train_size_orig * 0.25)
    ten_percent_test_orig = int(test_size_orig * 0.25)

    real_subset_train_data_orig = original_train_data[:ten_percent_train_orig]
    real_subset_train_labels_orig = original_train_labels[:ten_percent_train_orig]
    real_subset_test_data_orig = original_test_data[:ten_percent_test_orig]
    real_subset_test_labels_orig = original_test_labels[:ten_percent_test_orig]

    print(
        f"Real subset for CIFAR-10 shadow (from original start, overlaps with public): train={real_subset_train_data_orig.shape}, test={real_subset_test_data_orig.shape}")

    shadow_train_data = np.concatenate([base_shadow_train_data, real_subset_train_data_orig], axis=0)
    shadow_train_labels = np.concatenate([base_shadow_train_labels, real_subset_train_labels_orig], axis=0)
    shadow_test_data = np.concatenate([base_shadow_test_data, real_subset_test_data_orig], axis=0)
    shadow_test_labels = np.concatenate([base_shadow_test_labels, real_subset_test_labels_orig], axis=0)

    print(f"Final CIFAR-10 shadow train data: {shadow_train_data.shape}, labels: {shadow_train_labels.shape}")
    print(f"Final CIFAR-10 shadow test data: {shadow_test_data.shape}, labels: {shadow_test_labels.shape}")
    # ------------------------------------------------

    # ========== 5. 对 normal / shadow 数据使用 Dirichlet 划分并 Shuffle ==========
    num_classes_cifar = 10

    print("\nProcessing normal CIFAR-10 clients...")
    normal_proportions = gen_class_proportions_for_clients(num_classes_cifar, num_clients_normal, alpha)
    normal_train_clients = split_data_with_given_proportions(normal_train_data, normal_train_labels, normal_proportions)
    normal_test_clients = split_data_with_given_proportions(normal_test_data, normal_test_labels, normal_proportions)

    print("\nProcessing shadow CIFAR-10 clients...")
    shadow_proportions = gen_class_proportions_for_clients(num_classes_cifar, num_clients_shadow, alpha)
    shadow_train_clients = split_data_with_given_proportions(shadow_train_data, shadow_train_labels, shadow_proportions)
    shadow_test_clients = split_data_with_given_proportions(shadow_test_data, shadow_test_labels, shadow_proportions)

    print("\nShuffling CIFAR-10 client data...")
    shuffle_client_data(normal_train_clients)
    shuffle_client_data(normal_test_clients)
    shuffle_client_data(shadow_train_clients)
    shuffle_client_data(shadow_test_clients)

    # ========== 6. 保存分好客户端的 CIFAR-10 数据 ==========
    output_partition_dir_normal_cifar = 'partitioned_cifar10_normal_public5split'
    output_partition_dir_shadow_cifar = 'partitioned_cifar10_shadow_public5split'

    print("\nSaving normal CIFAR-10 client data...")
    save_client_data_cifar10(normal_train_clients, output_partition_dir_normal_cifar, prefix='train')
    save_client_data_cifar10(normal_test_clients, output_partition_dir_normal_cifar, prefix='test')

    print("\nSaving shadow CIFAR-10 client data...")
    save_client_data_cifar10(shadow_train_clients, output_partition_dir_shadow_cifar, prefix='train')
    save_client_data_cifar10(shadow_test_clients, output_partition_dir_shadow_cifar, prefix='test')

    print("[Info] Finished partitioning and saving normal/shadow CIFAR-10 client data.")

    # # ========== 7. 为每个客户端生成 CIFAR-10 攻击数据集 ==========
    # attack_out_dir_normal_cifar = "attack_data_per_client_cifar10_normal_public5split"
    # print(f"\nGenerating attack dataset for normal CIFAR-10 clients (output: {attack_out_dir_normal_cifar})...")
    # create_per_client_attack_dataset_cifar10(
    #     train_clients=normal_train_clients,
    #     test_clients=normal_test_clients,
    #     output_dir=attack_out_dir_normal_cifar,
    #     prefix="normal_cifar_client"  # Differentiated prefix
    # )
    #
    # attack_out_dir_shadow_cifar = "attack_data_per_client_cifar10_shadow_public5split"
    # print(f"\nGenerating attack dataset for shadow CIFAR-10 clients (output: {attack_out_dir_shadow_cifar})...")
    # create_per_client_attack_dataset_cifar10(
    #     train_clients=shadow_train_clients,
    #     test_clients=shadow_test_clients,
    #     output_dir=attack_out_dir_shadow_cifar,
    #     prefix="shadow_cifar_client"  # Differentiated prefix
    # )
    # print("[Info] Per-client CIFAR-10 attack dataset generation done.")