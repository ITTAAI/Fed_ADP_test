import os
import numpy as np
import matplotlib.pyplot as plt
import torch

#######################
# 1) 读取数据的部分
#######################
def read_data(dataset, idx, is_train=True):
    """
    根据 is_train=True/False，分别读取 train 或 test 的 .npz 文件。
    npz 文件结构假设为 {'data': {'x': X, 'y': Y}}
    """
    # 假设目录结构: ../dataset/{dataset}/train/train{idx}_.npz
    #            : ../dataset/{dataset}/test/test{idx}_.npz
    sub_dir = 'train' if is_train else 'test'
    data_dir = os.path.join('../dataset', dataset, sub_dir)
    file_name = f"{sub_dir}{idx}_.npz"
    file_path = os.path.join(data_dir, file_name)

    with open(file_path, 'rb') as f:
        content = np.load(f, allow_pickle=True)['data'].tolist()
    return content

def get_label_distribution(dataset, client_id, num_classes, is_train=True):
    """
    读取某个客户端的 train/test 数据，返回一个长度为 num_classes 的标签计数数组。
    """
    data_dict = read_data(dataset, idx=client_id, is_train=is_train)
    labels = data_dict['y']
    label_count = np.zeros(num_classes, dtype=int)
    for lbl in labels:
        label_count[lbl] += 1
    return label_count


#######################
# 2) 可视化气泡图部分
#######################
def plot_bubble_distribution(distribution_matrix,
                             num_clients,
                             num_classes,
                             title="Distribution",
                             is_validation=False,
                             beta=0.5,
                             color='red',
                             alpha=0.6):
    """
    绘制气泡图:
      - distribution_matrix: shape=(num_clients, num_classes)
      - x轴: client_id
      - y轴: class_id
      - 气泡大小: distribution_matrix[client_id, class_id]
      - title: 图表标题
      - color, alpha: 氣泡颜色与透明度
    """

    fig, ax = plt.subplots(figsize=(8, 8))

    # 准备散点坐标 & 尺寸
    x_vals = []
    y_vals = []
    bubble_sizes = []

    for client_id in range(num_clients):
        for class_id in range(num_classes):
            count = distribution_matrix[client_id, class_id]
            x_vals.append(client_id)
            y_vals.append(class_id)
            bubble_sizes.append(count)

    # 氣泡大小通常需要一个缩放系数，否则可能太大或太小
    scale_factor = 2  # 可根据自己数据量级调节
    bubble_sizes = [s * scale_factor for s in bubble_sizes]

    # 使用 scatter 绘制气泡
    sc = ax.scatter(
        x_vals,               # x坐标: 客户端ID
        y_vals,               # y坐标: 类别ID
        s=bubble_sizes,       # 点的大小
        c=color,              # 颜色
        alpha=alpha,          # 透明度
        edgecolors='face'     # 氣泡边框颜色与填充一致
    )

    # 设置网格
    ax.set_xlim(-0.5, num_clients - 0.5)
    ax.set_ylim(-0.5, num_classes - 0.5)
    ax.set_xticks(range(num_clients))
    ax.set_yticks(range(num_classes))
    ax.grid(True, which='major', axis='both', linestyle='-', alpha=0.5)

    # 设置标题 (示例: Validation Set Distribution (β=0.5))
    if is_validation:
        plt.title(f"Validation Set Distribution (β = {beta})")
    else:
        plt.title(title)

    # 标签
    ax.set_xlabel("Client IDs")
    ax.set_ylabel("Class IDs")

    # 可选: 如果想要加 colorbar 或者气泡大小图例，可以考虑在这里自定义。
    # 例如，使用伪 color map:
    # cbar = plt.colorbar(sc, ax=ax)
    # cbar.set_label('Sample Count')

    plt.tight_layout()
    plt.show()


#######################
# 3) 主程序示例
#######################
if __name__ == "__main__":
    dataset_name = "cifar-10-normal"  # 数据集名称
    num_clients = 21      # 客户端数
    num_classes = 10     # 示例: 这里演示200类(你的图看起来像200行),
                          # 如果是CIFAR-10则 num_classes=10。
                          # 或者你想绘制 200, 256 之类都行。

    # 是否画 "验证集" 分布 (仅示意做标题区分)
    is_validation = True
    beta_val = 0.5

    # 先创建一个 distribution_matrix (num_clients, num_classes)
    distribution_matrix = np.zeros((num_clients, num_classes), dtype=int)

    # 遍历每个客户端, 读取其标签分布
    # 这里以 test 数据(= validation) 为例进行可视化
    for cid in range(num_clients):
        dist_count = get_label_distribution(dataset_name,
                                            client_id=cid,
                                            num_classes=num_classes,
                                            is_train=True)  # False -> test data
        distribution_matrix[cid, :] = dist_count

    # 调用气泡图函数
    plot_bubble_distribution(distribution_matrix=distribution_matrix,
                             num_clients=num_clients,
                             num_classes=num_classes,
                             title="Validation Set Distribution",
                             is_validation=is_validation,
                             beta=beta_val,
                             color='red',   # 红色气泡
                             alpha=0.6)
    for cid in range(num_clients):
        dist_count = get_label_distribution(dataset_name,
                                            client_id=cid,
                                            num_classes=num_classes,
                                            is_train=False)  # False -> test data
        distribution_matrix[cid, :] = dist_count

    # 调用气泡图函数
    plot_bubble_distribution(distribution_matrix=distribution_matrix,
                             num_clients=num_clients,
                             num_classes=num_classes,
                             title="Validation Set Distribution",
                             is_validation=is_validation,
                             beta=beta_val,
                             color='red',   # 红色气泡
                             alpha=0.6)