# ============================
# run_mia_pipeline.py
# ============================

import torch
import copy
from model import FedAvgCNN, LocalModel
from train_attack_model import train_attack_model
from whitebox_mia_pipeline import GradientMIA, whitebox_membership_inference_attack_pipeline, plot_attack_results_per_client

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 1
EPOCHS = 100
LR = 1e-3
NUM_CLASSES = 10
NUM_CLIENTS = 20

# 构建目标模型结构（全复制）
def get_fresh_model():
    base = FedAvgCNN(in_features=3, num_classes=10, dim=1600).to(DEVICE)
    head = copy.deepcopy(base.fc)
    base.fc = torch.nn.Identity()
    return LocalModel(base, head)

# 构造 shadow 模型文件名
shadow_client_files = [
    f"shadow_model/results_cifar-10-shadow_client{i}_600_0.0050.pt"
    for i in range(NUM_CLIENTS)
]





# Step 1: 一次性训练所有攻击模型
# for target_label in range(NUM_CLASSES):
#     print(f"==== Training Attack Model for Label: {target_label} ====")
#     shadow_model = get_fresh_model()
#     train_attack_model(
#         shadow_model,
#         shadow_client_files,
#         target_label,
#         batch_size=BATCH_SIZE,
#         device=DEVICE,
#         epochs=EPOCHS,
#         lr=LR,
#         num_clients=NUM_CLIENTS,
#     )

# 构造 target 模型文件名（多个版本用于不同结构对比）
target_model_names = [""]
target_client_files = {
    name: [
        f"dp_model/results_cifar-10-normal_client{i}_600_0.0050{name}.pt"
        for i in range(NUM_CLIENTS)
    ] for name in target_model_names
}

# 存储不同模型部分结果

# Step 2: 执行白盒攻击评估
all_results_by_part = [[] for _ in range(len(target_model_names))]  # 每个 part 一组结果

for target_label in range(NUM_CLASSES):
    print(f"==== Evaluating Attack on Target Label: {target_label} ====")
    attack_model = GradientMIA().to(DEVICE)
    attack_model.load_state_dict(torch.load(f"attack_model{target_label}.pth"))
    attack_model.eval()

    for idx, name in enumerate(target_model_names):
        target_model = get_fresh_model()
        client_results = whitebox_membership_inference_attack_pipeline(
            target_client_files[name],
            target_model,
            target_label,
            BATCH_SIZE,
            DEVICE,
            attack_model,
            num_clients=NUM_CLIENTS,
        )
        all_results_by_part[idx].append(client_results)  # 按 label 存 client 列表

plot_attack_results_per_client(all_results_by_part, target_model_names)
