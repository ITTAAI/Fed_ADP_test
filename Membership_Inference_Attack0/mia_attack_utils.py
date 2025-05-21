# ============================
# mia_attack_utils.py
# ============================

import torch
import torch.nn.functional as F
import numpy as np

# 获取模型输出 + 梯度信息
def get_model_outputs_labels_and_grads(model, dataloader, device):
    model.eval()
    outputs_list, labels_list = [], []
    head_grads, feat_grads = {}, {}

    for name, param in model.head.named_parameters():
        head_grads[name] = []
    for name, param in model.feature_extractor.named_parameters():
        feat_grads[name] = []

    loss_fn = torch.nn.CrossEntropyLoss()

    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        model.zero_grad()
        logits = model(batch_x)
        loss = loss_fn(logits, batch_y)
        loss.backward()

        softmax = torch.nn.functional.softmax(logits.detach().cpu(), dim=1)
        outputs_list.append(softmax.numpy())

        labels_list.append(batch_y.detach().cpu().numpy())

        for name, param in model.head.named_parameters():
            grad = param.grad.detach().cpu().numpy().copy() if param.grad is not None else None
            head_grads[name].append(grad)

        for name, param in model.feature_extractor.named_parameters():
            grad = param.grad.detach().cpu().numpy().copy() if param.grad is not None else None
            feat_grads[name].append(grad)

    outputs = np.concatenate(outputs_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    outputs = np.concatenate(outputs_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)

    # 🔍 添加 DEBUG 检查
    expected_len = len(outputs)
    for key, grad_list in head_grads.items():
        if len(grad_list) != expected_len:
            print(f"[ERROR] Gradient count mismatch in head layer '{key}': "
                  f"expected {expected_len}, got {len(grad_list)}")

    for key, grad_list in feat_grads.items():
        if len(grad_list) != expected_len:
            print(f"[ERROR] Gradient count mismatch in feature layer '{key}': "
                  f"expected {expected_len}, got {len(grad_list)}")

    if len(labels) != expected_len:
        print(f"[ERROR] Label count mismatch: expected {expected_len}, got {len(labels)}")

    # 🛡️ 可选：如果你想防止数据异常时继续运行，可以抛出异常
    assert len(labels) == expected_len, "Mismatch between outputs and labels"
    for grad_dict in [head_grads, feat_grads]:
        for k, v in grad_dict.items():
            assert len(v) == expected_len, f"Mismatch in gradient count for {k}"

    return outputs, labels, head_grads, feat_grads

    return outputs, labels, head_grads, feat_grads

# 白盒攻击输入预处理（reshape）
def prepare_attack_model_inputs(outputs, head_grads, feat_grads):
    softmax_out = torch.tensor(F.softmax(torch.tensor(outputs), dim=1), dtype=torch.float32)

    def stack_grads(grad_list):
        grads = [torch.tensor(g, dtype=torch.float32) for g in grad_list if g is not None]
        return torch.stack(grads, dim=0)

    grad_conv1 = stack_grads(feat_grads['conv1.0.weight'])
    grad_conv2 = stack_grads(feat_grads['conv2.0.weight'])
    grad_fc1   = stack_grads(feat_grads['fc1.0.weight'])
    grad_fc = stack_grads(head_grads['weight'])  # ✅ 正确的 key


    def reshape_tensor(tensor):
        B = tensor.shape[0]
        flat = tensor.view(B, -1)
        side = int(flat.shape[1] ** 0.5)
        if side * side != flat.shape[1]:
            pad = side * side - flat.shape[1]
            flat = F.pad(flat, (0, pad), value=0)
        return flat.view(B, 1, side, side)

    return reshape_tensor(grad_conv1), reshape_tensor(grad_conv2), reshape_tensor(grad_fc1), reshape_tensor(grad_fc), softmax_out