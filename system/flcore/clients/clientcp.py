import copy
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from utils.data_utils import read_client_data
import os
import matplotlib.pyplot as plt


class clientCP:
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        self.model = copy.deepcopy(args.model)
        self.dataset = args.dataset
        self.device = args.device
        self.id = id
        self.dp = args.difference_privacy
        self.num_classes = args.num_classes
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.local_steps = args.local_steps
        self.noise = {}
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.round = 0
        self.param_diff = {}
        self.inital_pra = {}
        if self.dp:
            result_dir =f"{self.dataset}_gradient_log_dp"
        else:
            result_dir = f"{self.dataset}_gradient_log"
        os.makedirs(result_dir, exist_ok=True)
        filename = f"gradient_log_client{self.id}_{args.dataset}_{args.global_rounds}_{args.local_learning_rate:.4f}.txt"

        self.filepath = os.path.join(result_dir, filename)
        self.lamda = args.lamda

        in_dim = list(args.model.head.parameters())[0].shape[1]
        self.context = torch.rand(1, in_dim).to(self.device)
        self.opt = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

    def compute_norm(self, param_dict):
        total_norm = 0.0
        for name, tensor in param_dict.items():
            total_norm += tensor.norm().item() ** 2
        return total_norm ** 0.5

    def get_module_diff_norm(self, visualize=False,iftest=False) -> float:
        """
        计算模型参数与初始参数之间差异（参数变化量）的 L2 范数并返回。
        如果 visualize 为 True，则根据参数差异生成热度图并保存，文件名中包含 self.id 和 self.round 信息。

        其中：
          - 参数差异通过 (param - self.initial_params[name]).detach() 计算；
          - head_diff 保存所有名称以 "head" 开头的参数差异；
          - feat_diff 保存所有名称以 "feature_extractor" 开头的参数差异。
        """
        # 计算各参数与初始参数之间的差异



        total_norm_sq = 0.0

        # 遍历所有参数差异并累加其 L2 范数的平方
        for name, diff in self.param_diff.items():
            diff_norm = diff.norm(2)
            total_norm_sq += diff_norm.item() ** 2

            if visualize:
                # 根据参数差异数据的维度生成对应的热度图
                if diff.dim() == 1:
                    # 1D 张量转换为 1 行的二维数组
                    heatmap_data = diff.unsqueeze(0).abs().cpu().numpy()
                elif diff.dim() == 2:
                    heatmap_data = diff.abs().cpu().numpy()
                elif diff.dim() == 4:
                    # 针对卷积层的 4D 权重 (out_channels, in_channels, kH, kW)
                    # 对 in_channels 维度求平均，得到 (out_channels, kH, kW)
                    heatmap_data = diff.abs().mean(dim=1).cpu().numpy()
                    # 如果有多个滤波器，默认取第一个显示
                    if heatmap_data.shape[0] > 1:
                        heatmap_data = heatmap_data[0]
                else:
                    # 其他情况：将前几维展平为二维数据
                    heatmap_data = diff.view(diff.size(0), -1).abs().cpu().numpy()

                plt.figure()
                plt.imshow(heatmap_data, cmap='hot')
                plt.title(f"Parameter Diff Heatmap for {name}")
                plt.colorbar(label='|Parameter Diff|')

                # 构造保存热度图的路径
                if self.dp:
                    base_folder = f"{self.dataset}_gradient_heatmap_dp"
                else:
                    base_folder = f"{self.dataset}_gradient_heatmap"
                if not os.path.exists(base_folder):
                    os.makedirs(base_folder)
                subfolder = os.path.join(base_folder, f"{self.id}_gradient_heatmap")
                if not os.path.exists(subfolder):
                    os.makedirs(subfolder)
                subsubfolder = os.path.join(subfolder, f"{name.replace('.', '_')}_gradient_heatmap")
                if not os.path.exists(subsubfolder):
                    os.makedirs(subsubfolder)
                if iftest:
                    file_name = f"param_diff_heatmap_{self.id}_{self.round}_test.png"
                else:
                    file_name = f"param_diff_heatmap_{self.id}_{self.round}_train.png"
                save_path = os.path.join(subsubfolder, file_name)
                plt.savefig(save_path)
                plt.close()
                print(f"已保存 {name} 的差异热度图，文件名：{save_path}")

        return total_norm_sq ** 0.5
    def get_module_grad_norm(self,model) -> float:
        """
        计算给定模块的所有参数梯度的 L2 范数并返回。
        如果某个参数没有梯度（grad=None），则跳过。
        """
        total_norm_sq = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm_sq += param_norm.item() ** 2
        return total_norm_sq ** 0.5
    def load_train_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        train_data = read_client_data(self.dataset, self.id, is_train=True)
        return DataLoader(train_data, batch_size, drop_last=True, shuffle=False)

    def load_test_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        test_data = read_client_data(self.dataset, self.id, is_train=False)
        return DataLoader(test_data, batch_size, drop_last=True, shuffle=False)

    def set_parameters(self, feature_extractor):

        for new_param, old_param in zip(feature_extractor.parameters(), self.model.feature_extractor.parameters()):
            old_param.data = new_param.data.clone()

    def set_head_g(self, head):
        headw_ps = []
        for name, mat in self.model.model.head.named_parameters():
            if 'weight' in name:
                headw_ps.append(mat.data)
        headw_p = headw_ps[-1]
        for mat in headw_ps[-2::-1]:
            headw_p = torch.matmul(headw_p, mat)
        headw_p.detach_()
        self.context = torch.sum(headw_p, dim=0, keepdim=True)

    def save_con_items(self, items, tag='', item_path=None):
        self.save_item(self.pm_train, 'pm_train' + '_' + tag, item_path)
        self.save_item(self.pm_test, 'pm_test' + '_' + tag, item_path)
        for idx, it in enumerate(items):
            self.save_item(it, 'item_' + str(idx) + '_' + tag, item_path)

    def test_metrics_before(self):
        testloader = self.load_test_data()
        self.model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []

        with torch.no_grad():
            for x, y in testloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                y_prob.append(F.softmax(output).detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')


        return test_acc, test_num, auc

    def test_metrics_after(self):
        testloader = self.load_test_data()
        self.model.train()
        self.pm_test = []

        with torch.enable_grad():
            for x, y in testloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                self.opt.zero_grad()
                loss.backward()
                break
            self.param_diff = {}
            for name, p in self.model.named_parameters():
                if p.grad is not None:
                    grad = p.grad.data
                    # 根据梯度下降更新规则，参数更新值为 -learning_rate * grad
                    self.param_diff[name] = -self.learning_rate * grad
            if self.round % 50 == 0 and self.round > 1:
                self.get_module_diff_norm(iftest=True, visualize=True)
            grad_norm_head = self.get_module_grad_norm(self.model.head)  # global model
            grad_norm_feat = self.get_module_grad_norm(self.model.feature_extractor)


            record_dict = {
                "round": self.round,
                "grad_norm_head": grad_norm_head,
                "grad_norm_feat": grad_norm_feat,
            }
            with open(self.filepath+"test", "a") as f:
                f.write(str(record_dict) + "\n")


        return

    def train_cs_model(self):

        testloader = self.load_test_data()

        trainloader = self.load_train_data()
        self.model.train()

        for _ in range(self.local_steps):
            self.pm_train = []
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                break

        self.inital_pra = {name: param.clone().detach() for name, param in self.model.named_parameters()}
        self.inital_pra_dp = {name: param.clone().detach() for name, param in
                              self.model.feature_extractor.named_parameters()}
        for _ in range(self.local_steps):
            self.pm_train = []
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

        with torch.enable_grad():
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                self.opt.zero_grad()
                loss.backward()
                break
            self.param_diff = {}
            for name, p in self.model.named_parameters():
                if p.grad is not None:
                    grad = p.grad.data
                    # 根据梯度下降更新规则，参数更新值为 -learning_rate * grad
                    self.param_diff[name] = -self.learning_rate * grad
            # if self.round % 50 == 0 and self.round > 1:
            #     self.get_module_diff_norm(iftest=True, visualize=True)
            grad_norm_head = self.get_module_grad_norm(self.model.head)  # global model
            grad_norm_feat = self.get_module_grad_norm(self.model.feature_extractor)

            record_dict = {
                "round": self.round,
                "grad_norm_head": grad_norm_head,
                "grad_norm_feat": grad_norm_feat,
            }
            with open(self.filepath + "before", "a") as f:
                f.write(str(record_dict) + "\n")
        # test 数据集测试
        with torch.enable_grad():
            for x, y in testloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                self.opt.zero_grad()
                loss.backward()
                break
            self.param_diff_test = {}
            for name, p in self.model.named_parameters():
                if p.grad is not None:
                    grad = p.grad.data
                    # 根据梯度下降更新规则，参数更新值为 -learning_rate * grad
                    self.param_diff_test[name] = -self.learning_rate * grad

        # Clip and add noise for DP if enabled
        clip_value=0.008
        epsilon = 0.8
        delta = 1e-5
        if self.round % 50 == 0 and self.round>1 :
            self.param_diff = {}
            for name, param in self.model.named_parameters():
                self.param_diff[name] = (param - self.inital_pra[name]).detach()

            self.get_module_diff_norm(visualize=True)
        if self.dp:
            param_diff = {}
            modules = {'conv1': self.model.feature_extractor.conv1,
                       'conv2': self.model.feature_extractor.conv2,
                       'fc1': self.model.feature_extractor.fc1}

            for module_name, module in modules.items():
                for name, param in module.named_parameters():
                    full_name = f"{module_name}.{name}"  # 确保参数名称唯一
                    param_diff[full_name] = (param - self.inital_pra_dp[full_name]).detach()

            for full_name, diff in param_diff.items():
                # 1) 计算 25% 分位数作为裁剪阈值
                threshold = torch.quantile(diff.abs().view(-1), 0.99)
                threshold_test= torch.quantile(self.param_diff_test["feature_extractor."+full_name].abs().view(-1), 0.01)
                clip_value= max(0.03,torch.quantile(diff.abs().view(-1), 0.75))
                clip_value=torch.quantile(diff.abs().view(-1), 0.75)
                mask = (diff.abs() <= threshold) & (self.param_diff_test["feature_extractor." + full_name].abs() >= threshold_test)

                # print(f"Layer {full_name}: {mask.sum().item()} / {mask.numel()} parameters masked.")
                masked_diff = diff[mask]

                norm = torch.norm(diff)
                if norm > clip_value:
                    diff = diff / norm * clip_value


                # -- (b) 加噪 --
                noise_std = 20*clip_value*torch.sqrt(
                    torch.tensor(2.0) * torch.log(torch.tensor(1.25 / delta)))/(len(trainloader)*epsilon)
                noise = torch.normal(mean=0, std=noise_std, size=masked_diff.shape).to(diff.device)
                masked_diff = masked_diff + noise

                # 4) 写回原来的 diff
                diff[mask] = masked_diff
                param_diff[full_name] = diff

                # 记录噪声
                self.noise[full_name] = torch.zeros_like(diff)
                self.noise[full_name][mask] = noise

            for name, param in self.model.feature_extractor.named_parameters():
                param.data = self.inital_pra_dp[name] + param_diff[name]

        with torch.enable_grad():
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                self.opt.zero_grad()
                loss.backward()
                break
            self.param_diff = {}
            for name, p in self.model.named_parameters():
                if p.grad is not None:
                    grad = p.grad.data
                    # 根据梯度下降更新规则，参数更新值为 -learning_rate * grad
                    self.param_diff[name] = -self.learning_rate * grad
            # if self.round % 50 == 0 and self.round > 1:
            #     self.get_module_diff_norm(iftest=True, visualize=True)
            grad_norm_head = self.get_module_grad_norm(self.model.head)  # global model
            grad_norm_feat = self.get_module_grad_norm(self.model.feature_extractor)

            record_dict = {
                "round": self.round,
                "grad_norm_head": grad_norm_head,
                "grad_norm_feat": grad_norm_feat,
            }
            with open(self.filepath + "after", "a") as f:
                f.write(str(record_dict) + "\n")

        # Save model at 100th round
        if round == 500:
            import os
            save_dir = "pretrain"
            os.makedirs(save_dir, exist_ok=True)  # Create folder if it doesn't exist
            filename = f"results_{args.dataset}_client{self.id}_{args.global_rounds}_{args.local_learning_rate:.4f}_{args.difference_privacy_layer}_{args.difference_privacy_layer2}.pt"
            save_path = os.path.join(save_dir, filename)
            torch.save(self.model.state_dict(), save_path)
            print(f"Model saved to {save_path}")


def MMD(x, y, kernel, device='cpu'):
    """Emprical maximum mean discrepancy. The lower the result
       the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = rx.t() + rx - 2. * xx  # Used for A in (1)
    dyy = ry.t() + ry - 2. * yy  # Used for B in (1)
    dxy = rx.t() + ry - 2. * zz  # Used for C in (1)

    XX, YY, XY = (torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device))

    if kernel == "multiscale":

        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a ** 2 * (a ** 2 + dxx) ** -1
            YY += a ** 2 * (a ** 2 + dyy) ** -1
            XY += a ** 2 * (a ** 2 + dxy) ** -1

    if kernel == "rbf":

        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5 * dxx / a)
            YY += torch.exp(-0.5 * dyy / a)
            XY += torch.exp(-0.5 * dxy / a)

    return torch.mean(XX + YY - 2. * XY)


