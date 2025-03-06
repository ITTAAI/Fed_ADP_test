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
        self.round=0
        result_dir = "gradient_log"
        os.makedirs(result_dir, exist_ok=True)
        if args.difference_privacy:
            filename = f"gradient_log_client{self.id}_{args.dataset}_{args.global_rounds}_{args.local_learning_rate:.4f}_{args.difference_privacy_layer}.txt"
        else:
            filename = f"gradient_log_client{self.id}_{args.dataset}_{args.global_rounds}_{args.local_learning_rate:.4f}.txt"
        self.filepath = os.path.join(result_dir, filename)
        self.lamda = args.lamda

        in_dim = list(args.model.head.parameters())[0].shape[1]
        self.context = torch.rand(1, in_dim).to(self.device)
        self.opt= torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)



    def compute_norm(self, param_dict):
        total_norm = 0.0
        for name, tensor in param_dict.items():
            total_norm += tensor.norm().item() ** 2
        return total_norm ** 0.5

    def get_module_grad_norm(self, model, visualize=False) -> float:
        """
        计算给定模块所有参数梯度的 L2 范数并返回。
        如果某个参数存在梯度（p.grad != None），则：
          - 累加该参数梯度的 L2 范数
          - 如果 visualize 为 True，则根据该参数梯度生成热度图并保存到指定目录中，
            文件名中包含 self.id 和 self.round 信息。
        """
        total_norm_sq = 0.0

        # 定义基础文件夹 "gradient_heatmap"
        base_folder = "gradient_heatmap"
        if not os.path.exists(base_folder):
            os.makedirs(base_folder)

        # 定义子文件夹，格式为 "self.id_gradient_heatmap"
        subfolder = os.path.join(base_folder, f"{self.id}_gradient_heatmap")
        if not os.path.exists(subfolder):
            os.makedirs(subfolder)

        for name, p in model.named_parameters():
            if p.grad is not None:
                grad = p.grad.data
                param_norm = grad.norm(2)
                total_norm_sq += param_norm.item() ** 2

                if visualize:
                    # 根据梯度数据的维度选择不同的处理方式
                    if grad.dim() == 1:
                        # 1D张量转为1行2D数组
                        heatmap_data = grad.unsqueeze(0).abs().cpu().numpy()
                    elif grad.dim() == 2:
                        heatmap_data = grad.abs().cpu().numpy()
                    elif grad.dim() == 4:
                        # 对卷积层的4D权重 (out_channels, in_channels, kH, kW)
                        # 对 in_channels 求平均，得到 (out_channels, kH, kW)
                        heatmap_data = grad.abs().mean(dim=1).cpu().numpy()
                        # 如果有多个滤波器，默认取第一个显示
                        if heatmap_data.shape[0] > 1:
                            heatmap_data = heatmap_data[0]
                    else:
                        # 其他情况：将前几维展平为2D
                        heatmap_data = grad.view(grad.size(0), -1).abs().cpu().numpy()

                    plt.figure()
                    plt.imshow(heatmap_data, cmap='hot')
                    plt.title(f"Gradient Heatmap for {name}")
                    plt.colorbar(label='|Gradient|')

                    # 构造文件名：gradient_heatmap_{self.id}_{self.round}_{name}.png
                    file_name = f"gradient_heatmap_{self.id}_{self.round}_{name.replace('.', '_')}.png"
                    save_path = os.path.join(subfolder, file_name)
                    plt.savefig(save_path)
                    plt.close()
                    print(f"已保存 {name} 的热度图，文件名：{save_path}")

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


    def test_metrics(self):
        testloader = self.load_test_data()
        self.model.train()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []
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
            grad_norm_head = self.get_module_grad_norm(self.model.head)  # global model
            grad_norm_feat = self.get_module_grad_norm(self.model.feature_extractor)

            # 接下来你可以将这些范数记录到日志、保存到文件、
            # 或者可视化，比如 print/log/wandb/tensorboard 等
            record_dict = {
                "round": round,
                "grad_norm_head": grad_norm_head,
                "grad_norm_feat": grad_norm_feat,
            }
            with open(self.filepath+"test", "a") as f:
                f.write(str(record_dict) + "\n")
        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')

        
        return test_acc, test_num, auc

                
    def train_cs_model(self):
        initial_params = {name: param.clone().detach() for name, param in self.model.named_parameters()}

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
                output= self.model(x)
                loss = self.loss(output, y)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
            param_diff = {}
            for name, param in self.model.named_parameters():
                param_diff[name] = (param - initial_params[name]).detach()
            head_diff = {name: diff for name, diff in param_diff.items() if name.startswith("head")}
            feat_diff = {name: diff for name, diff in param_diff.items() if name.startswith("feature_extractor")}

            # 4. 计算各部分的梯度范数（你可以定义一个 compute_norm 函数来计算参数范数）
            grad_norm_head = self.compute_norm(head_diff)
            grad_norm_feat = self.compute_norm(feat_diff)

            # 5. 将信息记录到字典中
            record_dict = {
                "round": self.round,
                "grad_norm_head": grad_norm_head,
                "grad_norm_feat": grad_norm_feat,
            }



            with open(self.filepath, "a") as f:
                f.write(str(record_dict) + "\n")
        # Clip and add noise for DP if enabled
        clip_value = 0.02
        epsilon = 0.5
        delta = 1e-5
        if self.round % 50 == 0:
            self.get_module_grad_norm(self.model, visualize=True)
        if self.dp:
            param_diff = {}
            diff_norms = []

            for name, param in self.model.head.named_parameters():
                param_diff[name] = (param - initial_params[name]).detach()
                diff_norm = param_diff[name].norm(p=2).item()
                diff_norms.append(diff_norm)


            for name, diff in param_diff.items():
                norm = torch.norm(diff)
                if norm > clip_value:
                    diff = diff / norm * clip_value

                noise_std = (clip_value / epsilon) * torch.sqrt(
                    torch.tensor(2.0) * torch.log(torch.tensor(1.25 / delta)))
                noise = torch.normal(mean=0, std=noise_std, size=diff.shape).to(diff.device)
                self.noise[name] = noise
                param_diff[name] += noise




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
    
    dxx = rx.t() + rx - 2. * xx # Used for A in (1)
    dyy = ry.t() + ry - 2. * yy # Used for B in (1)
    dxy = rx.t() + ry - 2. * zz # Used for C in (1)
    
    XX, YY, XY = (torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device))
    
    if kernel == "multiscale":
        
        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a**2 * (a**2 + dxx)**-1
            YY += a**2 * (a**2 + dyy)**-1
            XY += a**2 * (a**2 + dxy)**-1
            
    if kernel == "rbf":
      
        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5*dxx/a)
            YY += torch.exp(-0.5*dyy/a)
            XY += torch.exp(-0.5*dxy/a)
      
    return torch.mean(XX + YY - 2. * XY)


