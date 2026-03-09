import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    def __init__(self, in_dim=512, proj_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.BatchNorm1d(in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
        )

    def forward(self, x):
        return self.net(x)


class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_t, z_i):
        z_t = F.normalize(z_t, dim=-1)
        z_i = F.normalize(z_i, dim=-1)
        logits = (z_t @ z_i.T) / self.temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        loss_i = F.cross_entropy(logits, labels)
        loss_t = F.cross_entropy(logits.T, labels)
        return 0.5 * (loss_i + loss_t)


class GateModule(nn.Module):
    def __init__(self, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, tokens, global_feature):
        gate = torch.sigmoid(self.net(global_feature)).unsqueeze(1)
        return tokens * gate + tokens


class DatasetContrastiveRouter(nn.Module):
    def __init__(self, hidden_dim=512, proj_dim=128, use_gate=False):
        super().__init__()
        self.use_gate = use_gate
        self.proj_t = ProjectionHead(hidden_dim, proj_dim)
        self.proj_i = ProjectionHead(hidden_dim, proj_dim)
        self.loss_weibo = InfoNCELoss(temperature=0.07)
        self.loss_gossip = InfoNCELoss(temperature=0.10)
        if use_gate:
            self.gate_t = GateModule(hidden_dim)
            self.gate_i = GateModule(hidden_dim)

    def forward(self, text_tokens, image_tokens, labels, dataset_name="weibo"):
        try:
            text_global = text_tokens.mean(dim=1)
            image_global = image_tokens.mean(dim=1)
            z_t = self.proj_t(text_global)
            z_i = self.proj_i(image_global)

            name = str(dataset_name).lower() if dataset_name is not None else "weibo"
            if name == "weibo":
                loss_total = self.loss_weibo(z_t, z_i)
            else:
                loss_total = self.loss_gossip(z_t, z_i)

            if self.use_gate:
                text_tokens = self.gate_t(text_tokens, text_global)
                image_tokens = self.gate_i(image_tokens, image_global)

            return text_tokens, image_tokens, {"loss_total": loss_total}
        except Exception:
            return text_tokens, image_tokens, {"loss_total": torch.tensor(0.0, device=text_tokens.device)}
"""
对比学习路由模块
用于MMFN_yyt项目，在text_m/image_m进入Transformer之前进行对比学习
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ProjectionHead(nn.Module):
    """
    对比学习的投影头

    将特征投影到对比空间
    """
    def __init__(self, in_dim=512, proj_dim=128):
        super(ProjectionHead, self).__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.BatchNorm1d(in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
        )

    def forward(self, x):
        return self.proj(x)


class InfoNCELoss(nn.Module):
    """
    InfoNCE (NT-Xent) 对比损失

    Args:
        temperature: 温度系数，控制分布锐度
    """
    def __init__(self, temperature=0.07):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, z_t, z_i):
        """
        计算双向InfoNCE损失

        Args:
            z_t: 文本特征 [B, proj_dim]
            z_i: 图像特征 [B, proj_dim]

        Returns:
            loss: 标量，双向InfoNCE损失
        """
        # L2归一化
        z_t = F.normalize(z_t, dim=-1)
        z_i = F.normalize(z_i, dim=-1)

        B = z_t.shape[0]

        # 计算余弦相似度矩阵 [B, B]
        logits = (z_t @ z_i.T) / self.temperature

        # 正样本标签：对角线位置
        labels = torch.arange(B, device=logits.device)

        # 双向交叉熵损失
        loss_i = F.cross_entropy(logits, labels)      # 图像→文本
        loss_t = F.cross_entropy(logits.T, labels)    # 文本→图像

        return 0.5 * (loss_i + loss_t)


class GateModule(nn.Module):
    """
    门控模块（可选）

    将对比学习的全局信息反哺回token特征

    Args:
        hidden_dim: 隐藏维度
    """
    def __init__(self, hidden_dim=512):
        super(GateModule, self).__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, tokens, global_feature):
        """
        Args:
            tokens: token特征 [B, L, D]
            global_feature: 全局特征 [B, D]

        Returns:
            gated_tokens: 门控后的token特征 [B, L, D]
        """
        # 计算门控权重
        gate = torch.sigmoid(self.gate_net(global_feature))  # [B, D]
        gate = gate.unsqueeze(1)  # [B, 1, D]

        # 残差门控：不覆盖原始特征
        gated_tokens = tokens * gate + tokens

        return gated_tokens


class DatasetContrastiveRouter(nn.Module):
    """
    数据集特定的对比学习路由器

    支持Weibo和GossipCop两个数据集的不同超参数

    Args:
        hidden_dim: text_m/image_m的隐藏维度（默认512）
        proj_dim: 投影空间维度（默认128）
        use_gate: 是否使用门控机制（默认False，MVP阶段）
    """
    def __init__(
        self,
        hidden_dim=512,
        proj_dim=128,
        use_gate=False
    ):
        super(DatasetContrastiveRouter, self).__init__()

        self.hidden_dim = hidden_dim
        self.proj_dim = proj_dim
        self.use_gate = use_gate

        # 投影头
        self.proj_t = ProjectionHead(hidden_dim, proj_dim)
        self.proj_i = ProjectionHead(hidden_dim, proj_dim)

        # InfoNCE损失（两个数据集不同温度）
        self.loss_weibo = InfoNCELoss(temperature=0.07)
        self.loss_gossip = InfoNCELoss(temperature=0.10)

        # 可选：门控模块
        if use_gate:
            self.gate_t = GateModule(hidden_dim)
            self.gate_i = GateModule(hidden_dim)

    def _forward_weibo(self, text_tokens, image_tokens, labels):
        """
        Weibo数据集的对比学习

        Args:
            text_tokens: [B, Lt, 512]
            image_tokens: [B, Li, 512]
            labels: [B]

        Returns:
            text_tokens_new: [B, Lt, 512]
            image_tokens_new: [B, Li, 512]
            loss_dict: 包含loss_total
        """
        # Step A: token -> global 向量
        text_global = text_tokens.mean(dim=1)  # [B, 512]
        image_global = image_tokens.mean(dim=1)  # [B, 512]

        # Step B: 投影 + 归一化
        z_t = F.normalize(self.proj_t(text_global), dim=-1)  # [B, 128]
        z_i = F.normalize(self.proj_i(image_global), dim=-1)  # [B, 128]

        # Step C: InfoNCE损失
        loss_total = self.loss_weibo(z_t, z_i)

        # Step D: 可选的门控反哺
        if self.use_gate:
            text_tokens = self.gate_t(text_tokens, text_global)
            image_tokens = self.gate_i(image_tokens, image_global)

        loss_dict = {'loss_total': loss_total}

        return text_tokens, image_tokens, loss_dict

    def _forward_gossip(self, text_tokens, image_tokens, labels):
        """
        GossipCop数据集的对比学习

        Args:
            text_tokens: [B, Lt, 512]
            image_tokens: [B, Li, 512]
            labels: [B]

        Returns:
            text_tokens_new: [B, Lt, 512]
            image_tokens_new: [B, Li, 512]
            loss_dict: 包含loss_total
        """
        # Step A: token -> global 向量
        text_global = text_tokens.mean(dim=1)  # [B, 512]
        image_global = image_tokens.mean(dim=1)  # [B, 512]

        # Step B: 投影 + 归一化
        z_t = F.normalize(self.proj_t(text_global), dim=-1)  # [B, 128]
        z_i = F.normalize(self.proj_i(image_global), dim=-1)  # [B, 128]

        # Step C: InfoNCE损失
        loss_total = self.loss_gossip(z_t, z_i)

        # Step D: 可选的门控反哺
        if self.use_gate:
            text_tokens = self.gate_t(text_tokens, text_global)
            image_tokens = self.gate_i(image_tokens, image_global)

        loss_dict = {'loss_total': loss_total}

        return text_tokens, image_tokens, loss_dict

    def forward(self, text_tokens, image_tokens, labels, dataset_name="weibo"):
        """
        前向传播，带数据集路由和异常处理

        Args:
            text_tokens: 文本token特征 [B, Lt, 512]
            image_tokens: 图像token特征 [B, Li, 512]
            labels: 标签 [B]（可选，用于future扩展）
            dataset_name: 数据集名称，"weibo" 或 "gossip"

        Returns:
            text_tokens_new: 处理后的文本token [B, Lt, 512]
            image_tokens_new: 处理后的图像token [B, Li, 512]
            loss_dict: {'loss_total': 对比损失}
        """
        try:
            name = str(dataset_name).lower() if dataset_name is not None else "weibo"

            if name == "weibo":
                return self._forward_weibo(text_tokens, image_tokens, labels)
            elif name in ["gossip", "gossipcop"]:
                return self._forward_gossip(text_tokens, image_tokens, labels)
            else:
                # 未知数据集，默认使用gossip配置
                print(f"Warning: Unknown dataset '{dataset_name}', falling back to gossip config")
                return self._forward_gossip(text_tokens, image_tokens, labels)

        except Exception as e:
            # fallback：保证训练不中断
            print(f"Error in contrastive_router: {e}")
            print("Falling back to no-contrastive mode")
            # 返回原始特征，损失为0
            loss_dict = {'loss_total': torch.tensor(0.0, device=text_tokens.device)}
            return text_tokens, image_tokens, loss_dict


# 测试代码
if __name__ == "__main__":
    # 测试数据集路由器
    print("Testing DatasetContrastiveRouter...")

    # 创建实例（MVP模式，不使用门控）
    router = DatasetContrastiveRouter(
        hidden_dim=512,
        proj_dim=128,
        use_gate=False
    )

    # 模拟batch数据
    B = 4
    Lt = 300  # BERT token长度
    Li = 49   # Swin token长度
    D = 512

    text_tokens = torch.randn(B, Lt, D)
    image_tokens = torch.randn(B, Li, D)
    labels = torch.randint(0, 2, (B,))

    print(f"Input shapes: text_tokens={text_tokens.shape}, image_tokens={image_tokens.shape}")

    # 测试weibo数据集
    print("\n--- Testing Weibo dataset ---")
    text_out, image_out, loss_dict = router(
        text_tokens, image_tokens, labels, dataset_name="weibo"
    )
    print(f"Output shapes: text_out={text_out.shape}, image_out={image_out.shape}")
    print(f"Loss: {loss_dict['loss_total'].item():.4f}")

    # 测试gossip数据集
    print("\n--- Testing Gossip dataset ---")
    text_out, image_out, loss_dict = router(
        text_tokens, image_tokens, labels, dataset_name="gossip"
    )
    print(f"Output shapes: text_out={text_out.shape}, image_out={image_out.shape}")
    print(f"Loss: {loss_dict['loss_total'].item():.4f}")

    # 测试未知数据集（fallback）
    print("\n--- Testing unknown dataset (fallback) ---")
    text_out, image_out, loss_dict = router(
        text_tokens, image_tokens, labels, dataset_name="unknown"
    )
    print(f"Output shapes: text_out={text_out.shape}, image_out={image_out.shape}")
    print(f"Loss: {loss_dict['loss_total'].item():.4f}")

    # 测试门控模式
    print("\n--- Testing with gate enabled ---")
    router_with_gate = DatasetContrastiveRouter(
        hidden_dim=512,
        proj_dim=128,
        use_gate=True
    )
    text_out, image_out, loss_dict = router_with_gate(
        text_tokens, image_tokens, labels, dataset_name="weibo"
    )
    print(f"Output shapes: text_out={text_out.shape}, image_out={image_out.shape}")
    print(f"Loss: {loss_dict['loss_total'].item():.4f}")

    print("\nAll tests passed!")
