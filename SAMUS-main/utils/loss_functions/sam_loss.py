from pyexpat import model
import torch
import torch.nn as nn
from torch.nn.modules.loss import CrossEntropyLoss
import torch.nn.functional as F


class Classification_FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2, num_classes=3, size_average=True):
        super(Classification_FocalLoss, self).__init__()
        self.size_average = size_average

        # 如果 alpha 是列表，表示每个类的 alpha 值
        if isinstance(alpha, list):
            assert len(alpha) == num_classes
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha < 1
            # 默认背景类 alpha 较小，其他类 alpha 较大
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] = alpha  # 背景类 alpha
            self.alpha[1:] = 1 - alpha  # 其他类 alpha

        self.gamma = gamma
        self.num_classes = num_classes

    def forward(self, preds, labels):
        """
        计算 Focal Loss
        :param preds: 预测值，形状 [batch_size, num_classes]
        :param labels: 真实标签，形状 [batch_size]
        :return: 计算出的损失值
        """

        self.alpha = self.alpha.to(preds.device)  # 将 alpha 移到相同的设备上
        # 计算 softmax 并取对数，获得每个类别的预测概率
        preds_logsoftmax = F.log_softmax(preds, dim=1)
        preds_softmax = torch.exp(preds_logsoftmax)

        # 获取每个样本预测类别的概率和对应的 log 概率
        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))  # 选择对应标签的预测概率
        preds_logsoftmax = preds_logsoftmax.gather(1, labels.view(-1, 1))  # 选择对应标签的 log 概率

        # 选择对应标签的 alpha 值
        alpha = self.alpha.gather(0, labels.view(-1))

        # 计算 Focal Loss
        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma), preds_logsoftmax)
        loss = torch.mul(alpha, loss.squeeze())  # 根据 alpha 加权损失

        # 返回平均或总和损失
        if self.size_average:
            return loss.mean()  # 默认返回均值损失
        else:
            return loss.sum()  # 返回总和损失
class Focal_loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes=3, size_average=True):
        super(Focal_loss, self).__init__()
        self.size_average = size_average
        if isinstance(alpha, list):
            assert len(alpha) == num_classes
            print(f'Focal loss alpha={alpha}, will assign alpha values for each class')
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha < 1
            print(f'Focal loss alpha={alpha}, will shrink the impact in background')
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] = alpha
            self.alpha[1:] = 1 - alpha
        self.gamma = gamma
        self.num_classes = num_classes

    def forward(self, preds, labels):
        """
        Calc focal loss
        :param preds: size: [B, N, C] or [B, C], corresponds to detection and classification tasks  [B, C, H, W]: segmentation
        :param labels: size: [B, N] or [B]  [B, H, W]: segmentation
        :return:
        """
        self.alpha = self.alpha.to(preds.device)
        preds = preds.permute(0, 2, 3, 1).contiguous()
        preds = preds.view(-1, preds.size(-1))
        B, H, W = labels.shape
        assert B * H * W == preds.shape[0]
        assert preds.shape[-1] == self.num_classes
        preds_logsoft = F.log_softmax(preds, dim=1)  # log softmax
        preds_softmax = torch.exp(preds_logsoft)  # softmax

        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
        alpha = self.alpha.gather(0, labels.view(-1))
        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma),
                          preds_logsoft)  # torch.low(1 - preds_softmax) == (1 - pt) ** r

        loss = torch.mul(alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1)) # b h w -> b 1 h w
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes

class DC_and_BCE_loss(nn.Module):
    def __init__(self, classes=2, dice_weight=0.8):
        """
        DO NOT APPLY NONLINEARITY IN YOUR NETWORK!
        THIS LOSS IS INTENDED TO BE USED FOR BRATS REGIONS ONLY
        :param soft_dice_kwargs:
        :param bce_kwargs:
        :param aggregate:
        """
        super(DC_and_BCE_loss, self).__init__()

        self.ce =  CrossEntropyLoss()
        self.dc = DiceLoss(classes)
        self.dice_weight = dice_weight

    def forward(self, net_output, target):
        low_res_logits = net_output['low_res_logits']
        if len(target.shape) == 4:
            target = target[:, 0, :, :]
        loss_ce = self.ce(low_res_logits, target[:].long())
        loss_dice = self.dc(low_res_logits, target, softmax=True)
        loss = (1 - self.dice_weight) * loss_ce + self.dice_weight * loss_dice
        return loss

class MaskDiceLoss(nn.Module):
    def __init__(self):
        super(MaskDiceLoss, self).__init__()

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1)) # b h w -> b 1 h w
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, net_output, target, weight=None, sigmoid=False):
        if sigmoid:
            net_output = torch.sigmoid(net_output) # b 1 h w
        assert net_output.size() == target.size(), 'predict {} & target {} shape do not match'.format(net_output.size(), target.size())
        dice_loss = self._dice_loss(net_output[:, 0], target[:, 0])
        return dice_loss

class Mask_DC_and_BCE_loss(nn.Module):
    def __init__(self, pos_weight, dice_weight=0.8):
        """
        DO NOT APPLY NONLINEARITY IN YOUR NETWORK!
        THIS LOSS IS INTENDED TO BE USED FOR BRATS REGIONS ONLY
        :param soft_dice_kwargs:
        :param bce_kwargs:
        :param aggregate:
        """
        super(Mask_DC_and_BCE_loss, self).__init__()

        self.ce =  torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.dc = MaskDiceLoss()
        self.dice_weight = dice_weight

    def forward(self, net_output, target):
        low_res_logits = net_output['low_res_logits']
        if len(target.shape) == 5:
            target = target.view(-1, target.shape[2], target.shape[3], target.shape[4])
            low_res_logits = low_res_logits.view(-1, low_res_logits.shape[2], low_res_logits.shape[3], low_res_logits.shape[4])
        loss_ce = self.ce(low_res_logits, target)
        loss_dice = self.dc(low_res_logits, target, sigmoid=True)
        loss = (1 - self.dice_weight) * loss_ce + self.dice_weight * loss_dice
        return loss


class Mask_BCE_loss(nn.Module):
    def __init__(self, pos_weight):
        """
        DO NOT APPLY NONLINEARITY IN YOUR NETWORK!
        THIS LOSS IS INTENDED TO BE USED FOR BRATS REGIONS ONLY
        :param soft_dice_kwargs:
        :param bce_kwargs:
        :param aggregate:
        """
        super(Mask_BCE_loss, self).__init__()

        self.ce =  torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, net_output, target):
        low_res_logits = net_output['low_res_logits'] 
        loss = self.ce(low_res_logits, target)
        return loss

def _rfft_radial_mask(H, W, ratio, device, dtype):
    """构造 rfft2 布局（未 fftshift）下的径向高通掩码：r >= ratio 为 1，否则 0。
    水平轴为非负频率 0..W//2，垂直轴为自然 FFT 序（含负频）。
    """
    fy = torch.fft.fftfreq(H, device=device).to(dtype).abs() * 2.0    # [0,1]
    fx = torch.fft.rfftfreq(W, device=device).to(dtype) * 2.0         # [0,1]
    r = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    return (r >= ratio).to(dtype)


def focal_frequency_loss(pred, target, alpha=1.0, hf_ratio=0.0, eps=1e-8):
    """Focal Frequency Loss (Jiang et al., ICCV 2021) —— 在频谱域上对齐预测 mask 与 GT mask。

    直觉：空域 Dice/BCE 对**边界与细小结构**（即 mask 的高频成分）的梯度贡献被大块平坦区域淹没，
    而甲状腺结节分割的误差恰恰集中在边界。FFL 把残差搬到频谱，并用 |ΔF|^alpha 作为 focal 权重，
    自动把优化压力集中到“还没拟合好的频带”上。

    Args:
        pred:   预测概率图 (N,H,W) 或 (N,1,H,W)，值域 [0,1]（已过 sigmoid）
        target: GT mask，同形状
        alpha:  focal 指数，0 退化为普通频谱 L2，1 为原文默认
        hf_ratio: >0 时**只监督高频带**（r>=hf_ratio）。低频/DC 失配已由 Dice+BCE 负责，
                重复计入会使初期损失量级爆炸（实测随机预测时全带 9.55 vs 中期 0.038，相差 250 倍）。

    注：返回值按 **GT 频谱能量归一化**（相对频谱误差）。原始 FFL 的量级随 H×W 爆炸
    （128×128 的二值 mask 实测 ≈ 4e3，而基础分割损失仅 ≈ 0.5），不归一化会直接淹没主损失。
    """
    if pred.dim() == 4:
        pred = pred.squeeze(1)
    if target.dim() == 4:
        target = target.squeeze(1)
    Fp = torch.fft.rfft2(pred.float(), dim=(-2, -1))
    Fg = torch.fft.rfft2(target.float(), dim=(-2, -1))
    diff2 = (Fp.real - Fg.real) ** 2 + (Fp.imag - Fg.imag) ** 2      # |ΔF|^2
    gt2 = Fg.real ** 2 + Fg.imag ** 2
    if hf_ratio and hf_ratio > 0:                                   # 仅高频带参与监督
        m = _rfft_radial_mask(pred.shape[-2], pred.shape[-1], hf_ratio, pred.device, diff2.dtype)
        diff2 = diff2 * m
        gt2 = gt2 * m
    w = diff2.detach() ** (alpha / 2.0)                              # focal 权重 = |ΔF|^alpha
    w = w / (w.amax(dim=(-2, -1), keepdim=True) + eps)                # 逐样本归一化到 [0,1]
    denom = gt2.sum(dim=(-2, -1)) + eps                               # GT 频谱能量（同带内）
    return ((w * diff2).sum(dim=(-2, -1)) / denom).mean()


class FreqLossWrapper(nn.Module):
    """在任意基础分割损失上**叠加** FFL（方案D）。

    weight=0 时直接返回基础损失，与原行为逐位相同；不改变任何网络结构。
    ramp_epochs>0 时前 N 轮线性升温（避免初期频域项压倒主损失）。
    """

    def __init__(self, base, weight=0.1, alpha=1.0, hf_ratio=0.15, ramp_epochs=0):
        super(FreqLossWrapper, self).__init__()
        self.base = base
        self.weight = weight
        self.alpha = alpha
        self.hf_ratio = hf_ratio
        self.ramp_epochs = ramp_epochs
        self.epoch = 0            # 由训练循环每轮末尾调用 step_epoch() 推进

    def step_epoch(self):
        self.epoch += 1

    def cur_weight(self):
        if self.ramp_epochs > 0 and self.epoch < self.ramp_epochs:
            return self.weight * (self.epoch + 1) / float(self.ramp_epochs)
        return self.weight

    def forward(self, net_output, target):
        loss = self.base(net_output, target)
        if not self.weight:
            return loss
        logits = net_output['low_res_logits']
        # 与 Mask_DC_and_BCE_loss 保持一致的形状处理
        if len(target.shape) == 5:
            target = target.view(-1, target.shape[2], target.shape[3], target.shape[4])
            logits = logits.view(-1, logits.shape[2], logits.shape[3], logits.shape[4])
        pred = torch.sigmoid(logits)
        ffl = focal_frequency_loss(pred, target, alpha=self.alpha, hf_ratio=self.hf_ratio)
        return loss + self.cur_weight() * ffl


def _sobel_grad(x):
    """x: (N,H,W) -> Sobel 梯度幅值 (N,H,W)。固定核、replicate padding，不回传核参数。"""
    kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    ky = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    x4 = x.unsqueeze(1)
    gx = F.conv2d(x4, kx, padding=1, groups=1)
    gy = F.conv2d(x4, ky, padding=1, groups=1)
    return torch.sqrt(gx ** 2 + gy ** 2 + 1e-6).squeeze(1)


def edge_loss(pred, target):
    """空域边界增强损失：对齐预测概率图与 GT 的 Sobel 梯度幅值（L1）。

    直觉：Dice/BCE 被大块平坦区主导，轮廓(空间高频)监督不足，而跨域误差恰恰集中在边界
    (HD95 高达 45~51)。直接对齐梯度图把优化压力集中到轮廓。
    与崩掉的频域 FFL 不同：概率图∈[0,1]，Sobel 梯度**有界**，数值稳定，无需归一化爆炸。
    """
    if pred.dim() == 4:
        pred = pred.squeeze(1)
    if target.dim() == 4:
        target = target.squeeze(1)
    gp = _sobel_grad(pred.float())
    gt = _sobel_grad(target.float())
    return (gp - gt).abs().mean(dim=(-2, -1)).mean()   # 逐样本平均再 batch 平均


class EdgeLossWrapper(nn.Module):
    """在任意基础分割损失上**叠加**空域边界损失（边缘增强）。

    weight=0 时直接返回基础损失，与原行为逐位相同；不改网络结构。
    ramp_epochs>0 时前 N 轮线性升温（沿用 FreqLossWrapper 的稳定化经验）。
    """

    def __init__(self, base, weight=1.0, ramp_epochs=0):
        super(EdgeLossWrapper, self).__init__()
        self.base = base
        self.weight = weight
        self.ramp_epochs = ramp_epochs
        self.epoch = 0            # 由训练循环每轮末尾调用 step_epoch() 推进

    def step_epoch(self):
        self.epoch += 1
        if hasattr(self.base, "step_epoch"):   # 嵌套时向内层传播(如 Edge(Freq(base)))
            self.base.step_epoch()

    def cur_weight(self):
        if self.ramp_epochs > 0 and self.epoch < self.ramp_epochs:
            return self.weight * (self.epoch + 1) / float(self.ramp_epochs)
        return self.weight

    def forward(self, net_output, target):
        loss = self.base(net_output, target)
        if not self.weight:
            return loss
        logits = net_output['low_res_logits']
        # 与 Mask_DC_and_BCE_loss 保持一致的形状处理
        if len(target.shape) == 5:
            target = target.view(-1, target.shape[2], target.shape[3], target.shape[4])
            logits = logits.view(-1, logits.shape[2], logits.shape[3], logits.shape[4])
        pred = torch.sigmoid(logits)
        el = edge_loss(pred, target)
        return loss + self.cur_weight() * el


class CoarseToFineLoss(nn.Module):
    """coarse-to-fine 渐进式边界精炼的**深监督**损失。

    分工：粗阶段(low_res_logits_coarse)只用基础区域损失(Dice+BCE)保区域完整性；
    细阶段(low_res_logits, 已精炼)用 区域损失 + 边缘损失 锐化边界。
    这正好修掉 edge50 的短板：边缘损失全局施加时 PT 漏分割，现在区域由粗阶段兼顾。
    edge_ramp>0 时边缘项前 N 轮线性升温（沿用稳定化经验）。
    """

    def __init__(self, base, lam_coarse=1.0, lam_fine=1.0, edge_w=0.5, edge_ramp=0):
        super(CoarseToFineLoss, self).__init__()
        self.base = base
        self.lam_coarse = lam_coarse
        self.lam_fine = lam_fine
        self.edge_w = edge_w
        self.edge_ramp = edge_ramp
        self.epoch = 0

    def step_epoch(self):
        self.epoch += 1
        if hasattr(self.base, "step_epoch"):
            self.base.step_epoch()

    def cur_edge_w(self):
        if self.edge_ramp > 0 and self.epoch < self.edge_ramp:
            return self.edge_w * (self.epoch + 1) / float(self.edge_ramp)
        return self.edge_w

    def forward(self, net_output, target):
        coarse = net_output['low_res_logits_coarse']
        fine = net_output['low_res_logits']
        # 区域损失：粗 + 细（复用基础损失，它会自行处理 5 维 target）
        l_coarse = self.base({'low_res_logits': coarse}, target)
        l_fine = self.base({'low_res_logits': fine}, target)
        # 边缘损失：只加在细阶段
        t = target
        if len(t.shape) == 5:
            t = t.view(-1, t.shape[2], t.shape[3], t.shape[4])
        f = fine
        if len(f.shape) == 5:
            f = f.view(-1, f.shape[2], f.shape[3], f.shape[4])
        le = edge_loss(torch.sigmoid(f), t)
        return self.lam_coarse * l_coarse + self.lam_fine * (l_fine + self.cur_edge_w() * le)


def get_criterion(modelname='SAM', opt=None, freq_loss_w=0.0, freq_loss_alpha=1.0,
                  freq_loss_hf=0.15, freq_loss_ramp=0, edge_loss_w=0.0, edge_loss_ramp=0,
                  c2f=0, c2f_lam_coarse=1.0, c2f_lam_fine=1.0, c2f_edge_w=0.5, c2f_edge_ramp=10):
    device = torch.device(opt.device)
    pos_weight = torch.ones([1]).cuda(device=device)*2
    if modelname == "SAMed":
        criterion = DC_and_BCE_loss(classes=opt.classes)
    elif modelname == "MSA":
        criterion = Mask_BCE_loss(pos_weight=pos_weight)
    elif modelname == "Resnet18" or modelname == "Vit":
        criterion = Classification_FocalLoss(num_classes=opt.classifier_classes)
    else:
        criterion = Mask_DC_and_BCE_loss(pos_weight=pos_weight)
    if c2f:   # coarse-to-fine：深监督(粗区域 + 细区域+边缘)，直接返回，不再叠 freq/edge
        return CoarseToFineLoss(criterion, lam_coarse=c2f_lam_coarse, lam_fine=c2f_lam_fine,
                                edge_w=c2f_edge_w, edge_ramp=c2f_edge_ramp)
    if freq_loss_w and freq_loss_w > 0:   # 方案D：叠加频域监督
        criterion = FreqLossWrapper(criterion, weight=freq_loss_w, alpha=freq_loss_alpha,
                                    hf_ratio=freq_loss_hf, ramp_epochs=freq_loss_ramp)
    if edge_loss_w and edge_loss_w > 0:   # 边缘增强：叠加空域边界监督
        criterion = EdgeLossWrapper(criterion, weight=edge_loss_w, ramp_epochs=edge_loss_ramp)
    return criterion
