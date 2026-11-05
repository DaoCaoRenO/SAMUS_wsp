#!/usr/bin/env python3
# =============================================================================
# rag_head.py — 原型条件化精炼头 (PCR head, Prototype-Conditioned Refinement)
#   对应计划 §2。承接免训练 RAG 四轮定案：检索原型先验能稳定显著抬 SEN，但任何
#   **手工**加性融合都无法把召回转成净 Dice（跨域漏检在周边/离断区、与真空重叠）。
#   本轮假设：一个**可学习**的、检索条件化的头，能学会「base 不可靠时相信原型先验
#   到什么程度、在哪里补」，从而把召回增益转成 Dice。
#
#   设计（沿用 PBRHead 成熟范式）：
#     - 输入 imge(B,256,32,32) prompt-fused 解码侧特征、base_logits(B,1,256,256)、
#       sim_fg/sim_bg(B,1,32,32) 逐位置与聚合前景/背景原型的余弦。
#     - concat[ proj(imge 1x1→hidden), base32, sim_fg, sim_bg ] → 2 层 conv(hidden)
#       → 末层 1x1 conv 输出单通道 Δ，**权重零初始化** → Δ 上采样 → masks_fine=masks+Δ。
#     - 末层零初始化 → 接入瞬间 masks_fine==masks 严格恒等，不破坏冻结主干。
#     - hidden≈32，参数量 ~3-5 万，主干全冻结(唯一可训练)。
#   RNG 快照隔离：构造头会消耗全局随机流，用 get/set_rng_state 保证 dataloader 顺序
#     与既有基线逐位一致(沿用 PBRHead 坑记忆)。
# =============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F


class PCRHead(nn.Module):
    """原型条件化精炼头。forward 返回 masks_fine = base_logits + Δ(零初始化残差)。"""

    def __init__(self, feat_dim: int = 256, hidden: int = 32) -> None:
        super().__init__()
        # 解码侧稠密特征降到 hidden 通道(1x1)，与 base/sim 图在 32×32 网格拼接
        self.proj = nn.Sequential(
            nn.Conv2d(feat_dim, hidden, 1),
            nn.GroupNorm(1, hidden), nn.GELU(),
        )
        # concat 通道数 = hidden(特征) + 1(base32) + 1(sim_fg) + 1(sim_bg)
        self.fuse = nn.Sequential(
            nn.Conv2d(hidden + 3, hidden, 3, padding=1),
            nn.GroupNorm(1, hidden), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GroupNorm(1, hidden), nn.GELU(),
        )
        self.out = nn.Conv2d(hidden, 1, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)   # 末层零初始化 → 初始恒等残差

    def forward(self, imge: torch.Tensor, base_logits: torch.Tensor,
                sim_fg: torch.Tensor, sim_bg: torch.Tensor) -> torch.Tensor:
        """imge(B,256,h,w)、base_logits(B,1,H,W)、sim_fg/sim_bg(B,1,h,w) → masks_fine(B,1,H,W)。"""
        H, W = base_logits.shape[-2:]
        gh, gw = imge.shape[-2:]
        base32 = F.interpolate(base_logits, size=(gh, gw), mode="bilinear", align_corners=False)
        # sim 图若分辨率与 imge 不一致，对齐到 imge 网格
        if sim_fg.shape[-2:] != (gh, gw):
            sim_fg = F.interpolate(sim_fg, size=(gh, gw), mode="bilinear", align_corners=False)
            sim_bg = F.interpolate(sim_bg, size=(gh, gw), mode="bilinear", align_corners=False)
        x = torch.cat([self.proj(imge), base32, sim_fg, sim_bg], dim=1)
        delta_small = self.out(self.fuse(x))                       # (B,1,gh,gw)
        delta = F.interpolate(delta_small, size=(H, W), mode="bilinear", align_corners=False)
        return base_logits + delta


def build_pcr_head(hidden: int = 32, feat_dim: int = 256) -> PCRHead:
    """构造 PCRHead，并用 RNG 状态快照隔离，避免头初始化扰动全局随机流→破坏基线可比性。"""
    cpu_state = torch.get_rng_state()
    cuda_states = None
    if torch.cuda.is_available():
        cuda_states = [torch.cuda.get_rng_state(i) for i in range(torch.cuda.device_count())]
    head = PCRHead(feat_dim=feat_dim, hidden=hidden)
    torch.set_rng_state(cpu_state)
    if cuda_states is not None:
        for i, s in enumerate(cuda_states):
            torch.cuda.set_rng_state(s, i)
    return head


class RagEvalRunner:
    """评测用的轻量包装(非 nn.Module)：把 backbone + PCR 头的「检索→精炼」路径接成
       一次 model(imgs) 调用，供 utils.evaluation.get_eval 直接复用。返回 dict 中
       'masks'=masks_fine(供指标)，'low_res_logits' 保留原 128 粗输出(供 val_loss)。
       只读 backbone 与 bank，不传 GT；val 图不在训练库内→exclude=None 无自泄漏。"""

    def __init__(self, model, bank, k, retr="roi") -> None:
        self.model = model
        self.bank = bank
        self.k = k
        self.retr = retr

    def __call__(self, imgs):
        import rag_common as R
        out = self.model(imgs.to(dtype=torch.float32))
        imge, base = out["imge"], out["masks"]
        sim_fg, sim_bg = R.prototypes_for_query(imge, self.bank, self.k, exclude=None, retr=self.retr)
        fine = self.model.pcr(imge, base, sim_fg, sim_bg)
        return {"masks": fine, "low_res_logits": out["low_res_logits"], "imge": imge}

    def eval(self):
        self.model.eval()
        return self

    def train(self, mode=True):
        self.model.train(mode)
        return self
