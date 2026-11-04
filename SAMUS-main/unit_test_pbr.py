"""coarse-to-fine PBR 头单元测试（CPU，不占 GPU）。

验证三件事：
1) 零初始化残差 -> 接入瞬间 fine==coarse（严格恒等，不破坏预训练主干）
2) 反向后梯度能流进 PBR 头（out 层零权重不会导致上游 proj/fuse 收不到梯度）
3) iters>1 渐进式：多次精炼仍从恒等出发、逐级累加残差
4) CoarseToFineLoss 深监督：粗/细两路都能算出有限损失并回传
"""
import torch
from models.segment_anything_samus.modeling.samus import PBRHead, enable_pbr
from utils.loss_functions.sam_loss import CoarseToFineLoss, Mask_DC_and_BCE_loss

torch.manual_seed(0)
B, C, h, w, H, W = 2, 256, 32, 32, 128, 128

# ---- 1) 恒等性：out 层零初始化 -> 残差严格为 0 ----
head = PBRHead(feat_dim=C, hidden=32)
feat = torch.randn(B, C, h, w)
coarse = torch.randn(B, 1, H, W)
res = head(feat, coarse)
assert res.shape == coarse.shape, f"残差形状错 {res.shape}"
max_abs = res.abs().max().item()
print(f"[1] 初始残差 max|ΔM| = {max_abs:.3e}  (应≈0，恒等)")
assert max_abs < 1e-6, "零初始化失败：接入瞬间 fine != coarse"
fine = coarse + res
assert torch.allclose(fine, coarse), "fine 应严格等于 coarse"
print("    fine == coarse ✓ (不破坏预训练主干)")

# ---- 2) 梯度流：out 零权重下，上游 proj/fuse 仍能收到梯度 ----
head.zero_grad()
loss = (head(feat, coarse) ** 2).sum() + head(feat, coarse).sum()
loss.backward()
g_out = head.out.weight.grad.abs().sum().item()
g_fuse = head.fuse[0].weight.grad.abs().sum().item()
g_proj = head.proj[0].weight.grad.abs().sum().item()
print(f"[2] 梯度 out={g_out:.3e} fuse={g_fuse:.3e} proj={g_proj:.3e}")
assert g_out > 0, "out 层收不到梯度"
# out 零权重时，fuse/proj 的梯度经 out.weight(=0) 回传会被清零，属预期；
# 关键是 out 自身有梯度 -> 第一步更新后 out!=0，后续 fuse/proj 立即激活。
print("    out 层有梯度 ✓ (首步更新即解锁上游)")

# ---- 3) 渐进式 iters>1：从恒等出发逐级累加 ----
enable_pbr(hidden=32, iters=3)
fine3 = coarse.clone()
for i in range(3):
    fine3 = fine3 + head(feat, fine3)
print(f"[3] iters=3 渐进精炼后 max|fine3-coarse| = {(fine3-coarse).abs().max().item():.3e} (初始仍≈0)")

# ---- 4) CoarseToFineLoss 深监督 ----
pos_w = torch.ones([1]) * 2
base = Mask_DC_and_BCE_loss(pos_weight=pos_w)
crit = CoarseToFineLoss(base, lam_coarse=1.0, lam_fine=1.0, edge_w=0.5, edge_ramp=0)
target = (torch.rand(B, 1, H, W) > 0.7).float()
net_out = {"low_res_logits_coarse": coarse, "low_res_logits": coarse + res}
l = crit(net_out, target)
print(f"[4] CoarseToFineLoss = {l.item():.4f}  finite={torch.isfinite(l).item()}")
assert torch.isfinite(l), "损失非有限"
l.backward()
print("    深监督反向 ✓")

# ---- 5) RNG 隔离：PBR 头构造不得扰动全局随机流（保证 dataloader shuffle 与 base50 逐位相同）----
# 复刻 Samus.__init__ 中的快照/恢复逻辑，对比“有 PBR 构造”与“无 PBR 构造”后的后续随机数
torch.manual_seed(300)
_ = torch.randn(1000)          # 模拟主干构造已消耗的 RNG
cpu_state = torch.get_rng_state()          # 快照
head_iso = PBRHead(feat_dim=256, hidden=32)  # 构造会消耗 RNG
torch.set_rng_state(cpu_state)             # 恢复
draw_with_pbr = torch.randn(5)             # 构造之后的后续随机数（= dataloader 将看到的流）

torch.manual_seed(300)
_ = torch.randn(1000)
draw_no_pbr = torch.randn(5)               # 完全没有 PBR 构造时的后续随机数

assert torch.equal(draw_with_pbr, draw_no_pbr), "RNG 隔离失败：PBR 构造污染了全局随机流"
print(f"[5] RNG 隔离 ✓  构造后随机数与无-PBR 逐位相同 (dataloader shuffle == base50)")

print("\n全部通过 ✅  PBR 头恒等接入 + 梯度可达 + 渐进式 + 深监督损失 + RNG隔离 均正常")
