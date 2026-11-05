#!/usr/bin/env python3
"""
extract_blsamus_ckpt.py — 从 blsamus SAMUSAdapter checkpoint 提取 SAMUS-main 兼容 state_dict。
  用法: python3 extract_blsamus_ckpt.py <input.pth> <output.pth>
  逻辑: 去掉 'module.samus_model.' 前缀，保留 image_encoder/mask_decoder/prompt_encoder/auto_prompt_generator。
"""
import sys, torch

def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input.pth> <output.pth>"); sys.exit(1)
    inp, outp = sys.argv[1], sys.argv[2]
    sd = torch.load(inp, map_location='cpu')
    if isinstance(sd, dict) and 'model' in sd:
        sd = sd['model']
    # Strip DDP + SAMUSAdapter wrapper prefix
    PREFIX = "module.samus_model."
    KEEP_PREFIXES = ("image_encoder.", "mask_decoder.", "prompt_encoder.", "auto_prompt_generator.")
    new_sd = {}
    for k, v in sd.items():
        stripped = k.replace(PREFIX, "") if k.startswith(PREFIX) else k
        if stripped.startswith(KEEP_PREFIXES):
            new_sd[stripped] = v
    print(f"[extract] input={len(sd)} keys → output={len(new_sd)} keys (dropped {len(sd)-len(new_sd)})")
    torch.save(new_sd, outp)
    print(f"[extract] saved → {outp}")

if __name__ == "__main__":
    main()
