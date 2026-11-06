#!/usr/bin/env python3
"""
bl_nothy_to_jsonl.py — 把 blsamus omni_test.py 输出的 result.csv 里的 DDTI/private_Thyroid/TN3K
三个 dice 指标转成 SAMUS-main crossdomain_results.jsonl 里的三行 (tag=bl_nothy) 记录。

由于 blsamus 224 输入 vs SAMUS-main 256 输入存在 shape 差异 (rel_pos_h/w 27 vs 63),
B 权重无法通过 extract_blsamus_ckpt → rag_segment 走通; 改用 blsamus 自带 omni_test 评,
再由本脚本桥接到统一 jsonl。指标对齐 dice (SEN/SPC/HD95 留空, make_*.py 会显示 N/A)。
"""
import argparse, csv, json, os


MAP_DS = {
    "DDTI": "DDTIz",
    "private_Thyroid": "PTz",
    "TN3K": "TN3K",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-csv", required=True, help="blsamus/exp_out/result.csv")
    ap.add_argument("--emit-table", required=True, help="SAMUS-main/crossdomain_results.jsonl")
    ap.add_argument("--tag", default="bl_nothy")
    args = ap.parse_args()

    # 只取 seg 任务的行
    want = []
    with open(args.result_csv) as f:
        rd = csv.reader(f)
        header = next(rd)
        for row in rd:
            if len(row) < 3: continue
            ds_name, task, perf = row[0], row[1], row[2]
            if not task.startswith("omni_seg"): continue
            if ds_name not in MAP_DS: continue
            try:
                dice_pct = float(perf) * 100.0
            except Exception:
                continue
            want.append((MAP_DS[ds_name], dice_pct))

    print(f"[bl_jsonl] found {len(want)} seg rows from {args.result_csv}")
    for ds, d in want:
        print(f"  {ds}: dice={d:.2f}")

    # upsert: 移除同 (tag, ds) 旧行, 追加新行
    tag = args.tag
    rows = []
    if os.path.exists(args.emit_table):
        for line in open(args.emit_table):
            line = line.strip()
            if not line: continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get("tag") == tag and d.get("ds") in [x[0] for x in want]:
                continue
            rows.append(line)
    for ds, d in want:
        rec = {
            "task": ds, "model": "SAMUS", "ckpt": "bl_nothy",
            "tag": tag, "ds": ds,
            "params": {"k": 1, "lambda": 0.0, "tau": None, "retr": None,
                       "agg": None, "stack_e": None},
            "dice_std": None, "base_dice": None, "base_se": None,
            "dice": d, "iou": None, "acc": None, "se": None, "sp": None, "hdis": None,
            "note": "dice-only from blsamus omni_test.py; 224 vs 256 img_size prevents rag_segment reuse",
        }
        rows.append(json.dumps(rec))
    with open(args.emit_table, "w") as f:
        f.write("\n".join(rows) + "\n")
    print(f"[bl_jsonl] upsert tag={tag} rows={len(want)} -> {args.emit_table}")


if __name__ == "__main__":
    main()
