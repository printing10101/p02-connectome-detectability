#!/usr/bin/env python
"""G0 闸门探针: 在当前 llama-server 实例上评估 GENE_INIT vs GENE_REF (12 验证题, 512 token)。
写出 results/exp_gep_llm_formal/g0_probe.json, 供监督循环判定闸门是否通过。"""
from __future__ import annotations
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
OUT = Path(__file__).resolve().parents[1] / "results" / "exp_gep_llm_formal"

import importlib.util
spec = importlib.util.spec_from_file_location("gep", Path(__file__).resolve().parent / "exp_gep_llm_formal.py")
gep = importlib.util.module_from_spec(spec)
sys.argv = ["probe"]
spec.loader.exec_module(gep)

val_tasks, _ = gep.build_tasks()
i = gep.eval_gene(gep.GENE_INIT, val_tasks, "G0 init")
r = gep.eval_gene(gep.GENE_REF, val_tasks, "G0 ref")
gate_pass = (r["score"] - i["score"]) >= 2
res = {
    "gene_init": i["score"], "gene_ref": r["score"],
    "gate_pass": gate_pass,
    "rule": "gate: gene_ref - gene_init >= 2 (参考基因须显著值钱, handwire 纪律)",
    "init_fails": [f["id"] for f in i["fails"]],
    "ref_fails": [f["id"] for f in r["fails"]],
}
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "g0_probe.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(res, ensure_ascii=False))
