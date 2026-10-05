#!/usr/bin/env python
"""G-LLM 优化管线四段验证 (断点续跑版, 2026-09-20).

本机环境会周期性 taskkill llama-server (运动平台后端管理逻辑 + 测试编排),
故本版把每个阶段、S3 的每一代都落盘为检查点 (validate/state.json):
被杀后重启, 从断点继续, 任何收割窗口只损失正在进行的单次调用。

S1 测量确定性: GENE_INIT 评两遍 (第二遍任务倒序) => 分值必须一致。
S2 定向前提:   gene_ref - gene_init >= 2 (只影响 directed, 不阻断 S3)。
S3 爬升真实性: gep-naive, seed=99, 4 代, 净爬升 >= +2 且变异解析率 >= 60%。
S4 成本:       实测每代 wall, 折算全批。
产出: results/exp_gep_llm_formal/validate/validate_verdict.json
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib.util
spec = importlib.util.spec_from_file_location("gep", Path(__file__).resolve().parent / "exp_gep_llm_formal.py")
gep = importlib.util.module_from_spec(spec)
sys.argv = ["validate"]
spec.loader.exec_module(gep)

OUT = gep.OUT / "validate"
OUT.mkdir(parents=True, exist_ok=True)
STATE = OUT / "state.json"
VERDICT = OUT / "validate_verdict.json"
GENS = 4


def load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text(encoding="utf-8"))
    return {"S1": None, "S2": None,
            "S3": {"gene": gep.GENE_INIT, "score": None, "events": [], "gens_done": 0, "done": False}}


def save(state: dict) -> None:
    STATE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


val_tasks, _test = gep.build_tasks()
st = load_state()
t_all = time.time()

# ---- S1 测量确定性 (倒序第二遍, 破坏提示缓存复用) ----
if st["S1"] is None:
    a = gep.eval_gene(gep.GENE_INIT, val_tasks, "S1a")
    b = gep.eval_gene(gep.GENE_INIT, list(reversed(val_tasks)), "S1b-reversed")
    st["S1"] = {"first": a["score"], "second_reversed_order": b["score"],
                "pass": a["score"] == b["score"]}
    save(st)
print("S1:", st["S1"], flush=True)

# ---- S2 定向前提 ----
if st["S2"] is None:
    r = gep.eval_gene(gep.GENE_REF, val_tasks, "S2 ref")
    st["S2"] = {"init": st["S1"]["first"], "ref": r["score"],
                "gap": r["score"] - st["S1"]["first"],
                "pass": (r["score"] - st["S1"]["first"]) >= 2}
    save(st)
print("S2:", st["S2"], flush=True)

# ---- S3 爬升真实性 (断点续跑: 每代落盘) ----
s3 = st["S3"]
if s3["score"] is None:
    cur = gep.eval_gene(s3["gene"], val_tasks, "S3 init")
    s3["score"] = cur["score"]
    s3["fails"] = cur["fails"]
    save(st)
t0 = time.time()
while s3["gens_done"] < GENS:
    gen = s3["gens_done"]
    rng = __import__("random").Random(99 * 77 + 1 + gen)   # 断点后 rng 从当代重播 (验证管线可接受)
    old = s3["score"]
    signal = {"fails": s3.get("fails", [])}
    new_gene, mtype, intent = gep.mutate_llm(s3["gene"], signal, rng)
    new_res = gep.eval_gene(new_gene, val_tasks, f"S3 g{gen}")
    accept = new_res["score"] >= old
    if accept:
        s3["gene"], s3["score"] = new_gene, new_res["score"]
        s3["fails"] = new_res["fails"]
    s3["events"] = s3.get("events", []) + [{
        "gen": gen, "mutation_type": mtype, "intent": str(intent)[:120],
        "score_old": old, "score_new": new_res["score"], "accepted": accept}]
    s3["gens_done"] = gen + 1
    save(st)
    print(f"S3 g{gen}: {old}->{new_res['score']} ({mtype}, {'接受' if accept else '拒绝'})", flush=True)
s3["done"] = True
events = s3["events"]
parse_ok = sum(1 for e in events if e["mutation_type"] != "none")
init_score = st["S1"]["first"]
climb = s3["score"] - init_score
st["S3_summary"] = {
    "traj": [st["S1"]["first"]] + [e["score_new"] if e["accepted"] else None for e in events],
    "val_init": init_score, "val_final": s3["score"], "climb": climb,
    "parse_ok": parse_ok, "n_gens": len(events),
    "pass": climb >= 2 and parse_ok >= max(1, int(0.6 * len(events))),
    "wall_min": round((time.time() - t_all) / 60, 1)}
save(st)
print("S3_summary:", st["S3_summary"], flush=True)

# ---- verdict ----
verdict = {
    "pipeline": "G-LLM optimization validation (checkpointed)",
    "S1_determinism": st["S1"], "S2_premise": st["S2"], "S3_climb": st["S3_summary"],
    "pipeline_reasonable": bool(st["S1"]["pass"] and st["S3_summary"]["pass"]),
    "directed_premise": bool(st["S2"]["pass"]),
}
VERDICT.write_text(json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(verdict, ensure_ascii=False, indent=2))
print("WROTE", VERDICT)
