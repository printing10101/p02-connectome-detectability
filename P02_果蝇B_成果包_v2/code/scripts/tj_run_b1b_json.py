#!/usr/bin/env python
"""TJ-B1b: 结构化输出通道重跑 (30B 档) — 冻结判据 TJ-B1b-frozen-v1.0.json

与 B1 (tj_run_b1_llm.py) 的全部差异 (其余逐字锁同):
  1. 仅 mutate_llm 的 LLM 调用加 response_format=json_schema 受限解码 (schema 固定);
     eval_gene 评分调用保持自由文本温度0 —— 复用 b1.evolve, 只换变异通道。
  2. 只跑 directed + naive; sham 复用 B1 批 30B 事件文件 (程序化变异 + 温度0评分,
     逐文件 sha256 留痕), 不重跑。
  3. 只跑 qwen3-instruct-30b。
  4. 新增通道纪律: 开跑前 5 次带 schema 的格式探针 (>=4/5 解析才进批);
     判决新增 B1B_CHANNEL_FAIL (directed 解析失败率 > 0.05, 效应量只算不解释)。
  5. 判据/输出独立: 判据在 仓库根 TJ-B1b-frozen-v1.0.json (先冻结后运行),
     结果写 results/tj_b1b_llm_json/, 进度写该目录 progress.md (不动 TJ1_batch_progress.md)。
  6. 逐代断点 (evolve_ckpt/run_one_ckpt): 每个 gen 结束落盘 partial, 进程被杀后
     从断点恢复, 协议状态逐字段同构 —— 纯工程容错, 判决/判据/臂语义零改动。

判决 (TJ-B1b-frozen-v1.0.json, 本脚本不改动其任何阈值):
  B1B_CHANNEL_FAIL : directed 解析失败率 > 0.05 —— 通道伪影检验失败, 不解释效应
  B1B_DIRECTED_WINS: directed−sham 配对 95% CI 排除 0 且 >=7/10 种子同向
  B1B_EQUIV        : TOST 等价成立 (delta=1.0/20题, 沿用 B1 操作化)
  B1B_NOISE        : 种子均方 > 10x 臂均方
  自报误分率 > 0.2 : 仅约束 directed, 判 B1B_SELF_REPORT_INVALID (naive 只报告不约束)
  其余             : B1B_INCONCLUSIVE
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tj_run_b1_llm as b1  # 复用: chat重试/eval_gene/evolve/统计与分类器, 全部不改

ROOT = Path(__file__).resolve().parents[2]          # 数字果蝇进化研究/
_TAG = ""            # main() 按 --tag 设置: ''=B1b 原判据/原目录; 'b1c'=B1C 判据/独立目录
CRIT_PATH = ROOT / "TJ-B1b-frozen-v1.0.json"
KEY = "TJ_B1B_llm_jsonschema_30b"
MUTATE_MAX_TOKENS = 800   # main() 按 --mutate-max-tokens 设置 (B1c=2048, 修复截断)
OUT = b1.ROOT / "results" / "tj_b1b_llm_json"        # b1.ROOT = 数字果蝇进化研究/code
REUSE_SHAM_DIR = b1.OUT / "qwen3-instruct-30b"       # B1 批 30B 事件目录
MODEL = "qwen3-instruct-30b"

MISCLASS_MAX = 0.2        # 与 B1 冻结阈值同值, 作用域仅 directed (判据文件已声明)
TOST_DELTA = b1.TOST_DELTA
NOISE_MS_RATIO = b1.NOISE_MS_RATIO
CONDS = ("directed", "naive")  # sham 来自复用文件


def load_crit() -> dict:
    global CRIT_PATH
    if _TAG:
        CRIT_PATH = ROOT / f"TJ-{_TAG.upper()}-frozen-v1.0.json"
    if not CRIT_PATH.exists():
        raise RuntimeError(f"B1 冻结判据缺失: {CRIT_PATH} —— 拒绝运行")
    return json.loads(CRIT_PATH.read_bytes().decode("utf-8"))


def crit_sha() -> str:
    return hashlib.sha256(CRIT_PATH.read_bytes()).hexdigest()


# ---------- 受限解码变异通道 (唯一的方法学改动) ----------

SCHEMA = {
    "type": "object",
    "properties": {
        "mutation_type": {"type": "string", "enum": ["param", "arch"]},
        "intent": {"type": "string"},
        "new_gene": {"type": "string"},
    },
    "required": ["mutation_type", "intent", "new_gene"],
}
RESPONSE_FORMAT = {"type": "json_schema",
                   "json_schema": {"name": "mutation", "strict": True, "schema": SCHEMA}}


def chat_schema(messages: list[dict], temperature: float, max_tokens: int,
                meter: b1.TokenMeter, model: str, base_url: str) -> str:
    """与 b1.chat 同一重试/超时/计量, 仅多 response_format 字段."""
    last_err = None
    for attempt in range(30):
        try:
            t0 = time.time()
            r = requests.post(
                f"{base_url}/chat/completions",
                headers=b1.AUTH,
                json={"model": model, "messages": messages,
                      "temperature": temperature, "max_tokens": max_tokens,
                      "response_format": RESPONSE_FORMAT,
                      "cache_prompt": True,
                      "chat_template_kwargs": {"enable_thinking": False}},
                timeout=420,
            )
            r.raise_for_status()
            body = r.json()
            meter.add(body.get("usage", {}))
            out = body["choices"][0]["message"]["content"] or ""
            if "</think>" in out:
                out = out.rsplit("</think>", 1)[-1]
            print(f"    [llm {time.time() - t0:.0f}s]", flush=True)
            return out
        except Exception as e:
            last_err = e
            if attempt == 29:
                raise
            if attempt % 5 == 0:
                print(f"  [retry {attempt + 1}/30 after 30s] {type(e).__name__}", flush=True)
            time.sleep(30)
    raise last_err


def mutate_llm_schema(gene: str, signal: dict, meter: b1.TokenMeter, model: str,
                      base_url: str) -> tuple[str, str, str, bool]:
    """与 b1.mutate_llm 同一提示词, 唯一差异: 受限解码; 解析只走严格路径."""
    slim = [{"id": f["id"], "expected": f["expected"]} for f in signal["fails"][:4]]
    user = (f"当前基因:\n{gene}\n\n最近验证未通过的任务(优先修复这些族):\n"
            f"{json.dumps(slim, ensure_ascii=False)}\n\n产出一个改进版基因。")
    out = chat_schema([{"role": "system", "content": b1.MUTATE_SYS},
                       {"role": "user", "content": user}], temperature=0.8,
                      max_tokens=MUTATE_MAX_TOKENS,
                      meter=meter, model=model, base_url=base_url)
    try:
        j = json.loads(out)
    except json.JSONDecodeError:
        # 受限解码下不应发生; 发生即通道缺陷, 照 B1 口径记 PARSE_FAIL
        return gene, "none", "PARSE_FAIL:" + out[:150], False
    return (j.get("new_gene", gene), j.get("mutation_type", "param"),
            j.get("intent", ""), True)


def format_probe(model: str, base_url: str) -> tuple[int, int]:
    """L4 纪律: 通道接入前先做格式探针. 5 次带 schema 变异调用, >=4 次可解析才进批."""
    meter = b1.TokenMeter()
    signal = {"fails": [{"id": "probe", "expected": "0"}]}
    ok = 0
    for i in range(5):
        _, mtype, _, parse_ok = mutate_llm_schema(
            b1.GENE_INIT, signal, meter, model, base_url)
        ok += bool(parse_ok and mtype in ("param", "arch"))
        print(f"  [probe {i + 1}/5] parse_ok={parse_ok} type={mtype}", flush=True)
    print(f"格式探针: {ok}/5 可解析 (阈值 >=4)", flush=True)
    return ok, meter.snapshot()["completion_tokens"]


# ---------- sham 复用 ----------

def load_reused_sham() -> tuple[list[dict], list[dict]]:
    runs, provenance = [], []
    for seed in range(1, 11):
        f = REUSE_SHAM_DIR / f"events_sham_s{seed}.json"
        if not f.exists():
            raise RuntimeError(f"sham 复用文件缺失: {f} —— 拒绝运行 (不许只跑两臂出判决)")
        raw = f.read_bytes()
        r = json.loads(raw.decode("utf-8"))
        if r.get("cond") != "sham":
            raise RuntimeError(f"{f.name} cond != sham —— 复用源不合法")
        runs.append(r)
        provenance.append({"file": str(f.relative_to(ROOT)),
                           "sha256": hashlib.sha256(raw).hexdigest(),
                           "test_final": r["test_final"]})
    print(f"sham 复用: 10 个 run 已装载 (B1 批原文件)", flush=True)
    return runs, provenance


# ---------- 判决 (B1 优先级 + 通道前置闸门) ----------

def verdict_for_model(runs: list[dict], model_id: str, sham_prov: list[dict]) -> dict:
    per = {c: {r["seed"]: r["test_final"] for r in runs if r["cond"] == c}
           for c in ("directed", "sham", "naive")}
    need = sorted({r["seed"] for r in runs})
    complete = len(need) >= 2 and all(all(s in per[c] for s in need) for c in per)
    if not complete:
        return {"model": model_id, "verdict": "INCOMPLETE",
                "n_runs": len(runs), "expected_runs": len(need) * 3}

    diffs = [per["directed"][s] - per["sham"][s] for s in need]
    ci = b1._paired_ci(diffs)
    n_pos = sum(1 for d in diffs if d > 0)
    tost = b1._tost_equivalent(diffs, TOST_DELTA)
    varcomp = b1._variance_components({s: {c: per[c][s] for c in per} for s in need})
    mis_d = b1.misclassification_rate(runs, "directed")
    mis_n = b1.misclassification_rate(runs, "naive")

    attempts_d = mis_d["n_self_reported"] + mis_d["n_parse_fail"]
    parse_fail_rate = (mis_d["n_parse_fail"] / attempts_d) if attempts_d else 0.0
    self_report_valid = not (mis_d["rate"] is not None and mis_d["rate"] > MISCLASS_MAX)

    if parse_fail_rate > 0.05:
        core = "B1B_CHANNEL_FAIL"
    elif ci["ci_excludes_zero"] and n_pos >= 7:
        core = "B1B_DIRECTED_WINS"
    elif tost["equivalent_at_0.05"]:
        core = "B1B_EQUIV"
    elif varcomp["seed_over_arm_ratio"] > NOISE_MS_RATIO:
        core = "B1B_NOISE"
    else:
        core = "B1B_INCONCLUSIVE"

    return {
        "model": model_id,
        "experiment": KEY,
        "criteria": load_crit()["criteria"],
        "criteria_sha256": crit_sha(),
        "operationalization": {
            "tost_delta": TOST_DELTA, "noise_ms_ratio": NOISE_MS_RATIO,
            "parse_fail_rate_max": 0.05,
            "misclass_scope": "directed only (naive 报告不约束), 与 B1 判决代码一致",
        },
        "channel": {"directed_attempts": attempts_d,
                    "directed_parse_fail": mis_d["n_parse_fail"],
                    "directed_parse_fail_rate": parse_fail_rate},
        "directed_minus_sham": {**ci, "n_seeds_positive": n_pos},
        "tost": tost,
        "variance_components": varcomp,
        "self_report_misclassification": {"directed": mis_d, "naive": mis_n,
                                          "frozen_max": MISCLASS_MAX,
                                          "directed_valid": self_report_valid},
        "test_final_per_cond": {c: [per[c][s] for s in need] for c in sorted(per)},
        "sham_reuse_provenance": sham_prov,
        "verdict": ("B1B_SELF_REPORT_INVALID|" + core) if not self_report_valid else core,
        "verdict_if_valid": core,
    }


_PROG_PATH = OUT / "progress.md"  # main() 按 --smoke 重设


# ---------- 逐代断点续跑 (工程容错, 不涉判据; 与 b1.evolve/run_one 逻辑锁同) ----------

def evolve_ckpt(cond: str, seed: int, val_tasks, test_tasks, gens: int,
                model: str, base_url: str, meter: b1.TokenMeter, part_path) -> dict:
    """b1.evolve 的逐代 checkpoint 版: 每个 gen 结束后落盘 partial.

    状态 (gene/cur/事件/token计量) 全部随 partial 恢复, 续跑后的协议状态与
    不间断运行逐字段同构; 判决、判据、臂语义零改动。
    """
    rng = __import__("random").Random(seed * 77 + 1)
    if part_path.exists():
        st = json.loads(part_path.read_text(encoding="utf-8"))
        gene, events, cur = st["gene"], st["events"], st["cur"]
        meter.calls = st["meter"]["llm_calls"]
        meter.prompt_tokens = st["meter"]["prompt_tokens"]
        meter.completion_tokens = st["meter"]["completion_tokens"]
        start_gen = len(events)
        print(f"  [{cond} s{seed}] partial 恢复: 从 g{start_gen} 继续 (分 {cur['score']})",
              flush=True)
    else:
        gene = {"naive": b1.GENE_INIT, "directed": b1.GENE_REF}[cond]
        events = []
        cur = b1.eval_gene(gene, val_tasks, f"{cond} s{seed} init", meter, model, base_url)
        start_gen = 0
    for gen in range(start_gen, gens):
        signal = {"fails": cur["fails"]}
        new_gene, mtype_self, intent, parse_ok = mutate_llm_schema(
            gene, signal, meter, model, base_url)
        new_res = b1.eval_gene(new_gene, val_tasks, f"{cond} s{seed} g{gen}", meter, model, base_url)
        accept = new_res["score"] >= cur["score"]
        ev = {"seed": seed, "cond": cond, "gen": gen,
              "mutation_type": mtype_self, "intent": intent,
              "score_old": cur["score"], "score_new": new_res["score"],
              "accepted": accept}
        mtype_prog = b1.classify_edit(gene, new_gene)
        ev["mutation_type_prog"] = mtype_prog
        ev["self_report_parse_ok"] = parse_ok
        ev["misclassified"] = bool(parse_ok and mtype_self in ("param", "arch")
                                   and mtype_self != mtype_prog)
        events.append(ev)
        print(f"  [{cond} s{seed}] g{gen}: {cur['score']}->{new_res['score']} "
              f"({mtype_self}, {'接受' if accept else '拒绝'})", flush=True)
        if accept:
            gene, cur = new_gene, new_res
        part_path.write_text(json.dumps(
            {"gene": gene, "events": events, "cur": cur,
             "meter": meter.snapshot()}, ensure_ascii=False), encoding="utf-8")
    test = b1.eval_gene(gene, test_tasks, f"{cond} s{seed} TEST", meter, model, base_url)
    return {"cond": cond, "seed": seed, "val_final": cur["score"],
            "test_final": test["score"], "events": events, "final_gene": gene,
            "token_usage": meter.snapshot()}


def run_one_ckpt(model_id: str, cond: str, seed: int, val_tasks, test_tasks, gens: int,
                 base_url: str, out_dir: Path) -> dict:
    """与 b1.run_one 同一落盘/跳过语义, 增加逐代 partial 与其清理."""
    out_file = out_dir / f"events_{cond}_s{seed}.json"
    if out_file.exists():
        print("SKIP", model_id, out_file.name, flush=True)
        return json.loads(out_file.read_text(encoding="utf-8"))
    part_path = out_dir / f"events_{cond}_s{seed}.partial.json"
    resumed = part_path.exists()
    meter = b1.TokenMeter()
    t0 = time.time()
    r = evolve_ckpt(cond, seed, val_tasks, test_tasks, gens, model_id, base_url,
                    meter, part_path)
    r["model"] = model_id
    r["wall_min"] = round((time.time() - t0) / 60, 1)
    r["resumed"] = resumed
    out_file.write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
    part_path.unlink(missing_ok=True)
    print(f"  WROTE {out_file.name} (test {r['test_final']}/{len(test_tasks)}, "
          f"tok {meter.prompt_tokens}/{meter.completion_tokens}, {r['wall_min']} min)", flush=True)
    return r


def progress(msg: str):
    line = f"- [{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    try:
        with open(_PROG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError as e:
        print(f"[progress write failed: {e}]", flush=True)


def main() -> None:
    global b1
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=b1.BASE_URL)
    ap.add_argument("--seeds", default="1-10")
    ap.add_argument("--gens", type=int, default=b1.GENS)
    ap.add_argument("--tag", default="",
                    help="实验标签: ''=B1b(原判据/原目录); 如 b1c=读 TJ-B1C-frozen-v1.0.json, "
                         "输出写 results/tj_b1b_llm_json_b1c/")
    ap.add_argument("--mutate-max-tokens", type=int, default=800,
                    help="变异调用 token 上限 (B1c=2048, 修复 new_gene 截断); 评分调用不受此参数影响")
    ap.add_argument("--smoke", action="store_true",
                    help="冒烟: 1 种子 x 2 代 x 4+4 题, 输出隔离目录")
    args = ap.parse_args()

    globals()["_TAG"] = args.tag.strip().lower()
    globals()["MUTATE_MAX_TOKENS"] = args.mutate_max_tokens
    if _TAG == "b1c":
        globals()["KEY"] = "TJ_B1C_mutate_maxtok_30b"

    if args.seeds.count("-") and "," not in args.seeds:
        a, b = args.seeds.split("-")
        seeds = tuple(range(int(a), int(b) + 1))
    else:
        seeds = tuple(int(t) for t in args.seeds.split(","))

    out_root = OUT
    if _TAG:
        out_root = OUT.parent / f"{OUT.name}_{_TAG}"
    if args.smoke:
        seeds, args.gens = (1,), 2
        b1.N_VAL = b1.N_TEST = 4
        out_root = OUT.parent / (out_root.name + "_smoke")
    out_root.mkdir(parents=True, exist_ok=True)
    globals()["_PROG_PATH"] = out_root / "progress.md"

    load_crit()  # 判据必须在开跑前可读, 否则响亮失败
    val_tasks, test_tasks = b1.build_tasks()
    assert len(val_tasks) == b1.N_VAL and len(test_tasks) == b1.N_TEST
    assert len({t["prompt"] for t in val_tasks + test_tasks}) == b1.N_VAL + b1.N_TEST

    progress(f"b1b{_TAG} 开始 (model={MODEL}; directed+naive {len(seeds)}种子 x {args.gens}代; "
             f"sham 复用 B1 批; 任务 {b1.N_VAL}验证+{b1.N_TEST}测试; "
             f"变异max_tokens={MUTATE_MAX_TOKENS}; 冒烟={args.smoke})")

    ok, probe_tok = format_probe(MODEL, args.base_url)
    if ok < 4:
        progress(f"b1b 中止: 格式探针仅 {ok}/5 —— 受限解码通道不可用, 不进批")
        raise SystemExit(2)

    sham_runs, sham_prov = load_reused_sham()
    m_dir = out_root / MODEL
    m_dir.mkdir(parents=True, exist_ok=True)

    runs = list(sham_runs)
    t_m = time.time()
    for cond in CONDS:
        for seed in seeds:
            t_r = time.time()
            r = run_one_ckpt(MODEL, cond, seed, val_tasks, test_tasks, args.gens,
                             args.base_url, m_dir)
            runs.append(r)
            progress(f"b1b {cond}_s{seed} 完成 (test {r['test_final']}/{b1.N_TEST}, "
                     f"tok {r['token_usage']['prompt_tokens']}/"
                     f"{r['token_usage']['completion_tokens']}, "
                     f"wall {r.get('wall_min', round((time.time() - t_r) / 60, 1))} min)")

    verd = verdict_for_model(runs, MODEL, sham_prov)
    verd["format_probe"] = {"parse_ok": ok, "completion_tokens": probe_tok}
    verd["token_usage_batch"] = {
        "llm_calls": sum(r["token_usage"]["llm_calls"] for r in runs if r["cond"] != "sham"),
        "prompt_tokens": sum(r["token_usage"]["prompt_tokens"] for r in runs if r["cond"] != "sham"),
        "completion_tokens": sum(r["token_usage"]["completion_tokens"] for r in runs if r["cond"] != "sham"),
    }
    verd["wall_min"] = round((time.time() - t_m) / 60, 1)
    vpath = out_root / f"tj_{_TAG or 'b1b'}_{MODEL}_verdict.json"
    vpath.write_text(json.dumps(verd, ensure_ascii=False, indent=2), encoding="utf-8")
    progress(f"b1b 完成: verdict = {verd['verdict']}, wall {verd['wall_min']} min -> {vpath.name}")
    print(json.dumps(verd, ensure_ascii=False, indent=2))
    print("WROTE", vpath)


if __name__ == "__main__":
    main()
