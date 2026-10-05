#!/usr/bin/env python
"""实验 G-LLM: GEP 六阶段自修改在真实 LLM 上的最小受控验证 (P2.4).

载体: 本地 llama-server (Qwen3-30B-A3B, OpenAI 兼容), 遵守 AGENTS.md 本地推理纪律.
基因(Gene, GEP 语义): 紧凑策略文本(π步骤/α避免/c约束), 作为 system 提示拼在每个任务前.
六阶段循环: scan(验证分全评) → signal(失败报告) → intent+mutate(LLM 改写基因并自报粒度)
  → validate(验证分接受: >= 现状才写回) → solidify(Event 不可变日志).
任务: 24 个程序化验证任务(4 族), 12 验证分(选择可见) + 12 测试分(只测终局, 防验证分过拟合).
G0: 先评 gene_init vs gene_ref(专家手工基因=handwire), 证明参考基因值钱才继续.
条件 × 2 种子:
  gep-naive  通用选择: 平凡基因起步, 六阶段自由进化
  directed   定向通道: 参考基因起步(结构由通道给定), 同一循环只做验收调优
  sham       随机突变算子 + 同一验证接受 —— 漂移基线
读数: 测试分终局(directed vs naive vs sham); 变异粒度分拣(P1 预测: 架构级变异的验证增益 ≈ 随机).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "exp_gep_llm"

BASE_URL = "http://127.0.0.1:1235/v1"
MODEL = "qwen3-30b-a3b"
AUTH = {"Authorization": "Bearer " + os.environ.get("LLAMA_API_KEY", "")}
GENS = 5
N_VAL, N_TEST = 8, 8
SEEDS = (1, 2)


# ---------- 任务集: 4 族 × 6, 全部程序化验证 ----------

def _gen_modchain(rng) -> dict:
    x, a, b, m = rng.randint(10, 59), rng.randint(2, 9), rng.randint(2, 5), rng.choice([7, 9, 11, 13])
    prompt = (f"从整数 {x} 开始：第一步加上 {a}，第二步乘以 {b}，第三步对 {m} 取模。"
              f"给出最终数值。")
    y = ((x + a) * b) % m
    return {"id": "mod", "prompt": prompt, "check": lambda s, y=y: _extract_int(s) == y,
            "answer": str(y)}

def _gen_caesar(rng) -> dict:
    words = ["cloud", "stone", "river", "light", "brand", "glass", "metal", "front"]
    w = rng.choice(words)
    k = rng.randint(1, 9)
    prompt = (f"将单词 {w} 的每个字母替换为字母表中位置比它小 {k} 的字母（位置从 a=0 数起，"
              f"a 的前面是 z），输出结果单词。")
    y = "".join(chr((ord(c) - 97 - k) % 26 + 97) for c in w)
    return {"id": "caesar", "prompt": prompt, "check": lambda s, y=y: _extract_word(s) == y,
            "answer": y}

def _gen_constrained(rng) -> dict:
    g = rng.choice([8, 9, 11, 12])
    prompt = (f"给出 4 个互不相同的两位数（10-99），严格递增，且每个数的两位数字之和都等于 {g}。"
              f"以逗号分隔输出这 4 个数。")
    cands = [n for n in range(10, 100) if n // 10 + n % 10 == g]
    def check(s, cands=cands):
        nums = _extract_int_list(s)
        return (len(nums) == 4 and len(set(nums)) == 4
                and all(cands[0] <= n <= cands[-1] for n in nums)
                and all(n // 10 + n % 10 == g for n in nums)
                and nums == sorted(nums) and all(n in cands for n in nums))
    return {"id": "clist", "prompt": prompt, "check": check, "answer": "cands 子集"}

def _gen_wordswap(rng) -> dict:
    words = ["silver path", "morning star", "green field", "quiet ocean"]
    p = rng.choice(words)
    prompt = (f"将短语「{p}」的每个单词的字母顺序反转，然后单词之间用短横线 - 连接输出。")
    y = "-".join(w[::-1] for w in p.split())
    return {"id": "word", "prompt": prompt, "check": lambda s, y=y: _extract_word(s) == y,
            "answer": y}

GENS_FAM = [_gen_modchain, _gen_caesar, _gen_constrained, _gen_wordswap]

def build_tasks() -> tuple[list, list]:
    import random
    rng = random.Random(20260918)
    val, test = [], []
    for i in range(N_VAL + N_TEST):
        fam = GENS_FAM[i % 4]
        (val if i < N_VAL else test).append(fam(rng))
    return val, test


def _extract_int(s: str):
    m = re.findall(r"ANSWER:\s*\$?(-?\d+)", s or "")
    return int(m[-1]) if m else None

def _extract_word(s: str):
    m = re.findall(r"ANSWER:\s*([A-Za-z0-9, \-]+)", s or "")
    if not m:
        return None
    return m[-1].strip().strip(".").replace(" ", "").lower()

def _extract_int_list(s: str):
    m = re.findall(r"ANSWER:\s*([0-9, ]+)", s or "")
    if not m:
        return []
    return [int(t) for t in m[-1].replace(" ", "").split(",") if t]


# ---------- 基因 ----------

GENE_INIT = """### STRATEGY (π)
1. 仔细阅读题目要求。
2. 逐步计算或构造。
3. 检查一遍再回答。
### AVOID (α)
- 不要跳过题目中的任何条件。
### CONSTRAINTS (c)
- 最终答案必须独占一行, 以 ANSWER: 开头。"""

GENE_REF = """### STRATEGY (π)
1. 逐行书面计算, 禁止心算: 每个中间结果必须单独写成一行, 格式为「表达式 = 结果」。
2. 模运算链: 每步计算完立即取模, 格式如「33 × 2 = 66 → 66 mod 13 = 1」; 取模前先估算商再减。
3. 字母移位: 写出位置表, 每个字母一行「字母 位置 → (位置 - k) mod 26 = 新位置 → 新字母」。
4. 数字和构造: 先逐个列出满足数字和的所有两位数并逐个验证数字和, 再从小到大取前 4 个。
5. 单词反转: 逐字母对位写出「第1字母→末位, 第2字母→倒数第2位」, 全部对位后再拼词。
### AVOID (α)
- 禁止心算与跳步; 禁止把中间结果抄错——每行写完立即与上一行核对数字。
- 反转单词时禁止凭感觉抄写, 必须逐字母对位。
### CONSTRAINTS (c)
- 最终答案必须独占一行, 以 ANSWER: 开头, 只含答案本身。"""


# ---------- LLM 调用 ----------

def chat(messages: list[dict], temperature: float, max_tokens: int = 220) -> str:
    # 超时放宽到 420s: 冷启动时 llama-server 从磁盘 mmap 换入 17GB 专家页, 首查极慢
    for attempt in (1, 2):
        try:
            t0 = time.time()
            r = requests.post(
                f"{BASE_URL}/chat/completions",
                headers=AUTH,  # 环境里的 LLAMA_API_KEY 会让 server 启用鉴权
                json={"model": MODEL, "messages": messages,
                      "temperature": temperature, "max_tokens": max_tokens,
                      "cache_prompt": True},
                timeout=420,
            )
            r.raise_for_status()
            out = r.json()["choices"][0]["message"]["content"] or ""
            print(f"    [llm {time.time() - t0:.0f}s]", flush=True)
            return out
        except Exception as e:
            if attempt == 2:
                raise
            print(f"  [retry] {e}", flush=True)
            time.sleep(5)


def eval_gene(gene: str, tasks: list[dict], label: str = "") -> dict:
    """scan/validate 阶段: 贪心解码(温度0)让分数成为基因的确定函数, 验收机制才不被采样噪声污染."""
    n_ok, fails = 0, []
    for t in tasks:
        out = chat(
            [{"role": "system", "content": gene},
             {"role": "user", "content": t["prompt"] + "\n\n过程不超过 3 行, 最后以 ANSWER: 给出答案。"}],
            temperature=0.0, max_tokens=320,
        )
        ok = bool(t["check"](out))
        n_ok += ok
        if not ok:
            fails.append({"id": t["id"], "prompt": t["prompt"][:120],
                          "expected": str(t["answer"]), "got": out[-160:]})
    print(f"  [{label}] {n_ok}/{len(tasks)}", flush=True)
    return {"score": n_ok, "fails": fails}


# ---------- 变异算子 ----------

MUTATE_SYS = """你是策略基因进化器。给定当前基因与失败报告, 产出一个改进版基因。
只输出 JSON: {"mutation_type": "param", "intent": "一句话改动意图", "new_gene": "完整新基因文本"}。
mutation_type 二选一: "param"=只微调现有步骤的措辞/数值/补充提示; "arch"=增删步骤/改变步骤顺序/重组结构。
新基因必须保留三段结构(STRATEGY/AVOID/CONSTRAINTS)与 ANSWER: 输出约束。"""

def mutate_llm(gene: str, signal: dict, rng) -> tuple[str, str, str]:
    # 喂给变异器的失败样本只留 id/期望值, 模型自己的乱输出会带歪 JSON
    slim = [{"id": f["id"], "expected": f["expected"]} for f in signal["fails"][:4]]
    user = (f"当前基因:\n{gene}\n\n最近验证未通过的任务(优先修复这些族):\n"
            f"{json.dumps(slim, ensure_ascii=False)}\n\n产出一个改进版基因。")
    out = chat([{"role": "system", "content": MUTATE_SYS},
                {"role": "user", "content": user}], temperature=0.8, max_tokens=800)
    m = re.search(r"\{.*\}", out, re.S)
    if not m:
        return gene, "none", "PARSE_FAIL:" + out[:150]
    try:
        j = json.loads(m.group(0))
    except json.JSONDecodeError:
        # 宽松兜底: 模型常在字符串里嵌真实换行(非法 JSON), 按字段正则抽取再反转义
        txt = m.group(0)
        mt_ = re.search(r'"mutation_type"\s*:\s*"(param|arch)"', txt)
        ng = re.search(r'"new_gene"\s*:\s*"(.*)"\s*[,}]', txt, re.S)
        if mt_ and ng:
            return (ng.group(1).replace("\\n", "\n").replace('\\"', '"'),
                    mt_.group(1), "lenient")
        return gene, "none", "PARSE_FAIL:" + out[:150]
    return j.get("new_gene", gene), j.get("mutation_type", "param"), j.get("intent", "")

def mutate_sham(gene: str, signal: dict, rng) -> tuple[str, str, str]:
    """程序化随机编辑: 与 LLM 变异同层的随机扰动, 不含任何失败信息."""
    lines = gene.splitlines()
    pi_idx = [i for i, l in enumerate(lines) if re.match(r"^\d+\. ", l)]
    if len(pi_idx) >= 2 and rng.random() < 0.5:
        i, j = rng.sample(pi_idx, 2)
        lines[i], lines[j] = lines[j], lines[i]
        why = "交换两行步骤"
    elif pi_idx:
        del lines[pi_idx[-1]]
        why = "删除最后一步"
    else:
        lines.append("4. 再检查一遍。")
        why = "追加一步"
    return "\n".join(lines), "arch", f"sham:{why}"


# ---------- 六阶段进化 ----------

def evolve(cond: str, seed: int, val_tasks, test_tasks, gens: int = GENS) -> dict:
    rng = __import__("random").Random(seed * 77 + 1)
    gene = {"gep-naive": GENE_INIT, "directed": GENE_REF, "sham": GENE_INIT}[cond]
    events, cur = [], eval_gene(gene, val_tasks, f"{cond} s{seed} init")
    for gen in range(gens):
        signal = {"fails": cur["fails"]}
        if cond == "sham":
            new_gene, mtype, intent = mutate_sham(gene, signal, rng)
        else:
            new_gene, mtype, intent = mutate_llm(gene, signal, rng)
        new_res = eval_gene(new_gene, val_tasks, f"{cond} s{seed} g{gen}")          # validate
        accept = new_res["score"] >= cur["score"]          # 验证分不退步才写回
        ev = {"seed": seed, "cond": cond, "gen": gen,
              "mutation_type": mtype, "intent": intent,
              "score_old": cur["score"], "score_new": new_res["score"],
              "accepted": accept}
        events.append(ev)
        print(f"  [{cond} s{seed}] g{gen}: {cur['score']}->{new_res['score']} "
              f"({mtype}, {'接受' if accept else '拒绝'})", flush=True)
        if accept:
            gene, cur = new_gene, new_res
    test = eval_gene(gene, test_tasks, f"{cond} s{seed} TEST")
    return {"cond": cond, "seed": seed, "val_final": cur["score"],
            "test_final": test["score"], "events": events, "final_gene": gene}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="冒烟: 1 种子 x 2 代 x 1 条件")
    args = ap.parse_args()
    gens, seeds = (2, (1,)) if args.quick else (GENS, SEEDS)

    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    val_tasks, test_tasks = build_tasks()
    if args.quick:  # 冒烟再砍一半任务
        val_tasks, test_tasks = val_tasks[:4], test_tasks[:4]

    # ---- G0: 参考基因必须值钱 (handwire 纪律) ----
    g0_init, g0_ref = eval_gene(GENE_INIT, val_tasks, "G0 init"), eval_gene(GENE_REF, val_tasks, "G0 ref")
    print(f"G0: gene_init 验证分 {g0_init['score']}/{N_VAL} | gene_ref 验证分 {g0_ref['score']}/{N_VAL}",
          flush=True)

    runs = []
    conds = ["gep-naive"] if args.quick else ["gep-naive", "directed", "sham"]
    for cond in conds:
        for seed in seeds:
            runs.append(evolve(cond, seed, val_tasks, test_tasks, gens=gens))

    # ---- 汇总 ----
    def mt(cond):
        rs = [r for r in runs if r["cond"] == cond]
        return float(np_mean([r["test_final"] for r in rs])) if rs else None
    summary = {
        "protocol": (
            f"GEP 六阶段 on llama-server {MODEL}; 任务 4 族 x {N_VAL}+{N_TEST}; "
            f"{gens}代 x {len(seeds)}种子; 验证分接受; 测试分只测终局"
        ),
        "G0": {"gene_init_val": g0_init["score"], "gene_ref_val": g0_ref["score"]},
        "test_final_per_cond": {c: mt(c) for c in conds},
        "mutation_sort": mutation_sort(runs),
        "verdict": verdict_of(mt, conds),
        "wall_min": round((time.time() - t0) / 60, 1),
    }
    (OUT / "exp_gep_llm_verdict.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    for r in runs:  # Event 不可变日志
        (OUT / f"events_{r['cond']}_s{r['seed']}.json").write_text(
            json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("WROTE", OUT / "exp_gep_llm_verdict.json")


def np_mean(xs):
    return sum(xs) / len(xs)

def mutation_sort(runs) -> dict:
    """P1 预测: 架构级变异的验证增益不高于参数级."""
    acc = {"param": [], "arch": []}
    for r in runs:
        for ev in r["events"]:
            if ev["accepted"] and ev["mutation_type"] in acc:
                acc[ev["mutation_type"]].append(ev["score_new"] - ev["score_old"])
    return {
        "param": {"n_accepted": len(acc["param"]),
                  "mean_gain": np_mean(acc["param"]) if acc["param"] else None},
        "arch": {"n_accepted": len(acc["arch"]),
                 "mean_gain": np_mean(acc["arch"]) if acc["arch"] else None},
    }

def verdict_of(mt, conds) -> str:
    n, d, s = mt("gep-naive"), mt("directed"), mt("sham")
    if None in (n, d, s):
        return "INCOMPLETE"
    if d - n >= 2 and n - s >= 1:
        return "DIRECTED_CHANNEL_WINS"
    if d - n >= 2:
        return "DIRECTED_AHEAD_NO_SHAM_GAP"
    return "NO_CLEAR_DIRECTED_ADVANTAGE"


if __name__ == "__main__":
    main()
