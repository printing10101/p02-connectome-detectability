#!/usr/bin/env python
"""实验 G-LLM 正式版: GEP 六阶段自修改在真实 LLM 上的受控验证 (P2.4 正式判决).

与试点 exp_gep_llm.py 的协议差异(其余全部锁同):
  任务集 v2 (2026-09-19 预注册重设计): v1 任务在当前推理栈上朴素作答即 10/12,
  专家策略无用武之地 (G0 闸门失败的根因, 见 g0_probe_v1tasks.json); v2 四族加难
  (五步混合运算 / 9-10 字母移位 / 三位数数字和构造 / 三词反转), 测试集 20 题
  —— 验证集同步扩大, 直接回应 L4 质控计量学发现(8 题套件撑不起验收决策)
  种子: 1/2/3 (试点 2); 参考基因按 v2 任务重调 (验证题调优, 测试题全程隔离)
  任务去重: 同族参数撞车时重抽(有界), 族与生成器不变
判定规则(预注册, 与试点一致):
  DIRECTED_CHANNEL_WINS: directed-naive >= 2 且 naive-sham >= 1
产物: results/exp_gep_llm_formal/ (verdict json + 每 run 不可变 Event 日志, 单 run 即落盘可续跑)
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
OUT = ROOT / "results" / "exp_gep_llm_formal"

BASE_URL = "http://127.0.0.1:1236/v1"  # 1235 已被 gpt-oss-20b 实例占用, Qwen3 正式批改挂 1236
MODEL = "qwen3-30b-a3b"
AUTH = {"Authorization": "Bearer " + os.environ.get("LLAMA_API_KEY", "")}
GENS = 5
N_VAL, N_TEST = 12, 20
SEEDS = (1, 2, 3)


# ---------- 任务集 v2 (预注册重设计 2026-09-19): 4 族加难, 让朴素作答真实变弱 ----------
# v1 任务在 Qwen3-30B 上朴素作答即 10/12, 专家策略无用武之地 (G0 闸门失败的根因)。
# v2: 五步混合运算(3 位数中间值) / 9-10 字母移位 / 三位数数字和构造 / 三词反转。

def _gen_modchain(rng) -> dict:
    x = rng.randint(100, 999)
    a, b, c = rng.randint(11, 89), rng.randint(3, 9), rng.randint(11, 89)
    m, d = rng.choice([7, 9, 11, 13, 17]), rng.randint(2, 9)
    prompt = (f"从整数 {x} 开始：第一步加上 {a}，第二步乘以 {b}，第三步减去 {c}，"
              f"第四步对 {m} 取模，第五步加上 {d}。给出最终数值。")
    y = ((x + a) * b - c) % m + d
    return {"id": "mod", "prompt": prompt, "check": lambda s, y=y: _extract_int(s) == y,
            "answer": str(y)}

def _gen_caesar(rng) -> dict:
    words = ["framework", "lighthouse", "wavelength", "soundstage", "campground",
             "masterpiece", "warehouse", "strawberry", "nightstand", "chalkboard"]
    w = rng.choice(words)
    k = rng.randint(3, 9)
    prompt = (f"将单词 {w} 的每个字母替换为字母表中位置比它小 {k} 的字母（位置从 a=0 数起，"
              f"a 的前面是 z），输出结果单词。")
    y = "".join(chr((ord(c) - 97 - k) % 26 + 97) for c in w)
    return {"id": "caesar", "prompt": prompt, "check": lambda s, y=y: _extract_word(s) == y,
            "answer": y}

def _gen_constrained(rng) -> dict:
    # 三位数版: 各位数字和为 g 的三位数取 5 个 (g=12..24 每档候选数 6~55 个, 保证可解)
    g = rng.choice([12, 15, 18, 21, 24])
    prompt = (f"给出 5 个互不相同的三位数（100-999），严格递增，且每个数的各位数字之和都等于 {g}。"
              f"以逗号分隔输出这 5 个数。")
    cands = [n for n in range(100, 1000) if n // 100 + (n // 10) % 10 + n % 10 == g]
    def check(s, cands=cands):
        nums = _extract_int_list(s)
        return (len(nums) == 5 and len(set(nums)) == 5
                and all(cands[0] <= n <= cands[-1] for n in nums)
                and all(n // 100 + (n // 10) % 10 + n % 10 == g for n in nums)
                and nums == sorted(nums) and all(n in cands for n in nums))
    return {"id": "clist", "prompt": prompt, "check": check, "answer": "cands 子集"}

def _gen_wordswap(rng) -> dict:
    phrases = ["golden harvest", "morning horizon", "silver mountain", "crystal fountain",
               "winter garden", "velvet thunder", "hidden meadow", "ancient harbor",
               "electric orchid", "midnight library", "copper lantern", "frozen riverbed"]
    p = rng.choice(phrases)
    prompt = (f"将短语「{p}」的每个单词的字母顺序反转，然后单词之间用短横线 - 连接输出。")
    y = "-".join(w[::-1] for w in p.split())
    return {"id": "word", "prompt": prompt, "check": lambda s, y=y: _extract_word(s) == y,
            "answer": y}

GENS_FAM = [_gen_modchain, _gen_caesar, _gen_constrained, _gen_wordswap]

def build_tasks() -> tuple[list, list]:
    import random
    rng = random.Random(20260918)
    val, test, seen = [], [], set()
    for i in range(N_VAL + N_TEST):
        fam = GENS_FAM[i % 4]
        for _ in range(50):            # 去重: 同族参数撞车时重抽(有界)
            t = fam(rng)
            if t["prompt"] not in seen:
                break
        seen.add(t["prompt"])
        (val if i < N_VAL else test).append(t)
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


# ---------- 基因 (与试点逐字相同) ----------

GENE_INIT = """### STRATEGY (π)
1. 仔细阅读题目要求。
2. 逐步计算或构造。
3. 检查一遍再回答。
### AVOID (α)
- 不要跳过题目中的任何条件。
### CONSTRAINTS (c)
- 最终答案必须独占一行, 以 ANSWER: 开头。"""

GENE_REF = """### STRATEGY (π)
1. 逐行书面计算, 禁止心算: 总过程严格压缩在 3 行以内, 每行打包一个步骤的全部中间结果。
2. 模运算链: 按题意顺序写成一行链式「x + a = t1 → t1 × b = t2 → t2 − c = t3 → t3 mod m = t4 → t4 + d = 答案」; 每次乘法先估算数量级再精确计算。
3. 字母移位: 只写一行位置对照「字母:位置 → (位置 − k) mod 26 → 新字母」, 全部字母写在一行, 按顺序拼接; 输出前把结果再正向移位回去核对一遍。
4. 数字和构造: 从最小候选开始, 一行内按百位→十位→个位顺序列出所有满足数字和的数（逗号分隔, 逐个验数字和）, 再取最小的 5 个。
5. 单词反转: 每词一行逐字母对位「第1字母→末位, 第2字母→倒数第2位」, 对位完再拼词; 拼完逐字母读一遍核对。
### AVOID (α)
- 禁止心算与跳步; 严禁过程超过 3 行——若步骤导致超行, 压缩书写而不是省略验证。
- 答案行写完后, 必须逐字符重读一遍核对拼写与数字顺序（尤其相邻两个字符是否写反）。
### CONSTRAINTS (c)
- 最终答案必须独占一行, 以 ANSWER: 开头, 只含答案本身。"""


# ---------- LLM 调用 ----------

def chat(messages: list[dict], temperature: float, max_tokens: int = 220) -> str:
    # 超时放宽到 420s: 冷启动时 llama-server 从磁盘 mmap 换入 17GB 专家页, 首查极慢。
    # 重试 30 次 x 30s (共 15 min): 本机环境会周期性 taskkill llama-server (测试编排/run_all.sh),
    # 监督循环负责在 ~10 min 内重启服务, 管线进程必须自己扛过这个窗口才不丢进度。
    last_err = None
    for attempt in range(30):
        try:
            t0 = time.time()
            r = requests.post(
                f"{BASE_URL}/chat/completions",
                headers=AUTH,
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
            last_err = e
            if attempt == 29:
                raise
            if attempt % 5 == 0:
                print(f"  [retry {attempt + 1}/30 after 30s] {type(e).__name__}", flush=True)
            time.sleep(30)
    raise last_err


def eval_gene(gene: str, tasks: list[dict], label: str = "") -> dict:
    """scan/validate 阶段: 贪心解码(温度0)让分数成为基因的确定函数, 验收机制才不被采样噪声污染.
    max_tokens=1280 (试点 320, v1 512): 任务集 v2 的五步链+三位数枚举在 512 内写不完,
    ANSWER 截断 → G0 假阴性。属测量基础设施修正 (L4: max_tokens 必须按最啰嗦的合法策略给足)。"""
    n_ok, fails = 0, []
    for t in tasks:
        out = chat(
            [{"role": "system", "content": gene},
             {"role": "user", "content": t["prompt"] + "\n\n过程不超过 3 行, 最后以 ANSWER: 给出答案。"}],
            temperature=0.0, max_tokens=1280,
        )
        ok = bool(t["check"](out))
        n_ok += ok
        if not ok:
            fails.append({"id": t["id"], "prompt": t["prompt"][:120],
                          "expected": str(t["answer"]), "got": out[-160:]})
    print(f"  [{label}] {n_ok}/{len(tasks)}", flush=True)
    return {"score": n_ok, "fails": fails}


# ---------- 变异算子 (与试点逐字相同) ----------

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


def run_one(cond: str, seed: int, val_tasks, test_tasks, gens: int) -> dict:
    """单 run 即落盘: 已有结果文件则跳过 (崩溃可续跑)."""
    out_file = OUT / f"events_{cond}_s{seed}.json"
    if out_file.exists():
        print("SKIP", out_file.name, flush=True)
        return json.loads(out_file.read_text(encoding="utf-8"))
    r = evolve(cond, seed, val_tasks, test_tasks, gens=gens)
    out_file.write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="冒烟: 1 条件 x 1 种子 x 1 代 x 减半任务")
    ap.add_argument("--smoke-tasks", type=int, default=4)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    val_tasks, test_tasks = build_tasks()
    assert len(val_tasks) == N_VAL and len(test_tasks) == N_TEST
    assert len({t["prompt"] for t in val_tasks + test_tasks}) == N_VAL + N_TEST, "任务去重失败"
    if args.quick:
        val_tasks, test_tasks = val_tasks[:args.smoke_tasks], test_tasks[:args.smoke_tasks]
        conds, seeds, gens = ["gep-naive"], (1,), 1
    else:
        conds, seeds, gens = ["gep-naive", "directed", "sham"], SEEDS, GENS

    # ---- G0: 参考基因必须值钱 (handwire 纪律) ----
    g0_init = eval_gene(GENE_INIT, val_tasks, "G0 init")
    g0_ref = eval_gene(GENE_REF, val_tasks, "G0 ref")
    print(f"G0: gene_init 验证分 {g0_init['score']}/{len(val_tasks)} | "
          f"gene_ref 验证分 {g0_ref['score']}/{len(val_tasks)}", flush=True)

    runs = [run_one(c, s, val_tasks, test_tasks, gens) for c in conds for s in seeds]

    # ---- 汇总 ----
    def mt(cond):
        rs = [r for r in runs if r["cond"] == cond]
        return sum(r["test_final"] for r in rs) / len(rs) if rs else None

    def per_seed(cond):
        return [r["test_final"] for r in runs if r["cond"] == cond]

    summary = {
        "protocol": (
            f"GEP 六阶段(正式版) on llama-server {MODEL}; 任务 4 族 x {N_VAL}验证+{N_TEST}测试(去重); "
            f"{gens}代 x {len(seeds)}种子; 验证分接受; 测试分只测终局; 单 run 落盘可续跑"
        ),
        "G0": {"gene_init_val": g0_init["score"], "gene_ref_val": g0_ref["score"]},
        "test_final_per_cond": {c: mt(c) for c in conds},
        "test_final_per_seed": {c: per_seed(c) for c in conds},
        "val_final_per_cond": {
            c: sum(r["val_final"] for r in runs if r["cond"] == c) /
               max(1, len([r for r in runs if r["cond"] == c])) for c in conds},
        "mutation_sort": mutation_sort(runs),
        "verdict": verdict_of(mt, conds),
        "wall_min": round((time.time() - t0) / 60, 1),
    }
    (OUT / "exp_gep_llm_formal_verdict.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("WROTE", OUT / "exp_gep_llm_formal_verdict.json")


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
