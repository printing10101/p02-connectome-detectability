#!/usr/bin/env python
"""TJ-B1 真实 LLM 三臂自修改实验 (冻结判据 TJ_B1_llm_three_arm).

管线复制 exp_gep_llm_formal.py (任务集 v2 / GENE_INIT / GENE_REF / 六阶段进化, 其余锁同),
按 B1 冻结设计改造:
  三臂语义 (naive 参照 exp_g_transfer L137-143 的"无先验自由进化"分支):
    directed = 专家参考基因起步 (GENE_REF) + LLM 自报变异 mutate_llm —— 定向通道
    sham     = 朴素基因起步 (GENE_INIT) + 程序化随机编辑 mutate_sham —— 不含失败信息
    naive    = 朴素基因起步 (GENE_INIT) + LLM 自报变异 mutate_llm —— 朴素进化
  模型/端口参数化 (默认走 E:\\llama-cpp\\model-proxy.js 的 8080, 不另起独立实例 ——
  代理 killUpstream 按映像名 taskkill 全部 llama-server.exe, 独立实例必被误杀);
  SEEDS 1-10 × GENS 10; 每 run 落盘 (含 usage 字段累计的 token 用量, 断点续跑跳过已有 run);
  自报变异类型 vs 程序化分类的一致性逐次记录 (冻结阈值 0.2, 超过判 directed 无效)。

判决 (TJ1-预注册判据-frozen-v1.0.json, 本脚本不改动其任何阈值):
  B1_DIRECTED_WINS : directed−sham 配对 95% CI 排除 0 且 >=7/10 种子同向
  B1_EQUIV         : TOST 等价成立 (等价边界 delta=1.0/20 题, 与试点"最小有意义差"同源)
  B1_NOISE         : 种子方差淹没臂方差 (MS_seed > 10x MS_arm, 操作化定义留痕 verdict JSON)
  自报误分率 > 0.2 : 判 directed 无效, verdict=B1_SELF_REPORT_INVALID (其余判决照算留痕)
授权: TJ1-B1执行授权.md (2026-09-22, Qwen3 系列 + token 不设上限)。
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import requests
from scipy import stats as sps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flyevo.tj_criteria import load as load_criteria, sha256_of_json

KEY = "TJ_B1_llm_three_arm"
CRIT = load_criteria(KEY)
MISCLASS_MAX = float(CRIT["self_report_misclassification_rate_max"])

OUT = ROOT / "results" / "tj_b1_llm"
PROGRESS_MD = ROOT.parent / "TJ1_batch_progress.md"

BASE_URL = "http://127.0.0.1:8080/v1"  # model-proxy.js; 上游 llama-server 在 8081
API_KEY = os.environ.get("LLAMA_API_KEY") or os.environ.get("LLAMA_LOCAL_API_KEY") or ""
if not API_KEY:
    try:
        API_KEY = Path(r"E:\llama-cpp\.api-key").read_text(encoding="utf-8").splitlines()[0].strip()
    except OSError:
        API_KEY = ""
AUTH = {"Authorization": "Bearer " + API_KEY}

GENS = 10
N_VAL, N_TEST = 12, 20
SEEDS = tuple(range(1, 11))
CONDS = ("directed", "sham", "naive")  # 判决先要 directed/sham, naive 垫后
# 等价边界: 20 题测试分上的 1 题 —— 试点判决规则里"最小有意义差"(naive-sham>=1)的同源取值。
# 冻结 JSON 未固定该阈值, 此为操作化定义, 与其余判决常数一并留痕。
TOST_DELTA = 1.0
NOISE_MS_RATIO = 10.0  # B1_NOISE 操作化: 种子均方 > 10 倍臂均方


# ---------- 任务集 v2 (与 exp_gep_llm_formal.py 逐字锁同) ----------

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
    # B1 启动前缺陷修复 (2026-09-22, 留痕 TJ1-B1执行授权.md): 原正式版 g∈{12,15,18,21,24}
    # 仅 5 个取值, 而 32 题中 clist 族抽 8 题 —— prompt 只由 g 决定, 去重断言鸽笼必炸,
    # 正式批在原码下永远无法启动。扩为 12..24 全档 (同一难度带), 家族语义与难度不变。
    g = rng.randint(12, 24)
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
        for _ in range(50):
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


# ---------- 基因 (与 exp_gep_llm_formal.py 逐字锁同) ----------

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


# ---------- token 计量 ----------

class TokenMeter:
    """从响应 usage 字段累计 prompt/completion tokens 与调用次数, 每 run 重置."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def add(self, usage: dict):
        self.calls += 1
        self.prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
        self.completion_tokens += int(usage.get("completion_tokens", 0) or 0)

    def snapshot(self) -> dict:
        return {"llm_calls": self.calls, "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens}


# ---------- LLM 调用 ----------

def chat(messages: list[dict], temperature: float, max_tokens: int, meter: TokenMeter,
         model: str, base_url: str = BASE_URL) -> str:
    # 超时/重试沿用 exp_gep_llm_formal: 冷加载 17GB 专家页极慢, 且代理会按需换模型,
    # 在途请求可能被 kill —— 30 次 x 30s 的重试窗口足以扛过冷加载与误杀重启。
    # chat_template_kwargs 对非思考模型 (30B-Instruct-2507) 被模板忽略, 对混合思考模型
    # (Qwen3-14B) 关闭思考, 防止 1280 token 预算被 <think> 吃光导致 ANSWER 截断假阴性。
    last_err = None
    for attempt in range(30):
        try:
            t0 = time.time()
            r = requests.post(
                f"{base_url}/chat/completions",
                headers=AUTH,
                json={"model": model, "messages": messages,
                      "temperature": temperature, "max_tokens": max_tokens,
                      "cache_prompt": True, "chat_template_kwargs": {"enable_thinking": False}},
                timeout=420,
            )
            r.raise_for_status()
            body = r.json()
            meter.add(body.get("usage", {}))
            out = body["choices"][0]["message"]["content"] or ""
            # 测量归一化: 思考模型万一仍输出 <think> 块, 判分只看其后的正文;
            # 对三臂一视同仁, 不引入方向性偏差。
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


def eval_gene(gene: str, tasks: list[dict], label: str, meter: TokenMeter,
              model: str, base_url: str = BASE_URL) -> dict:
    """scan/validate: 贪心解码(温度0)让分数成为基因的确定函数 (与正式版锁同)."""
    n_ok, fails = 0, []
    for t in tasks:
        out = chat(
            [{"role": "system", "content": gene},
             {"role": "user", "content": t["prompt"] + "\n\n过程不超过 3 行, 最后以 ANSWER: 给出答案。"}],
            temperature=0.0, max_tokens=1280, meter=meter, model=model, base_url=base_url,
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

def mutate_llm(gene: str, signal: dict, meter: TokenMeter, model: str,
               base_url: str = BASE_URL) -> tuple[str, str, str, bool]:
    slim = [{"id": f["id"], "expected": f["expected"]} for f in signal["fails"][:4]]
    user = (f"当前基因:\n{gene}\n\n最近验证未通过的任务(优先修复这些族):\n"
            f"{json.dumps(slim, ensure_ascii=False)}\n\n产出一个改进版基因。")
    out = chat([{"role": "system", "content": MUTATE_SYS},
                {"role": "user", "content": user}], temperature=0.8, max_tokens=800,
               meter=meter, model=model, base_url=base_url)
    m = re.search(r"\{.*\}", out, re.S)
    if not m:
        return gene, "none", "PARSE_FAIL:" + out[:150], False
    try:
        j = json.loads(m.group(0))
    except json.JSONDecodeError:
        txt = m.group(0)
        mt_ = re.search(r'"mutation_type"\s*:\s*"(param|arch)"', txt)
        ng = re.search(r'"new_gene"\s*:\s*"(.*)"\s*[,}]', txt, re.S)
        if mt_ and ng:
            return (ng.group(1).replace("\\n", "\n").replace('\\"', '"'),
                    mt_.group(1), "lenient", True)
        return gene, "none", "PARSE_FAIL:" + out[:150], False
    return (j.get("new_gene", gene), j.get("mutation_type", "param"),
            j.get("intent", ""), True)

def mutate_sham(gene: str, signal: dict, rng) -> tuple[str, str, str]:
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


def _steps_of(gene: str) -> list[str]:
    return [re.sub(r"^\d+\. ", "", l).strip()
            for l in gene.splitlines() if re.match(r"^\d+\. ", l)]

def classify_edit(old_gene: str, new_gene: str) -> str:
    """程序化 param/arch 判别 (自报一致性的对照面; 操作化定义, 与冻结阈值配套).

    arch: 步骤数量变化 / 步骤顺序变化 / 同位步骤被实质性改写 (相似度 < 0.6);
    param: 步骤数量与顺序不变, 且每个变化的步骤与旧步骤高度相似 (>= 0.6,
    difflib.SequenceMatcher, 允许措辞/数值微调与步内补充)。
    sham 的三种编辑(交换/删除/追加)全部落在 arch 侧, 分类器对 sham 无歧义。
    """
    so, sn = _steps_of(old_gene), _steps_of(new_gene)
    if len(so) != len(sn):
        return "arch"  # 增删步骤
    if sorted(so) == sorted(sn):
        # 多重集相等: 步骤集合未变 —— 逐位相同为无结构变化, 否则是换序
        return "param" if so == sn else "arch"
    sims = [difflib.SequenceMatcher(None, a, b).ratio() for a, b in zip(so, sn)]
    return "param" if min(sims) >= 0.6 else "arch"


# ---------- 六阶段进化 (与 exp_gep_llm_formal.py 锁同, 臂语义按 B1) ----------

def evolve(cond: str, seed: int, val_tasks, test_tasks, gens: int,
           model: str, base_url: str, meter: TokenMeter) -> dict:
    rng = __import__("random").Random(seed * 77 + 1)
    # naive = 朴素基因起步 (exp_g_transfer 的 naive 分支: 无先验, 自由进化); directed = 参考基因起步
    gene = {"naive": GENE_INIT, "directed": GENE_REF, "sham": GENE_INIT}[cond]
    events, cur = [], eval_gene(gene, val_tasks, f"{cond} s{seed} init", meter, model, base_url)
    for gen in range(gens):
        signal = {"fails": cur["fails"]}
        if cond == "sham":
            new_gene, mtype_self, intent = mutate_sham(gene, signal, rng)
            parse_ok = True  # 程序化编辑无解析失败概念, 且不进入误分率分母
        else:
            new_gene, mtype_self, intent, parse_ok = mutate_llm(
                gene, signal, meter, model, base_url)
        new_res = eval_gene(new_gene, val_tasks, f"{cond} s{seed} g{gen}", meter, model, base_url)
        accept = new_res["score"] >= cur["score"]
        ev = {"seed": seed, "cond": cond, "gen": gen,
              "mutation_type": mtype_self, "intent": intent,
              "score_old": cur["score"], "score_new": new_res["score"],
              "accepted": accept}
        if cond in ("directed", "naive"):
            mtype_prog = classify_edit(gene, new_gene)
            ev["mutation_type_prog"] = mtype_prog
            ev["self_report_parse_ok"] = parse_ok
            ev["misclassified"] = bool(parse_ok and mtype_self in ("param", "arch")
                                       and mtype_self != mtype_prog)
        events.append(ev)
        print(f"  [{cond} s{seed}] g{gen}: {cur['score']}->{new_res['score']} "
              f"({mtype_self}, {'接受' if accept else '拒绝'})", flush=True)
        if accept:
            gene, cur = new_gene, new_res
    test = eval_gene(gene, test_tasks, f"{cond} s{seed} TEST", meter, model, base_url)
    rec = {"cond": cond, "seed": seed, "val_final": cur["score"],
           "test_final": test["score"], "events": events, "final_gene": gene,
           "token_usage": meter.snapshot()}
    return rec


def run_one(model_id: str, cond: str, seed: int, val_tasks, test_tasks, gens: int,
            base_url: str, out_dir: Path) -> dict:
    """单 run 即落盘: 已有结果文件则跳过 (崩溃/中断可续跑)."""
    out_file = out_dir / f"events_{cond}_s{seed}.json"
    if out_file.exists():
        print("SKIP", model_id, out_file.name, flush=True)
        return json.loads(out_file.read_text(encoding="utf-8"))
    meter = TokenMeter()
    t0 = time.time()
    r = evolve(cond, seed, val_tasks, test_tasks, gens, model_id, base_url, meter)
    r["model"] = model_id
    r["wall_min"] = round((time.time() - t0) / 60, 1)
    out_file.write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  WROTE {out_file.name} (test {r['test_final']}/{len(test_tasks)}, "
          f"tok {meter.prompt_tokens}/{meter.completion_tokens}, {r['wall_min']} min)", flush=True)
    return r


# ---------- 冻结判据判决 ----------

def misclassification_rate(runs: list[dict], cond: str) -> dict:
    """自报 mutation_type vs 程序化分类的不一致率 (仅统计自报解析成功的变异).

    分母含 naive 臂对照; 冻结阈值只约束 directed (超 0.2 判 directed 无效)。
    """
    n = bad = parse_fail = 0
    for r in runs:
        if r["cond"] != cond:
            continue
        for ev in r["events"]:
            if "mutation_type_prog" not in ev:
                continue
            if not ev.get("self_report_parse_ok", False):
                parse_fail += 1
                continue
            n += 1
            bad += 1 if ev.get("misclassified") else 0
    return {"n_self_reported": n, "n_parse_fail": parse_fail,
            "n_misclassified": bad, "rate": (bad / n) if n else None}


def _paired_ci(diffs: list[float]) -> dict:
    n = len(diffs)
    mean = sum(diffs) / n
    sd = float(np.std(diffs, ddof=1))
    sem = sd / (n ** 0.5)
    if sem == 0:  # 10 种子差值全同的退化情形: CI 塌缩为点
        return {"mean": mean, "sd": sd, "ci95": [mean, mean],
                "ci_excludes_zero": mean != 0}
    lo, hi = sps.t.interval(0.95, df=n - 1, loc=mean, scale=sem)
    return {"mean": mean, "sd": sd, "ci95": [float(lo), float(hi)],
            "ci_excludes_zero": bool(lo > 0 or hi < 0)}

def _tost_equivalent(diffs: list[float], delta: float) -> dict:
    """双单侧 t 检验 (TOST): H01 mu<=-delta vs H02 mu>=delta, 两p均<0.05 判等价."""
    n = len(diffs)
    mean = sum(diffs) / n
    sem = float(np.std(diffs, ddof=1)) / (n ** 0.5)
    if sem == 0:  # 退化: 无采样噪声, 等价与否直接看均值与边界的距离
        return {"delta": delta, "p1_gt_minus_delta": 0.0 if mean > -delta else 1.0,
                "p2_lt_delta": 0.0 if mean < delta else 1.0,
                "equivalent_at_0.05": abs(mean) < delta}
    p1 = float(sps.t.sf((mean + delta) / sem, df=n - 1))  # mu > -delta
    p2 = float(sps.t.cdf((mean - delta) / sem, df=n - 1))  # mu < delta
    return {"delta": delta, "p1_gt_minus_delta": p1, "p2_lt_delta": p2,
            "equivalent_at_0.05": bool(max(p1, p2) < 0.05)}

def _variance_components(vals_by_seed: dict[int, dict[str, float]]) -> dict:
    """单因素方差分解 (seed x 臂 值, vals_by_seed[seed][arm]): 种子均方 vs 臂均方."""
    arms = sorted({a for row in vals_by_seed.values() for a in row})
    seeds = sorted(vals_by_seed)
    if len(arms) < 2 or len(seeds) < 2:  # 自由度为 0 的退化 (冒烟等): 比值记无穷, 判读另走
        return {"ms_seed": float("nan"), "ms_arm": float("nan"),
                "seed_over_arm_ratio": float("inf")}
    grand = sum(v for row in vals_by_seed.values() for v in row.values()) / (len(seeds) * len(arms))
    ss_seed = sum((sum(row.values()) / len(arms) - grand) ** 2 for row in vals_by_seed.values()) * len(arms)
    ss_arm = sum(
        (sum(vals_by_seed[s][a] for s in seeds) / len(seeds) - grand) ** 2
        for a in arms) * len(seeds)
    ms_seed = ss_seed / (len(seeds) - 1)
    ms_arm = ss_arm / (len(arms) - 1)
    return {"ms_seed": ms_seed, "ms_arm": ms_arm,
            "seed_over_arm_ratio": ms_seed / ms_arm if ms_arm else float("inf")}


def verdict_for_model(runs: list[dict], model_id: str) -> dict:
    """按冻结判据出判决; 阈值未 frozen 的操作化常数在字段中留痕."""
    per = {c: {r["seed"]: r["test_final"] for r in runs if r["cond"] == c} for c in CONDS}
    need = sorted({r["seed"] for r in runs})
    complete = len(need) >= 2 and all(all(s in per[c] for s in need) for c in CONDS)
    # >=2 种子才进统计: 单种子下 CI/TOST/方差分解全部无定义, 如实报 INCOMPLETE
    if not complete:
        return {"model": model_id, "verdict": "INCOMPLETE",
                "n_runs": len(runs), "expected_runs": len(need) * len(CONDS)}

    diffs = [per["directed"][s] - per["sham"][s] for s in need]
    ci = _paired_ci(diffs)
    n_pos = sum(1 for d in diffs if d > 0)
    tost = _tost_equivalent(diffs, TOST_DELTA)
    varcomp = _variance_components({s: {c: per[c][s] for c in CONDS} for s in need})
    mis_d = misclassification_rate(runs, "directed")
    mis_n = misclassification_rate(runs, "naive")

    self_report_valid = not (mis_d["rate"] is not None and mis_d["rate"] > MISCLASS_MAX)

    if ci["ci_excludes_zero"] and n_pos >= 7:
        core = "B1_DIRECTED_WINS"
    elif tost["equivalent_at_0.05"]:
        core = "B1_EQUIV"
    elif varcomp["seed_over_arm_ratio"] > NOISE_MS_RATIO:
        core = "B1_NOISE"
    else:
        core = "B1_INCONCLUSIVE"

    return {
        "model": model_id,
        "experiment": KEY,
        "criteria": CRIT,
        "criteria_sha256": sha256_of_json(),
        "operationalization": {
            "tost_delta": TOST_DELTA,
            "tost_note": "20 题测试分的 1 题, 与试点规则『最小有意义差 naive-sham>=1』同源; 冻结 JSON 未固定",
            "noise_ms_ratio": NOISE_MS_RATIO,
            "noise_note": "种子均方 > 10x 臂均方; 冻结 JSON 未固定",
        },
        "directed_minus_sham": {**ci, "n_seeds_positive": n_pos},
        "tost": tost,
        "variance_components": varcomp,
        "self_report_misclassification": {"directed": mis_d, "naive": mis_n,
                                          "frozen_max": MISCLASS_MAX,
                                          "directed_valid": self_report_valid},
        "test_final_per_cond": {c: [per[c][s] for s in need] for c in CONDS},
        "verdict": ("B1_SELF_REPORT_INVALID|" + core) if not self_report_valid else core,
        "verdict_if_valid": core,
    }


# ---------- 进度留痕 ----------

SMOKE = False  # 冒烟模式只打印进度, 不写入正式 batch 进度文件

def progress(msg: str):
    line = f"- [{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    if SMOKE:
        return
    try:
        with open(PROGRESS_MD, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError as e:
        print(f"[progress write failed: {e}]", flush=True)


def np_mean(xs):
    return sum(xs) / len(xs)


def mutation_sort(runs) -> dict:
    """P1 预测: 架构级变异的验证增益不高于参数级 (与试点口径一致, 按 LLM 自报)."""
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


# ---------- 主流程 ----------

def main() -> None:
    global N_VAL, N_TEST, GENS, SMOKE
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="qwen3-instruct-30b,qwen3-14b")
    ap.add_argument("--base-url", default=BASE_URL)
    ap.add_argument("--seeds", default="1-10", help="如 1-10 或 1,2,3")
    ap.add_argument("--gens", type=int, default=GENS)
    ap.add_argument("--smoke", action="store_true", help="冒烟: 首模型 x 1 种子 x 2 代 x 4+4 题")
    ap.add_argument("--out-dir", default=None, help="覆盖输出目录 (冒烟隔离用)")
    args = ap.parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    if args.seeds.count("-") and "," not in args.seeds:
        a, b = args.seeds.split("-")
        seeds = tuple(range(int(a), int(b) + 1))
    else:
        seeds = tuple(int(t) for t in args.seeds.split(","))

    custom_out = args.out_dir is not None
    out_root = Path(args.out_dir) if custom_out else OUT
    if args.smoke:
        global SEEDS
        SMOKE = True
        seeds, args.gens = (1,), 2
        N_VAL = N_TEST = 4
        if not custom_out:
            out_root = out_root.parent / (out_root.name + "_smoke")
    out_root.mkdir(parents=True, exist_ok=True)

    val_tasks, test_tasks = build_tasks()
    assert len(val_tasks) == N_VAL and len(test_tasks) == N_TEST
    assert len({t["prompt"] for t in val_tasks + test_tasks}) == N_VAL + N_TEST, "任务去重失败"

    progress(f"b1 开始 (models={','.join(models)}; 3臂 {len(seeds)}种子 x {args.gens}代; "
             f"任务 {N_VAL}验证+{N_TEST}测试; endpoint {args.base_url}; 冒烟={args.smoke})")
    t_all = time.time()
    model_verdicts = {}

    for model_id in models:
        m_dir = out_root / model_id
        m_dir.mkdir(parents=True, exist_ok=True)
        progress(f"b1 {model_id} G0 闸门 + 3 臂批开始")
        t_m = time.time()

        # ---- G0: 参考基因必须值钱 (handwire 纪律; 记录不阻断, 冻结判据未含 G0) ----
        meter = TokenMeter()
        g0_init = eval_gene(GENE_INIT, val_tasks, f"G0 init [{model_id}]", meter, model_id, args.base_url)
        g0_ref = eval_gene(GENE_REF, val_tasks, f"G0 ref [{model_id}]", meter, model_id, args.base_url)
        g0_warn = g0_ref["score"] <= g0_init["score"]
        print(f"G0 [{model_id}]: gene_init {g0_init['score']}/{N_VAL} | "
              f"gene_ref {g0_ref['score']}/{N_VAL}"
              + ("  << 警告: 参考基因不优于朴素基因, directed 通道存疑" if g0_warn else ""), flush=True)

        runs = []
        for cond in CONDS:
            for seed in seeds:
                t_r = time.time()
                r = run_one(model_id, cond, seed, val_tasks, test_tasks, args.gens,
                            args.base_url, m_dir)
                runs.append(r)
                progress(f"b1 {model_id} {cond}_s{seed} 完成 (test {r['test_final']}/{N_TEST}, "
                         f"tok {r['token_usage']['prompt_tokens']}/{r['token_usage']['completion_tokens']}, "
                         f"wall {r.get('wall_min', round((time.time() - t_r) / 60, 1))} min)")

        verd = verdict_for_model(runs, model_id)
        verd["G0"] = {"gene_init_val": g0_init["score"], "gene_ref_val": g0_ref["score"],
                      "ref_not_better_than_init": g0_warn, "token_usage": meter.snapshot()}
        verd["mutation_sort"] = mutation_sort(runs)
        verd["token_usage_total"] = {
            "llm_calls": sum(r["token_usage"]["llm_calls"] for r in runs),
            "prompt_tokens": sum(r["token_usage"]["prompt_tokens"] for r in runs),
            "completion_tokens": sum(r["token_usage"]["completion_tokens"] for r in runs),
        }
        verd["wall_min"] = round((time.time() - t_m) / 60, 1)
        vpath = m_dir / f"tj_b1_{model_id}_verdict.json"
        vpath.write_text(json.dumps(verd, ensure_ascii=False, indent=2), encoding="utf-8")
        model_verdicts[model_id] = verd
        progress(f"b1 {model_id} 完成: {len(runs)}/{len(seeds)*len(CONDS)} run, "
                 f"verdict = {verd['verdict']}, wall {verd['wall_min']} min -> {vpath.name}")

    summary = {
        "experiment": KEY,
        "protocol": (
            f"TJ-B1 三臂自修改 on llama-server (proxy {args.base_url}); 任务集 v2 4 族 x "
            f"{N_VAL}验证+{N_TEST}测试; {args.gens}代 x {len(seeds)}种子 x 3臂(directed/sham/naive); "
            "验证分接受; 测试只测终局; 单 run 落盘可续跑; token 用量按 usage 字段逐次累计"
        ),
        "criteria_key": KEY,
        "criteria_sha256": sha256_of_json(),
        "authorization": "TJ1-B1执行授权.md (2026-09-22: Qwen3 系列, token 预算不设上限)",
        "models": models,
        "model_verdicts": model_verdicts,
        "wall_min": round((time.time() - t_all) / 60, 1),
    }
    spath = out_root / "tj_b1_llm_verdict.json"
    spath.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    progress(f"b1 全批完成: verdicts = "
             f"{ {m: v['verdict'] for m, v in model_verdicts.items()} }, "
             f"wall {summary['wall_min']} min -> {spath.name}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("WROTE", spath)


if __name__ == "__main__":
    main()
