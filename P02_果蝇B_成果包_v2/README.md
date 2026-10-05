# P02 成果包 v2(《Architecture Search in Self-Modifying Systems》工件包)

本包是论文 P02(规模衰减规律与随机对照审计纪律)的可审计工件快照。
每个文件均为实验产物的原样拷贝,未做任何编辑;顶层 `MANIFEST.sha256` 记录全部文件的 SHA256,
发布至 Zenodo/OSF 后请在论文数据可用性声明中回填 DOI,并保证哈希清单随行。

- 快照日期:2026-09-29
- 对应稿件:论文/P02_果蝇B-规模规律与审计纪律.docx(v2 修订版)
- 判据冻结 JSON 的 SHA256:`b2b05c0d96c6136767c984829f34ab7b971c4470c632b3055f5c5a63d9872939`

## 目录结构

```
code/
  scripts/            实验脚本(技能图平台调用、规模扫描、三因子、GEP 管线、三臂批)
  flyevo/             数字果蝇/技能图平台 Python 包(含 tj_criteria.py 判据实现)
  requirements.txt    依赖清单
  run_gep_formal_overnight.sh   正式管线过夜运行入口
results/
  exp_g_transfer/                 实验 G(五条件 × 5 种子 × 500 代)判决
  exp_scale_curve_v2/             规模扫描 v2 协议(种子 1–3)
  exp_scale_curve_v2_seeds456/    事后扩编批次(种子 4–6,判据协议不变,合并报告)
  exp_scale_curve_v2_PREFIX_rngdrift/   审计留痕:被硬自检捕获的跨规模 RNG 漂移缺陷版本
  exp_scale_curve_v2_POISONED_quickresume/ 审计留痕:快速续跑污染对照
  exp_gens_mut_scale/             三因子扫描(81 设计单元 / 63 实跑 × 800 代)
  exp_gens_mut_scale_PREFIX_rngdrift/   同上的 RNG 漂移审计留痕
  exp_gep_llm/                    LLM 六阶段试点(3 条件 × 2 种子 × 5 代)
  exp_gep_llm_formal/             正式化验证(G0 探针 + S1–S4 管线验证 + state.json)
  tj_b1_llm/                      预注册三臂判决批:两档 × 3 臂 × 10 种子 × 10 代
prereg/
  TJ1-预注册判据-frozen-v1.0.json  冻结判据(2026-09-22,含 B1 全部判据与全局纪律)
  TJ1-B1执行授权.md                书面授权、模型选型全记录、缺陷修复留痕、冻结 vs 实际对照
  TJ1-停机存档与续跑指南.md
  TJ1_batch_progress.md           批次进度日志
  TJ1-成果快照-2026-09-24.md
```

## 论文主张 → 工件对照(核查指南)

| 论文位置 | 主张 / 读数 | 核查文件 |
|---|---|---|
| §3.1 表 1 | 五条件终局 15.6/9.6/9.2/6.2/4.2;层消融 2.2/1.8;Gated 省 8% | `results/exp_g_transfer/exp_g_verdict.json` |
| §3.1 | G0 锁钥结构 1/1/1/16;三条预注册检查通过 | 同上(`G0_handwire`、`checks`) |
| §3.2 图 1b | sham 开边 17.4 / 命中 6.4;naive 16.0 / 8.2 | 同上(`mask_stats`,逐种子在列) |
| §3.3 表 2 | 达成率 1.000→0.167;sham 地板 0/48;诱饵边 0.6→46.8 | `results/exp_scale_curve_v2/` + `..._seeds456/`(合并即正文 12 run/格) |
| §3.3 | v1 指标退役与 RNG 漂移捕获 | `exp_scale_curve_v2_PREFIX_rngdrift/`、`..._POISONED_quickresume/` |
| §3.4 图 3 | 差值 −3.00/−0.33/0.00 等;保留率 9 格全正 | `results/exp_gens_mut_scale/gens_mut_scale_verdict.json` |
| §3.5 表 3 | S1 4=4;S2 margin −1;S3 climb 0、解析 4/4;S4 187 min | `results/exp_gep_llm_formal/validate/validate_verdict.json`、`g0_probe_v1tasks.json`、`validate/state.json` |
| §3.5 表 5 | 逐臂记账(提案/解析/接受/拒绝/增益) | `results/tj_b1_llm/qwen3-*/events_*.json`(60 份,逐代逐事件)+ 两档 verdict |
| §3.5 | 14B directed−sham=+4.9,CI [2.76,7.04],9/10;30B −0.4,CI 含零,3/10;TOST;方差分解 | `results/tj_b1_llm/tj_b1_llm_verdict.json` |
| §3.5 | 30B directed 解析失败 42/100;误分 1/58;naive 误分 60/93 | 同上(`self_report_misclassification`)+ 事件文件 `self_report_parse_ok` 字段 |
| §3.5 | LOO 10/10 维持 WINS | 由上述 events 重算(脚本见论文仓库 analysis/) |
| 附录 A 表 6 | 全部偏差与留痕 | `prereg/TJ1-B1执行授权.md`(模型选型、缺陷修复、冻结 vs 实际对照表) |

## 复现入口

1. `pip install -r code/requirements.txt`
2. 离散基底:`python code/scripts/exp_g_transfer.py`、`exp_scale_curve_v2.py`、`exp_gens_mut_scale.py`
3. LLM 管线:本地 llama-server + 代理链(见 `prereg/TJ1-B1执行授权.md` 基础设施节),入口 `run_gep_formal_overnight.sh` / `code/scripts/tj_run_b1_llm.py`

## 许可与引用

- 数据与日志按 CC-BY-4.0 发布;代码按 MIT 发布(如需变更请随 Zenodo 元数据声明)。
- 引用本包时请同时给出论文 DOI 与本包 DOI,并注明快照哈希。
