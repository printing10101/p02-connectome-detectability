# Zenodo 上传字段 · 逐字复制用

> 建立 2026-10-05。上传页：`https://zenodo.org/uploads/new`
> 内容全部取自 `论文/P02_果蝇B_EN_v2_full.md`（v2 底稿），与拟投 KBS 的版本一致。

---

## 1. 上传的文件

```
C:\Users\Lenovo\Desktop\文稿与知识产权\论文\P02_果蝇B_EN_v2.docx
```

⚠️ **只能用 v2**。旧版 `P02_果蝇B_bioRxiv提交版-v1.pdf` 已于今日清理时删除（M3 通道混淆修正之前的内容，与 v2 数字不一致）。

---

## 2. Resource type

| 字段 | 值 |
|---|---|
| 上传类型大类 | **Publication** |
| 具体类型 | **Preprint** |

❌ 不要选 `Dataset` / `Software` / `Poster`——Zenodo 的 DOI 类型由它决定，选错引文格式会不对。

---

## 3. Title（逐字复制）

```
Architecture Search in Self-Modifying Systems: The Scale-Decay Regularity of General Selection and Randomized Controlled Audit Discipline
```

---

## 4. Creators

| 字段 | 值 |
|---|---|
| Name | `Yulong Li` |
| Affiliation | `Independent Researcher, Guiyang, China` |
| ORCID | `0009-0006-0492-857X` |

---

## 5. Description（Abstract，逐字复制）

```
Progress claims of self-evolving AI systems generally lack null-hypothesis calibration: the evolutionary computation literature has a neutral-control tradition, yet representative self-modifying systems (Darwin Gödel Machine, AlphaEvolve) run only ablations in the cited reports, and an industrial self-evolving network in which over 84% of genes bypassed quality checks. Starting from the selection-visibility asymmetry quantified on a real Drosophila connectome (parametric changes amplified 3× by selection, architectural changes about 0.8×), we test architecture-level search on three substrates. On a discrete skill-graph platform, directed-channel installation with acceptance-only selection reaches 15.6/16 versus 9.2/16 for general two-layer search and 4.2/16 for drift; drift alone installs 17.4 edges, which an uncontrolled observer would credit to selection. A scale scan across the four tested scales (W = 35 to 2,331; 12 runs per cell; constant absolute flip budget F = 2.1) shows general-search discovery decaying monotonically with W (1.000 to 0.167, trend p < 0.001) while the directed channel holds at 15.333 — a designed invariance under frozen architecture and paired RNG. A generation × mutation × scale factorial refutes the crossover-generation hypothesis, exposes destruction growing as 14F/W, and supports absolute flip counts as the fairness lock for cross-scale comparison under a fixed per-generation budget. In a preregistered three-arm verdict on real LLMs (three arms × 10 seeds × 10 generations, two base tiers), the directed advantage holds at 14B (+4.9, 95% CI [2.76, 7.04], 9/10 seeds concordant; robust to leave-one-out re-estimation) and does not hold at 30B-A3B: a format-constrained rerun repaired the self-report channel (42% parse failures to 0/100) and the directed advantage remained absent (+0.2, 95% CI [−0.96, +1.36], 6/10 concordant seeds) — channel repairability and channel benefit are separate dimensions. Accepted architecture-type mutations show zero mean empirical gain at both tiers under the verdict's self-reported classification; the acceptance guard approves neutral edits wholesale. We distill five design principles: selection-visibility asymmetry, directed channels, sham audits, quality-control metrology, and scale-decaying architectural visibility.
```

---

## 6. Keywords（逐字复制，用逗号分隔）

```
self-evolving systems, architecture search, neutral control, sham audit, recursive self-improvement, neuroevolution
```

---

## 7. License

选 **`CC BY 4.0`**。

---

## 8. Related Identifiers（可选但建议填）

| 字段 | 值 |
|---|---|
| Related identifier | `URL` |
| Identifier | `https://github.com/printing10101/p02-connectome-detectability` |
| Relation | `IsSupplementTo`（或 `IsDocumentedBy`） |
| Scheme/identifier type | leave blank |

---

## 9. 社区（Communities）

**留空**。Zenodo 里的 curated community 多数是机构自建（如 EU Open Research Repository），个人投稿不需要选。

---

## 10. 提交后

- 点 **Publish** → Zenodo 分配 DOI（形如 `10.5281/zenodo.XXXXXXXX`）
- **把 DOI 发给我**，我回填到：定稿文件 §5 数据可用性声明 + 论文正文
- ⚠️ 勾上 **"Publish"** 前请确认所有必填项绿勾，否则会退回草稿

---

## 11. 明天补一个 v2

今天先传 v1 锁住**公开时间戳**（这对你比什么都重要）。明天声明块（§1–§7）并入稿末后：
`My dashboard → 选中该记录 → New version` → 上传含声明的版本。

Zenodo 支持 versioning，新版本会有自己的 DOI，旧 DOI 仍可解析。**这正好和你的预注册纪律同构——先锁时间戳，再迭代内容。**
