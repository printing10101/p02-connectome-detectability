# TJ1 · 停机存档与续跑指南

- **存档时间**:2026-09-22 12:00 前后(用户要求关机)
- **中断方式**:batch2 实现代理已干净停止;B1 运行进程已定点终止(PID 10100);llama-server 由关机带走
- **一句话现状**:batch1 四组判决已出并写入论文 A(v0.3);A2 阶梯终判待重跑;B1 跑完 1/60;batch2 四个启动器已就位未开跑
- **更新(09-22 22:41,已续跑)**:模型代理已拉起(8080,四模型在册含 qwen38-27b);B1 已续跑(G0 闸门进行中);batch2 已启动(a1a 首个 run fp_real_s4 运行中);A2b 重跑启动器已备好(`tj_run_a2b_neps10.py`,n_eps=10,判决词带 _NEPS10 后缀),**排 batch2 之后跑**,避免 CPU 争抢
- **更新(09-23 20:50,二次暂停)**:应用户要求再次暂停。**B1**:30B 档 29/60 完成(G0 + directed×10 + naive×10 + sham×9;sham_s10 被杀在途会自动重跑),14B 档 30 run 未跑。**batch2**:a1a ✅(18 run)并出判决 + **重议记录(见下)**;暂停于 a1d 中途(chemo/deg-random_s5 在途会自动重跑);a3/a4 未跑。**事故教训**:09-23 12:42 有外部请求向 8080 代理点 qwen38-27b,单槽代理在两模型间换载崩溃,B1 陪葬——**B1 运行期间勿向 8080 发其他模型的请求**
- **重要新成果:A1a 判决 + 重议**(`results/tj_a1a_fingerprint/`):冻结判据判 `A1A_WEIGHT_DISSOLVES`,但系**比率判据在零附近分母上的设计缺陷**(中性对照本就该在零附近摆动);直接重算原始日志:**12/12 种子配对为正,+16.32,CI [14.56, 18.07],可定义比率 7/7 全 ≥3.0——信号以更强形式成立**。详见 `A1A_REEXAMINATION_v1.1.md`(含论文 A 头条升级指令:n=12 配对口径)。论文 A 尚未并入此内容,排入下一轮整合
- **更新(09-24 09:15,三次暂停)**:应用户要求关机暂停。**B1 已全批完成并收割**:14B 判 `B1_DIRECTED_WINS`(+4.9,CI [2.76,7.04],9/10 种子)、30B 判 `B1_INCONCLUSIVE`(自报解析失败 72.4%,通道失效机制);两档 arch 变异零增益复现 sham 审计;**已并入论文 B v0.3**(`build_b.py`,含 72.4% 精度修正)。**batch2 暂停于 a1d 趋化族尾段**(deg-random_s9 在途会自动重跑);a3/a4 未跑;a2b_neps10 启动器就绪。**下次续跑只需 batch2 一条命令(B1 无需再跑,代理检查可跳过)**;详细成果看 `TJ1-成果快照-2026-09-24.md`
- **更新(09-24 13:18,事故+加固)**:09-24 09:32 续跑的 batch2 于 13:10 被**静默外部终止**(进程全灭、无 WER 记录、无 traceback;13:10 的 Application Error 1000 是用户侧 Tobii 眼动软件崩溃,与批无关;内存充足排除 OOM)——疑 TerminateProcess 类外力(任务管理器手动结束或某清理工具)。**对策:已部署看门狗**(`scripts/tj_batch2_watchdog.sh`,静默死亡 30 秒后自动重启、最多 8 次、exit 0 才停;断点逻辑保证重跑只补半截 run),batch2 已复活并验证(deg-random_s4 从头重跑,其余 run 跳过)。**若再次静默死亡,看门狗会自行处理,无需人工**;若用户侧有会杀 python 的清理工具,请排除本项目的 python 进程
- **更新(09-24 深夜,TJ-S1 spike-in 就绪)**:主稿第二轮精读的《修复结论》(同日 20:58,在 `文稿与知识产权\论文\`)把 **spike-in 受控注入定为唯一新增实验(B1,决定标题最终强度)**,论文 §4.3 已预注册其设计与判读规则。实现已落盘并冒烟通过:`scripts/tj_run_spikein.py`(预生成候选边提议表,real/neutral 两臂同表、只差适应度是否参与选择;引擎侧 `evolution.py`/`runner.py` 增 offer_table/offers_path 参数,**默认关闭,历史 run 逐位不变**,旧路径回归已验证)。**排 batch2 之后、a2b_neps10 之前跑(仅约 3h;若主稿优先级最高,这是下一个该跑的实验)**;设计与判读见 `SPIKEIN-受控注入-设计与判读-2026-09-24.md`

---

## 一、已锁定的成果(文件都在盘上,不会丢)

### 实验判决(全部按冻结判据自动判读)
| 实验 | 判决 | 关键数 | 判决文件 |
|---|---|---|---|
| TJ-C1 谱系重算 | `C1_GRADUAL` | 副本存续率跨协议 2%–24%,与中性无系统差;4.3% 系单种子失真 | `results/tj_c1_lineage/C1_VERDICT.md` |
| TJ-A1b 长跑 n=10 | `A1B_STRUCTURE_INSENSITIVE` | 0/10 过线,均值 0.844× | `code/results/tj_a1b_long/tj_a1b_verdict.json` |
| TJ-A1c 全断档 n=10 | `A1C_FRAGMENTARY` | 3/10 过线(种子 1/5/9,值 1.64/1.89/1.50),均值 1.103× | `code/results/tj_a1c_hardcut/tj_a1c_verdict.json` |
| TJ-A2a 权重冻结 n=10 | `A2A_STRUCTURE_INSENSITIVE` | 1/10 过线,均值 1.041×;行为增益 +0.115 低于噪声地板 | `code/results/tj_a2a_freeze/tj_a2a_verdict.json` |
| TJ-A2b 候选缩减 n=10 | `A2B_STRUCTURE_INSENSITIVE` | 0/10 过线,均值 1.066×;**行为增益 +0.880(CI +0.33~+1.43)不含零,10/10 种子定向边>诱饵——「承重但不固定」在缩减空间复现** | `code/results/tj_a2b_narrow/tj_a2b_verdict.json` |
| A2 阶梯终审 | `A2_SELECTION_BLIND` **暂缓主张**(噪声门未过) | 单回合评估 SD 0.686/0.851 > 效应量/3;重跑规格已立案 | `results/tj_a2_ladder/A2_LADDER_VERDICT.md` + `eval_noise.csv` |

### 论文与图表(原文件名原地更新)
- **论文 A**:`文稿与知识产权\论文\论文A初稿-真实果蝇连接组上的选择可见性不对称.docx` —— v0.3,已并入四组新判决 + 新 §3.7 阳性对照阶梯 + 图 5 阶梯森林图(`paper_drafts/make_fig_ladder.py` 可复现);源文件 `paper_drafts/build_a.py`
- **论文 B**:`论文B初稿-自修改系统架构搜索的规模定律与审计纪律.docx` —— 润色版(标题已改「规模衰减规律」,文件名未改);`build_b.py`
- **论文 C**:`论文C初稿-数字果蝇进化平台与受控实验方法学.docx` —— 润色版(补齐可得性/环境/位级复现限定);`build_c.py`
- 公式/西文字体已是 Times New Roman;11 张图顶刊风格重绘,Origin 数据表在 `paper_drafts/figures/origin_data/`(22 CSV + 导入指南;本机未装 Origin)

---

## 二、中断点与续跑步骤(下次开机按序执行)

### 0. 开机确认(1 分钟)
- 模型代理要在跑:`curl http://127.0.0.1:8080/v1/models`(Git Bash)。没起就先起 E:\llama-cpp\model-proxy.js(node);api key 在同目录 `.api-key`。infra 变更已留痕:models.json 增了 qwen3-14b 条目。

### 1. ~~续跑 B1~~ ✅ 已完成(09-24 08:10 全批 60 run 判决落地,14B DIRECTED_WINS / 30B INCONCLUSIVE,已并入论文 B v0.3;判决文件 `code/results/tj_b1_llm/tj_b1_llm_verdict.json`)
```bash
cd "E:\数据与成果\数字果蝇进化研究\code\scripts"
nohup "C:\Users\Lenovo\AppData\Local\Programs\Python\Python314\python.exe" -X utf8 tj_run_b1_llm.py > ..\results\tj_b1_llm\batch_console.log 2>&1 &
```
- 断点:已完成 run 靠结果文件跳过(**文件在 run 末尾一次性落盘,被杀的半截 run 会自动重跑,无脏数据**)
- 中断点:directed_s1 完成(30B 档,实测 29 min/run)→ 全批 60 run 约 28h(30B ≈12h + 14B ≈12h + G0 闸门)
- 判据:冻结 JSON 的 TJ_B1 键,判读自动化(B1_DIRECTED_WINS / B1_EQUIV / B1_NOISE)

### 2. 启动 batch2(A1a→A1d→A3→A4,约 4–5 天)
- 四个启动器 + `tj_batch2_runner.py` 已落盘;**但代理在冒烟阶段被停,未做过端到端验证**
- 首跑建议:先 `--stage a1a` 单段,盯前几个 run 正常再放全批;命令:
```bash
cd "E:\数据与成果\数字果蝇进化研究\code"
nohup "C:\Users\Lenovo\AppData\Local\Programs\Python\Python314\python.exe" scripts/tj_batch2_runner.py > results/tj_batch2_console.log 2>&1 &
```
- 若某启动器报错:修脚本(风格参照 tj_run_a1b_long.py),进度文件会记录到哪一段

### 3. A2b n_eps=10 重跑(约 30h,阶梯终判的最后一环)
- 规格:`results/tj_a2_ladder/A2_LADDER_VERDICT.md` 已写明(A2b 全套、n_eps=10、预计 SD 0.269 < 门限 0.293)
- 实现:复制 `tj_run_a2b_narrow.py` 改 n_eps=1→10、输出目录改 `results/tj_a2b_neps10`,判据读同一冻结键;跑完重算噪声比 → 过门则签发 A2_SELECTION_BLIND,显形则 A2_OPERATOR_BOTTLENECK
- 与 batch2 抢 CPU:二选一先跑,建议 batch2 先(A 线优先纪律)

### 3b. TJ-S1 spike-in 受控注入(已就绪,约 3h;建议排在 3 之前 —— 主稿标题等它裁决)
- 设计与判读:`SPIKEIN-受控注入-设计与判读-2026-09-24.md`(预注册锚点 = 论文 §4.3 + 修复结论 B1)
- 协议:A2b 逐键复制 + 预生成候选边提议表(两臂同表、只差适应度是否参与选择);判据复用冻结键 TJ_A1b_longrun_structure(1.5 判线 / 8-10 支持线)
- 冒烟与旧路径回归已通过(09-24 深夜);**CPU 纪律与 batch2/a2b_neps10 二选一,不并行**:
```bash
cd "E:\数据与成果\数字果蝇进化研究\code"
nohup "C:\Users\Lenovo\AppData\Local\Programs\Python\Python314\python.exe" -X utf8 scripts/tj_run_spikein.py > results/tj_spikein_console.log 2>&1 &
```
- 断点续跑:run 级跳过 + 提议表落盘复用;判读前自动重放配对自检(提议序列与表不一致会拒绝出判决)
- 判决落地 → 主稿标题与 §4.3/§5 按三分支规则改叙(见设计文档 §6)

### 4. 数据落地后回写论文
- B1 判决 → 论文 B(§3.5/§5 的 LLM 部分按判决改叙,预注册已写死三种改法)
- A2b 重跑判决 → 论文 A §3.7 + 摘要「暂缓」句定稿
- A1a/A1d/A3/A4 → 论文 A(n=10 指纹、双任务拓扑、技能图 20 种子)与 A4 λ(W) 新图

---

## 三、待用户事项(不阻塞实验)
1. 读三篇论文摘要+讨论(脊椎判断);B 篇文件名要不要从「定律」改成「规律」
2. 27B 是否加为 B1 第三档(E 盘有 Qwen3.8-27B-GSQ-RCO-IQ3_S,代理以量化质量落选并留痕;加档需你确认)
3. 投稿前:约 15 条「待核」文献网核、全脑突触定值、注册公开仓储/DOI、B 篇 [6] 标注指向复核

## 四、纪律提醒
- 冻结判据 `TJ1-预注册判据-frozen-v1.0.json` 不得回改;任何修订另立 v1.1 留痕
- 全部判决 JSON 带 criteria_sha256,可追溯
- 下次继续时把本文件给 ZCode 看即可无缝接续
