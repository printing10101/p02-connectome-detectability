# TJ1-B1 执行授权与解锁记录

- 日期:2026-09-22
- 解锁对象:`TJ1-预注册判据-frozen-v1.0.json` → `experiments.TJ_B1_llm_three_arm`
- 冻结状态:`CONDITIONAL_LLAMA_SERVER_NOT_RUNNING_TOKEN_BUDGET_NOT_FROZEN`,blockers:① llama-server 未运行;② 用户未拍板 token 预算与最终模型清单。
- 本文件的作用:解除上述 CONDITIONAL,允许 B1 开跑。**冻结判据本体(directed−sham CI、TOST 等价、种子方差、自报误分率 ≤0.2)一字不改**;判据 JSON 同样不回改。

## 1. 用户授权原话(2026-09-22)

- 模型:**用 Qwen3 系列**。
- token 预算:**不设上限("随便用")**。

## 2. 全盘 GGUF 模型搜索结果(2026-09-22 实测)

| 文件 | 大小 | 位置 | Qwen3 系列 |
|---|---|---|---|
| Qwen3-30B-A3B-Instruct-2507-UD-Q4_K_XL.gguf | 17G | E:\llama-cpp\models | 是(Qwen3-30B-A3B MoE,试点/正式管线已验证) |
| Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf | 11G | E:\llama-cpp\models | 是(Qwen3.8 代,IQ3_S 重压缩量化,GSQ-RCO 非标准微调) |
| gpt-oss-20b-Q4_K_M.gguf | 11G | E:\llama-cpp\models | 否(OpenAI 系,被本次授权排除) |
| Qwen3-14B-Q4_K_M.gguf | 8.4G | D:\llama\models | 是(标准 Q4_K_M 量化) |
| Qwen3-4B-Q4_K_M.gguf | ~2.5G | D:\扫描检测软件\models\llm | 是(容量过小,不进判决批) |
| Qwen3-Embedding-0.6B-Q8_0.gguf | ~0.7G | D:\llama\models | 嵌入模型,不适用 |

冻结 design 中的 qwen3-8b 本机不存在;gpt-oss-20b 与"Qwen3 系列"授权冲突,不再采用。

## 3. 最终选型与理由

- **qwen3-instruct-30b**(Qwen3-30B-A3B-Instruct-2507-UD-Q4_K_XL,ctx 65536,ncmoe 22):试点 exp_gep_llm.py 与正式版 exp_gep_llm_formal.py 全部在此模型上调通,frozen design 三模型之一,首选。
- **qwen3-14b**(Qwen3-14B-Q4_K_M,ctx 32768):补尺寸档位(8B 缺位,取最接近的 14B),标准量化无微调污染,满足"2 模型"批设计。
- 落选:Qwen3.8-27B(IQ3_S 重压缩 + 非标准微调,判决实验不引入额外量化/微调混杂)、gpt-oss-20b(授权排除)、Qwen3-4B(过弱,GEP 任务区分度不足)。

## 4. 基础设施与端口

- 服务:**复用既有 llama.cpp 代理链**,不另起独立实例。原因:`E:\llama-cpp\model-proxy.js` 的 killUpstream 按映像名 `taskkill /IM llama-server.exe /F`,独立实例会被其周期性误杀;代理自带随用随启/崩溃自愈/换模型串行化。
- 客户端端点:`http://127.0.0.1:8080/v1`(Bearer key = `E:\llama-cpp\.api-key`,即环境变量 LLAMA_API_KEY / LLAMA_LOCAL_API_KEY 所指)。
- 上游实例:`127.0.0.1:8081`(由代理按需经 `start-llama-server.ps1` 拉起,一次只驻留一个模型,批次内按"先 30B 全批、后 14B 全批"串行,把换模型开销压到每模型一次冷加载)。
- GPU:NVIDIA RTX 3080 Laptop 16GB(用户实测:30B ncmoe 22-26 约 12.5-14.6GB,28 t/s;14B 全卸载约 12GB,无显存冲突)。
- 基础设施侧变更(可逆):`E:\llama-cpp\models.json` 追加 qwen3-14b 条目 + GGUF 从 D:\llama\models 拷贝至 E:\llama-cpp\models(2026-09-22 10:32)。

## 5. 启动时间

- 全量批(2 模型 × 3 臂 × 10 种子 × 10 代)启动时间:见 `TJ1_batch_progress.md` 的 b1 条目(冒烟先于全批,冒烟产物验证后删除)。

## 6. 启动前缺陷修复(留痕,非判据改动)

- **任务生成器 clist 族碰撞缺陷**:exp_gep_llm_formal.py 的 `_gen_constrained` 注释自述 "g=12..24 每档候选数 6~55 个,保证可解",但代码仅取 g∈{12,15,18,21,24} 五值;clist 族 prompt 只由 g 决定,而任务集 v2(12 验证+20 测试)中 clist 抽 8 题 —— 5 个可能取值抽 8 题,脚本自身的去重断言鸽笼必炸。**正式版脚本在原码下永远无法通过自检、从未跑成**(其 results 目录仅有 G0 探针,无任何正式批产物,旁证一致)。
- 修复:`tj_run_b1_llm.py` 中 g 改为 `randint(12, 24)` 全档,回归注释的预注册意图;家族语义、难度带、题目结构不变;修复后 32 题全唯一、四族各 8、全档 g 可解已实测。
- `exp_gep_llm_formal.py` 原文件未动(留作冻结管线参照)。

## 7. 批设计对照

| 项 | frozen design | 实际执行 | 差异说明 |
|---|---|---|---|
| 模型 | qwen3-8b / gpt-oss-20b / qwen3-30b-a3b(3 个) | qwen3-30b-a3b + qwen3-14b(2 个) | 8B 不存在;gpt-oss 被 Qwen3 系列授权排除;如实用记录 |
| 臂 | directed / sham / naive | 同(3 臂) | 无差异 |
| 种子 | 1-10 | 1-10 | 无差异 |
| GENS | 10 | 10 | 无差异 |
| 任务 | 正式任务集 | exp_gep_llm_formal.py 任务集 v2(12 验证 + 20 测试) | 无差异 |
| 判据 | 冻结 criteria 原文 | 同,逐条实现 | 阈值未 frozen 的操作化定义(TOST 等价边界等)在 verdict JSON 内留痕 |
