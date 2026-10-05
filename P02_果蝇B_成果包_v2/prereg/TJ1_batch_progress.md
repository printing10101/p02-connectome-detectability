
## 批启动 2026-09-22 04:21:46
范围: 全批 a1b→a1c→a2a→a2b
- [2026-09-22 04:21:46] a1b 开始
- [2026-09-22 05:35:45] a1b 完成: 新跑 14 个 (real_s4, neutral_s4, real_s5, neutral_s5, real_s6, neutral_s6, real_s7, neutral_s7, real_s8, neutral_s8, real_s9, neutral_s9, real_s10, neutral_s10), 跳过 0 个 (-), wall 74.0 min, verdict = A1B_STRUCTURE_INSENSITIVE
- [2026-09-22 05:35:45] a1c 开始
- [2026-09-22 05:57:02] a1c 完成: 新跑 14 个 (real_s4, neutral_s4, real_s5, neutral_s5, real_s6, neutral_s6, real_s7, neutral_s7, real_s8, neutral_s8, real_s9, neutral_s9, real_s10, neutral_s10), 跳过 0 个 (-), wall 21.3 min, verdict = A1C_FRAGMENTARY
- [2026-09-22 05:57:02] a2a 开始
- [2026-09-22 07:35:24] a2a 完成: 新跑 20 个 (real_s1, neutral_s1, real_s2, neutral_s2, real_s3, neutral_s3, real_s4, neutral_s4, real_s5, neutral_s5, real_s6, neutral_s6, real_s7, neutral_s7, real_s8, neutral_s8, real_s9, neutral_s9, real_s10, neutral_s10), 跳过 0 个 (-), wall 98.4 min, verdict = A2A_STRUCTURE_INSENSITIVE
- [2026-09-22 07:35:24] a2b 开始
- [2026-09-22 10:33:01] a2b 完成: 新跑 20 个 (real_s1, neutral_s1, real_s2, neutral_s2, real_s3, neutral_s3, real_s4, neutral_s4, real_s5, neutral_s5, real_s6, neutral_s6, real_s7, neutral_s7, real_s8, neutral_s8, real_s9, neutral_s9, real_s10, neutral_s10), 跳过 0 个 (-), wall 177.6 min, verdict = A2B_STRUCTURE_INSENSITIVE
- [2026-09-22 10:33:01] 批结束 —— 全部段正常返回
- [2026-09-22 10:57:37] b1 开始 (models=qwen3-instruct-30b,qwen3-14b; 3臂 10种子 x 10代; 任务 12验证+20测试; endpoint http://127.0.0.1:8080/v1; 冒烟=False)
- [2026-09-22 10:57:37] b1 qwen3-instruct-30b G0 闸门 + 3 臂批开始
- [2026-09-22 11:31:01] b1 qwen3-instruct-30b directed_s1 完成 (test 8/20, tok 97553/53859, wall 29.0 min)
- [2026-09-22 22:41:49] b1 开始 (models=qwen3-instruct-30b,qwen3-14b; 3臂 10种子 x 10代; 任务 12验证+20测试; endpoint http://127.0.0.1:8080/v1; 冒烟=False)
- [2026-09-22 22:41:49] b1 qwen3-instruct-30b G0 闸门 + 3 臂批开始

## 批启动 2026-09-22 22:41:54 (batch2)
范围: 全批 a1a→a1d→a3→a4
- [2026-09-22 22:41:54] a1a 开始 (batch2)
- [2026-09-22 22:57:23] b1 qwen3-instruct-30b directed_s1 完成 (test 8/20, tok 97553/53859, wall 29.0 min)
- [2026-09-23 03:12:54] b1 qwen3-instruct-30b directed_s2 完成 (test 14/20, tok 106511/52758, wall 255.5 min)
- [2026-09-23 04:23:54] b1 qwen3-instruct-30b directed_s3 完成 (test 8/20, tok 124759/60350, wall 71.0 min)
- [2026-09-23 05:02:37] b1 qwen3-instruct-30b directed_s4 完成 (test 10/20, tok 106254/64838, wall 38.7 min)
- [2026-09-23 05:34:56] b1 qwen3-instruct-30b directed_s5 完成 (test 15/20, tok 112814/53797, wall 32.3 min)
- [2026-09-23 06:08:43] b1 qwen3-instruct-30b directed_s6 完成 (test 9/20, tok 121316/57376, wall 33.8 min)
- [2026-09-23 06:43:16] b1 qwen3-instruct-30b directed_s7 完成 (test 9/20, tok 116917/57483, wall 34.5 min)
- [2026-09-23 07:20:31] b1 qwen3-instruct-30b directed_s8 完成 (test 11/20, tok 111818/60842, wall 37.2 min)
- [2026-09-23 08:04:33] b1 qwen3-instruct-30b directed_s9 完成 (test 10/20, tok 118545/73656, wall 44.0 min)
- [2026-09-23 08:32:43] b1 qwen3-instruct-30b directed_s10 完成 (test 12/20, tok 121585/46589, wall 28.2 min)
- [2026-09-23 08:47:30] b1 qwen3-instruct-30b sham_s1 完成 (test 11/20, tok 20859/24509, wall 14.8 min)
- [2026-09-23 09:00:01] b1 qwen3-instruct-30b sham_s2 完成 (test 10/20, tok 20859/20759, wall 12.5 min)
- [2026-09-23 09:13:55] b1 qwen3-instruct-30b sham_s3 完成 (test 12/20, tok 20547/23439, wall 13.9 min)
- [2026-09-23 09:28:21] b1 qwen3-instruct-30b sham_s4 完成 (test 12/20, tok 21547/23699, wall 14.4 min)
- [2026-09-23 09:42:47] b1 qwen3-instruct-30b sham_s5 完成 (test 11/20, tok 20475/24005, wall 14.4 min)
- [2026-09-23 09:54:20] b1 qwen3-instruct-30b sham_s6 完成 (test 11/20, tok 19219/19162, wall 11.6 min)
- [2026-09-23 10:12:29] b1 qwen3-instruct-30b sham_s7 完成 (test 11/20, tok 20083/25919, wall 18.1 min)
- [2026-09-23 10:50:34] b1 qwen3-instruct-30b sham_s8 完成 (test 11/20, tok 19771/23581, wall 38.1 min)
- [2026-09-23 12:08:52] b1 qwen3-instruct-30b sham_s9 完成 (test 12/20, tok 20691/31377, wall 78.3 min)
- [2026-09-23 18:50:26] a1a 完成: 新跑 18 个 (fp_real_s4, fp_neutral_s4, fp_real_s5, fp_neutral_s5, fp_real_s6, fp_neutral_s6, fp_real_s7, fp_neutral_s7, fp_real_s8, fp_neutral_s8, fp_real_s9, fp_neutral_s9, fp_real_s10, fp_neutral_s10, fp_real_s11, fp_neutral_s11, fp_real_s12, fp_neutral_s12), 跳过/复用 0 个 (-), wall 1208.5 min, verdict = A1A_WEIGHT_DISSOLVES
- [2026-09-23 18:50:26] a1d 开始 (batch2)
- [2026-09-23 20:46:45] b1 开始 (models=qwen3-instruct-30b,qwen3-14b; 3臂 10种子 x 10代; 任务 12验证+20测试; endpoint http://127.0.0.1:8080/v1; 冒烟=False)
- [2026-09-23 20:46:45] b1 qwen3-instruct-30b G0 闸门 + 3 臂批开始
- [2026-09-23 20:56:27] b1 qwen3-instruct-30b directed_s1 完成 (test 8/20, tok 97553/53859, wall 29.0 min)
- [2026-09-23 20:56:27] b1 qwen3-instruct-30b directed_s2 完成 (test 14/20, tok 106511/52758, wall 255.5 min)
- [2026-09-23 20:56:27] b1 qwen3-instruct-30b directed_s3 完成 (test 8/20, tok 124759/60350, wall 71.0 min)
- [2026-09-23 20:56:27] b1 qwen3-instruct-30b directed_s4 完成 (test 10/20, tok 106254/64838, wall 38.7 min)
- [2026-09-23 20:56:27] b1 qwen3-instruct-30b directed_s5 完成 (test 15/20, tok 112814/53797, wall 32.3 min)
- [2026-09-23 20:56:27] b1 qwen3-instruct-30b directed_s6 完成 (test 9/20, tok 121316/57376, wall 33.8 min)
- [2026-09-23 20:56:27] b1 qwen3-instruct-30b directed_s7 完成 (test 9/20, tok 116917/57483, wall 34.5 min)
- [2026-09-23 20:56:27] b1 qwen3-instruct-30b directed_s8 完成 (test 11/20, tok 111818/60842, wall 37.2 min)
- [2026-09-23 20:56:27] b1 qwen3-instruct-30b directed_s9 完成 (test 10/20, tok 118545/73656, wall 44.0 min)
- [2026-09-23 20:56:27] b1 qwen3-instruct-30b directed_s10 完成 (test 12/20, tok 121585/46589, wall 28.2 min)
- [2026-09-23 20:56:27] b1 qwen3-instruct-30b sham_s1 完成 (test 11/20, tok 20859/24509, wall 14.8 min)
- [2026-09-23 20:56:27] b1 qwen3-instruct-30b sham_s2 完成 (test 10/20, tok 20859/20759, wall 12.5 min)
- [2026-09-23 20:56:27] b1 qwen3-instruct-30b sham_s3 完成 (test 12/20, tok 20547/23439, wall 13.9 min)
- [2026-09-23 20:56:27] b1 qwen3-instruct-30b sham_s4 完成 (test 12/20, tok 21547/23699, wall 14.4 min)
- [2026-09-23 20:56:27] b1 qwen3-instruct-30b sham_s5 完成 (test 11/20, tok 20475/24005, wall 14.4 min)
- [2026-09-23 20:56:27] b1 qwen3-instruct-30b sham_s6 完成 (test 11/20, tok 19219/19162, wall 11.6 min)
- [2026-09-23 20:56:27] b1 qwen3-instruct-30b sham_s7 完成 (test 11/20, tok 20083/25919, wall 18.1 min)
- [2026-09-23 20:56:27] b1 qwen3-instruct-30b sham_s8 完成 (test 11/20, tok 19771/23581, wall 38.1 min)
- [2026-09-23 20:56:27] b1 qwen3-instruct-30b sham_s9 完成 (test 12/20, tok 20691/31377, wall 78.3 min)

## 批启动 2026-09-23 22:43:55 (batch2)
范围: 全批 a1a→a1d→a3→a4
- [2026-09-23 22:43:55] a1a 开始 (batch2)
- [2026-09-23 22:43:58] a1a 完成: 新跑 0 个 (-), 跳过/复用 18 个 (fp_real_s4, fp_neutral_s4, fp_real_s5, fp_neutral_s5, fp_real_s6, fp_neutral_s6, fp_real_s7, fp_neutral_s7, fp_real_s8, fp_neutral_s8, fp_real_s9, fp_neutral_s9, fp_real_s10, fp_neutral_s10, fp_real_s11, fp_neutral_s11, fp_real_s12, fp_neutral_s12), wall 0.1 min, verdict = A1A_WEIGHT_DISSOLVES
- [2026-09-23 22:43:58] a1d 开始 (batch2)
- [2026-09-23 22:43:59] b1 开始 (models=qwen3-instruct-30b,qwen3-14b; 3臂 10种子 x 10代; 任务 12验证+20测试; endpoint http://127.0.0.1:8080/v1; 冒烟=False)
- [2026-09-23 22:43:59] b1 qwen3-instruct-30b G0 闸门 + 3 臂批开始
- [2026-09-23 23:01:12] b1 qwen3-instruct-30b directed_s1 完成 (test 8/20, tok 97553/53859, wall 29.0 min)
- [2026-09-23 23:01:12] b1 qwen3-instruct-30b directed_s2 完成 (test 14/20, tok 106511/52758, wall 255.5 min)
- [2026-09-23 23:01:12] b1 qwen3-instruct-30b directed_s3 完成 (test 8/20, tok 124759/60350, wall 71.0 min)
- [2026-09-23 23:01:12] b1 qwen3-instruct-30b directed_s4 完成 (test 10/20, tok 106254/64838, wall 38.7 min)
- [2026-09-23 23:01:12] b1 qwen3-instruct-30b directed_s5 完成 (test 15/20, tok 112814/53797, wall 32.3 min)
- [2026-09-23 23:01:12] b1 qwen3-instruct-30b directed_s6 完成 (test 9/20, tok 121316/57376, wall 33.8 min)
- [2026-09-23 23:01:12] b1 qwen3-instruct-30b directed_s7 完成 (test 9/20, tok 116917/57483, wall 34.5 min)
- [2026-09-23 23:01:12] b1 qwen3-instruct-30b directed_s8 完成 (test 11/20, tok 111818/60842, wall 37.2 min)
- [2026-09-23 23:01:12] b1 qwen3-instruct-30b directed_s9 完成 (test 10/20, tok 118545/73656, wall 44.0 min)
- [2026-09-23 23:01:12] b1 qwen3-instruct-30b directed_s10 完成 (test 12/20, tok 121585/46589, wall 28.2 min)
- [2026-09-23 23:01:12] b1 qwen3-instruct-30b sham_s1 完成 (test 11/20, tok 20859/24509, wall 14.8 min)
- [2026-09-23 23:01:12] b1 qwen3-instruct-30b sham_s2 完成 (test 10/20, tok 20859/20759, wall 12.5 min)
- [2026-09-23 23:01:12] b1 qwen3-instruct-30b sham_s3 完成 (test 12/20, tok 20547/23439, wall 13.9 min)
- [2026-09-23 23:01:12] b1 qwen3-instruct-30b sham_s4 完成 (test 12/20, tok 21547/23699, wall 14.4 min)
- [2026-09-23 23:01:12] b1 qwen3-instruct-30b sham_s5 完成 (test 11/20, tok 20475/24005, wall 14.4 min)
- [2026-09-23 23:01:12] b1 qwen3-instruct-30b sham_s6 完成 (test 11/20, tok 19219/19162, wall 11.6 min)
- [2026-09-23 23:01:12] b1 qwen3-instruct-30b sham_s7 完成 (test 11/20, tok 20083/25919, wall 18.1 min)
- [2026-09-23 23:01:12] b1 qwen3-instruct-30b sham_s8 完成 (test 11/20, tok 19771/23581, wall 38.1 min)
- [2026-09-23 23:01:12] b1 qwen3-instruct-30b sham_s9 完成 (test 12/20, tok 20691/31377, wall 78.3 min)
- [2026-09-23 23:53:14] b1 qwen3-instruct-30b sham_s10 完成 (test 9/20, tok 21243/34992, wall 52.0 min)
- [2026-09-24 00:21:29] b1 qwen3-instruct-30b naive_s1 完成 (test 12/20, tok 58162/24244, wall 28.3 min)
- [2026-09-24 00:42:04] b1 qwen3-instruct-30b naive_s2 完成 (test 9/20, tok 48858/20825, wall 20.6 min)
- [2026-09-24 01:13:12] b1 qwen3-instruct-30b naive_s3 完成 (test 13/20, tok 79031/26969, wall 31.1 min)
- [2026-09-24 01:39:56] b1 qwen3-instruct-30b naive_s4 完成 (test 11/20, tok 92577/28742, wall 26.7 min)
- [2026-09-24 02:06:34] b1 qwen3-instruct-30b naive_s5 完成 (test 11/20, tok 40637/23311, wall 26.6 min)
- [2026-09-24 02:30:21] b1 qwen3-instruct-30b naive_s6 完成 (test 10/20, tok 56533/22674, wall 23.8 min)
- [2026-09-24 02:51:59] b1 qwen3-instruct-30b naive_s7 完成 (test 9/20, tok 64255/19603, wall 21.6 min)
- [2026-09-24 03:18:31] b1 qwen3-instruct-30b naive_s8 完成 (test 9/20, tok 100935/23107, wall 26.5 min)
- [2026-09-24 03:40:28] b1 qwen3-instruct-30b naive_s9 完成 (test 10/20, tok 50661/21540, wall 21.9 min)
- [2026-09-24 04:06:16] b1 qwen3-instruct-30b naive_s10 完成 (test 9/20, tok 95226/24499, wall 25.8 min)
- [2026-09-24 04:06:16] b1 qwen3-instruct-30b 完成: 30/30 run, verdict = B1_INCONCLUSIVE, wall 322.3 min -> tj_b1_qwen3-instruct-30b_verdict.json
- [2026-09-24 04:06:16] b1 qwen3-14b G0 闸门 + 3 臂批开始
- [2026-09-24 04:23:54] b1 qwen3-14b directed_s1 完成 (test 6/20, tok 94323/30419, wall 16.0 min)
- [2026-09-24 04:40:34] b1 qwen3-14b directed_s2 完成 (test 9/20, tok 99971/32138, wall 16.7 min)
- [2026-09-24 04:53:54] b1 qwen3-14b directed_s3 完成 (test 3/20, tok 88355/25619, wall 13.3 min)
- [2026-09-24 05:08:08] b1 qwen3-14b directed_s4 完成 (test 1/20, tok 100387/27374, wall 14.2 min)
- [2026-09-24 05:22:10] b1 qwen3-14b directed_s5 完成 (test 2/20, tok 93786/27023, wall 14.0 min)
- [2026-09-24 05:38:22] b1 qwen3-14b directed_s6 完成 (test 9/20, tok 90009/31324, wall 16.2 min)
- [2026-09-24 05:54:59] b1 qwen3-14b directed_s7 完成 (test 8/20, tok 89744/32235, wall 16.6 min)
- [2026-09-24 06:13:14] b1 qwen3-14b directed_s8 完成 (test 7/20, tok 95450/35441, wall 18.2 min)
- [2026-09-24 06:26:45] b1 qwen3-14b directed_s9 完成 (test 8/20, tok 87715/26031, wall 13.5 min)
- [2026-09-24 06:43:07] b1 qwen3-14b directed_s10 完成 (test 8/20, tok 86428/31748, wall 16.4 min)
- [2026-09-24 06:46:55] b1 qwen3-14b sham_s1 完成 (test 2/20, tok 19607/7060, wall 3.8 min)
- [2026-09-24 06:50:37] b1 qwen3-14b sham_s2 完成 (test 1/20, tok 19639/6836, wall 3.7 min)
- [2026-09-24 06:54:23] b1 qwen3-14b sham_s3 完成 (test 1/20, tok 20575/6986, wall 3.8 min)
- [2026-09-24 06:58:26] b1 qwen3-14b sham_s4 完成 (test 1/20, tok 21259/7543, wall 4.0 min)
- [2026-09-24 07:02:08] b1 qwen3-14b sham_s5 完成 (test 1/20, tok 19639/6836, wall 3.7 min)
- [2026-09-24 07:05:52] b1 qwen3-14b sham_s6 完成 (test 1/20, tok 20059/6944, wall 3.7 min)
- [2026-09-24 07:09:33] b1 qwen3-14b sham_s7 完成 (test 1/20, tok 19639/6836, wall 3.7 min)
- [2026-09-24 07:13:21] b1 qwen3-14b sham_s8 完成 (test 2/20, tok 19607/7038, wall 3.8 min)
- [2026-09-24 07:17:03] b1 qwen3-14b sham_s9 完成 (test 1/20, tok 19639/6836, wall 3.7 min)
- [2026-09-24 07:20:47] b1 qwen3-14b sham_s10 完成 (test 1/20, tok 19939/6899, wall 3.7 min)
- [2026-09-24 07:26:25] b1 qwen3-14b naive_s1 完成 (test 2/20, tok 46272/10647, wall 5.6 min)
- [2026-09-24 07:31:03] b1 qwen3-14b naive_s2 完成 (test 2/20, tok 39359/9180, wall 4.6 min)
- [2026-09-24 07:36:10] b1 qwen3-14b naive_s3 完成 (test 1/20, tok 48421/10206, wall 5.1 min)
- [2026-09-24 07:41:06] b1 qwen3-14b naive_s4 完成 (test 2/20, tok 43199/9841, wall 4.9 min)
- [2026-09-24 07:46:10] b1 qwen3-14b naive_s5 完成 (test 2/20, tok 47270/10124, wall 5.1 min)
- [2026-09-24 07:50:54] b1 qwen3-14b naive_s6 完成 (test 1/20, tok 43714/9449, wall 4.7 min)
- [2026-09-24 07:55:54] b1 qwen3-14b naive_s7 完成 (test 3/20, tok 50820/9960, wall 5.0 min)
- [2026-09-24 08:00:43] b1 qwen3-14b naive_s8 完成 (test 3/20, tok 43780/9566, wall 4.8 min)
- [2026-09-24 08:05:41] b1 qwen3-14b naive_s9 完成 (test 1/20, tok 45542/9915, wall 5.0 min)
- [2026-09-24 08:10:15] b1 qwen3-14b naive_s10 完成 (test 2/20, tok 52321/9086, wall 4.6 min)
- [2026-09-24 08:10:15] b1 qwen3-14b 完成: 30/30 run, verdict = B1_DIRECTED_WINS, wall 244.0 min -> tj_b1_qwen3-14b_verdict.json
- [2026-09-24 08:10:15] b1 全批完成: verdicts = {'qwen3-instruct-30b': 'B1_INCONCLUSIVE', 'qwen3-14b': 'B1_DIRECTED_WINS'}, wall 566.3 min -> tj_b1_llm_verdict.json

## 批启动 2026-09-24 09:32:12 (batch2)
范围: 全批 a1a→a1d→a3→a4
- [2026-09-24 09:32:12] a1a 开始 (batch2)
- [2026-09-24 09:32:15] a1a 完成: 新跑 0 个 (-), 跳过/复用 18 个 (fp_real_s4, fp_neutral_s4, fp_real_s5, fp_neutral_s5, fp_real_s6, fp_neutral_s6, fp_real_s7, fp_neutral_s7, fp_real_s8, fp_neutral_s8, fp_real_s9, fp_neutral_s9, fp_real_s10, fp_neutral_s10, fp_real_s11, fp_neutral_s11, fp_real_s12, fp_neutral_s12), wall 0.0 min, verdict = A1A_WEIGHT_DISSOLVES
- [2026-09-24 09:32:15] a1d 开始 (batch2)

## 批启动 2026-09-24 13:17:56 (batch2)
范围: 全批 a1a→a1d→a3→a4
- [2026-09-24 13:17:56] a1a 开始 (batch2)
- [2026-09-24 13:17:59] a1a 完成: 新跑 0 个 (-), 跳过/复用 18 个 (fp_real_s4, fp_neutral_s4, fp_real_s5, fp_neutral_s5, fp_real_s6, fp_neutral_s6, fp_real_s7, fp_neutral_s7, fp_real_s8, fp_neutral_s8, fp_real_s9, fp_neutral_s9, fp_real_s10, fp_neutral_s10, fp_real_s11, fp_neutral_s11, fp_real_s12, fp_neutral_s12), wall 0.0 min, verdict = A1A_WEIGHT_DISSOLVES
- [2026-09-24 13:17:59] a1d 开始 (batch2)

## 批启动 2026-09-24 13:41:12 (batch2)
范围: 全批 a1a→a1d→a3→a4
- [2026-09-24 13:41:12] a1a 开始 (batch2)
- [2026-09-24 13:41:15] a1a 完成: 新跑 0 个 (-), 跳过/复用 18 个 (fp_real_s4, fp_neutral_s4, fp_real_s5, fp_neutral_s5, fp_real_s6, fp_neutral_s6, fp_real_s7, fp_neutral_s7, fp_real_s8, fp_neutral_s8, fp_real_s9, fp_neutral_s9, fp_real_s10, fp_neutral_s10, fp_real_s11, fp_neutral_s11, fp_real_s12, fp_neutral_s12), wall 0.0 min, verdict = A1A_WEIGHT_DISSOLVES
- [2026-09-24 13:41:15] a1d 开始 (batch2)

## 批启动 2026-09-24 21:11:12 (batch2)
范围: 全批 a1a→a1d→a3→a4
- [2026-09-24 21:11:12] a1a 开始 (batch2)
- [2026-09-24 21:11:14] a1a 完成: 新跑 0 个 (-), 跳过/复用 18 个 (fp_real_s4, fp_neutral_s4, fp_real_s5, fp_neutral_s5, fp_real_s6, fp_neutral_s6, fp_real_s7, fp_neutral_s7, fp_real_s8, fp_neutral_s8, fp_real_s9, fp_neutral_s9, fp_real_s10, fp_neutral_s10, fp_real_s11, fp_neutral_s11, fp_real_s12, fp_neutral_s12), wall 0.0 min, verdict = A1A_WEIGHT_DISSOLVES
- [2026-09-24 21:11:14] a1d 开始 (batch2)
- [2026-09-25 07:50:08] a1d 完成: 新跑 17 个 (pred/deg-random_s5, pred/block-shuffle_s5, pred/real_s6, pred/deg-random_s6, pred/block-shuffle_s6, pred/real_s7, pred/deg-random_s7, pred/block-shuffle_s7, pred/real_s8, pred/deg-random_s8, pred/block-shuffle_s8, pred/real_s9, pred/deg-random_s9, pred/block-shuffle_s9, pred/real_s10, pred/deg-random_s10, pred/block-shuffle_s10), 跳过/复用 18 个 (chemo/deg-random_s4, chemo/block-shuffle_s4, chemo/deg-random_s5, chemo/block-shuffle_s5, chemo/deg-random_s6, chemo/block-shuffle_s6, chemo/deg-random_s7, chemo/block-shuffle_s7, chemo/deg-random_s8, chemo/block-shuffle_s8, chemo/deg-random_s9, chemo/block-shuffle_s9, chemo/deg-random_s10, chemo/block-shuffle_s10, pred/real_s4, pred/deg-random_s4, pred/block-shuffle_s4, pred/real_s5), wall 638.9 min, verdict = A1D_TOPO_NOT_SUPPORTED
- [2026-09-25 07:50:08] a3 开始 (batch2)
- [2026-09-25 07:57:55] a3 完成: 新跑 100 个 (naive_s1, naive_s2, naive_s3, naive_s4, naive_s5, naive_s6, naive_s7, naive_s8, naive_s9, naive_s10, naive_s11, naive_s12, naive_s13, naive_s14, naive_s15, naive_s16, naive_s17, naive_s18, naive_s19, naive_s20, param-only_s1, param-only_s2, param-only_s3, param-only_s4, param-only_s5, param-only_s6, param-only_s7, param-only_s8, param-only_s9, param-only_s10, param-only_s11, param-only_s12, param-only_s13, param-only_s14, param-only_s15, param-only_s16, param-only_s17, param-only_s18, param-only_s19, param-only_s20, directed_s1, directed_s2, directed_s3, directed_s4, directed_s5, directed_s6, directed_s7, directed_s8, directed_s9, directed_s10, directed_s11, directed_s12, directed_s13, directed_s14, directed_s15, directed_s16, directed_s17, directed_s18, directed_s19, directed_s20, sham_s1, sham_s2, sham_s3, sham_s4, sham_s5, sham_s6, sham_s7, sham_s8, sham_s9, sham_s10, sham_s11, sham_s12, sham_s13, sham_s14, sham_s15, sham_s16, sham_s17, sham_s18, sham_s19, sham_s20, gated_s1, gated_s2, gated_s3, gated_s4, gated_s5, gated_s6, gated_s7, gated_s8, gated_s9, gated_s10, gated_s11, gated_s12, gated_s13, gated_s14, gated_s15, gated_s16, gated_s17, gated_s18, gated_s19, gated_s20), 跳过/复用 0 个 (-), wall 7.8 min, verdict = A3_REPLICATED
- [2026-09-25 07:57:55] a4 开始 (batch2)
- [2026-09-25 08:15:27] a4 完成: 新跑 1 个 (v2_rows+120), 跳过/复用 1 个 (v2_rows_reused_0), wall 17.5 min, verdict = A4_SCALE_LAW
- [2026-09-25 08:15:27] 批结束 (batch2) —— 全部段正常返回
