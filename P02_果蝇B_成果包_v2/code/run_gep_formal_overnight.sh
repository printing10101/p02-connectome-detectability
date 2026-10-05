#!/bin/bash
# G-LLM 正式批一键启动器 (自愈 + 断点续跑): 空闲时 (如睡前) 运行, 通宵自动完成。
# 用法:  bash code/run_gep_formal_overnight.sh
# 断点:  验证管线 => validate/state.json; 正式批 => 每 run 的 events_*.json。中断后重跑本脚本即续。
# 判决:  results/exp_gep_llm_formal/exp_gep_llm_formal_verdict.json
cd "E:/数据与成果/数字果蝇进化研究/code"
PY="C:/Users/Lenovo/AppData/Local/Programs/Python/Python314/python.exe"
SRV="E:/llama-cpp/bin/llama-server.exe"
GGUF="E:/llama-cpp/models/Qwen3-30B-A3B-Instruct-2507-UD-Q4_K_XL.gguf"

echo "[1/3] 启动 Qwen3 服务 (存档平台参数, 端口 1236)..."
"$SRV" -m "$GGUF" -c 65536 -np 1 -ngl 99 -fa on -ctk q8_0 -ctv q8_0 --cache-ram 0 \
  --cache-reuse 256 --dry-multiplier 0.8 --ctx-checkpoints 0 --jinja --threads 8 \
  --host 127.0.0.1 --port 1236 -a qwen3-30b-a3b --api-key "$LLAMA_API_KEY" \
  --n-cpu-moe 30 --temp 0.7 --top-p 0.8 --top-k 20 --min-p 0 --repeat-penalty 1.05 &
SRV_PID=$!
echo "等待模型热身 (最长 12 分钟)..."
ok=0
for w in $(seq 1 72); do
  C=$(curl -s -m 30 -o /dev/null -w "%{http_code}" -H "Authorization: Bearer $LLAMA_API_KEY" \
    -H "Content-Type: application/json" -d '{"model":"qwen3-30b-a3b","messages":[{"role":"user","content":"hi"}],"max_tokens":1}' \
    http://127.0.0.1:1236/v1/chat/completions 2>/dev/null)
  [ "$C" = "200" ] && { ok=1; echo "热身完成 ($((w*10))s)"; break; }
  sleep 10
done
[ "$ok" = "1" ] || { echo "热身失败, 退出 (重跑本脚本即重试)"; exit 1; }

echo "[2/3] G0 闸门探针 (ref - init >= 2 才继续)..."
"$PY" -X utf8 scripts/gep_g0_probe.py || { echo "G0 探针未完成, 退出"; kill $SRV_PID 2>/dev/null; exit 1; }
GATE=$("$PY" -X utf8 -c "import json;print(json.load(open('results/exp_gep_llm_formal/g0_probe.json',encoding='utf-8'))['gate_pass'])")
[ "$GATE" = "True" ] || { echo "G0 闸门未过 (见 g0_probe.json) —— 按预注册纪律不跑正式批"; kill $SRV_PID 2>/dev/null; exit 2; }
echo "G0 过闸。"

echo "[3/3] 正式批: 3 条件 x 3 种子 x 4 代 (断点续跑, 被杀后重跑本脚本即续)..."
for i in $(seq 1 30); do
  curl -s -m 3 -o /dev/null -H "Authorization: Bearer $LLAMA_API_KEY" http://127.0.0.1:1236/v1/models || {
    echo "[batch-guard] 服务器掉线, 重启..."
    kill $SRV_PID 2>/dev/null; sleep 5
    "$SRV" -m "$GGUF" -c 65536 -np 1 -ngl 99 -fa on -ctk q8_0 -ctv q8_0 --cache-ram 0 \
      --cache-reuse 256 --dry-multiplier 0.8 --ctx-checkpoints 0 --jinja --threads 8 \
      --host 127.0.0.1 --port 1236 -a qwen3-30b-a3b --api-key "$LLAMA_API_KEY" \
      --n-cpu-moe 30 --temp 0.7 --top-p 0.8 --top-k 20 --min-p 0 --repeat-penalty 1.05 &
    for w in $(seq 1 60); do
      C=$(curl -s -m 30 -o /dev/null -w "%{http_code}" -H "Authorization: Bearer $LLAMA_API_KEY" \
        -H "Content-Type: application/json" -d '{"model":"qwen3-30b-a3b","messages":[{"role":"user","content":"hi"}],"max_tokens":1}' \
        http://127.0.0.1:1236/v1/chat/completions 2>/dev/null)
      [ "$C" = "200" ] && break
      sleep 10
    done
  }
  "$PY" -X utf8 scripts/exp_gep_llm_formal.py
  [ -f results/exp_gep_llm_formal/exp_gep_llm_formal_verdict.json ] && { echo "正式批判决已落盘, 完成"; break; }
  sleep 20
done
echo "服务器保持运行 (PID $SRV_PID), 不用时请手动关闭"
