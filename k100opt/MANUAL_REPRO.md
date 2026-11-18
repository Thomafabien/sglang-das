# K100_AI 源码合入手工复现

## 1. 获取分支

```bash
git clone https://github.com/Thomafabien/sglang-das.git /workspace/sglang-das-k100opt
cd /workspace/sglang-das-k100opt
git fetch origin feat/qwen36-k100ai-opt
git switch --track origin/feat/qwen36-k100ai-opt
git status --short
```

已有仓库且工作树干净时：

```bash
git switch feat/qwen36-k100ai-opt
git pull --ff-only origin feat/qwen36-k100ai-opt
```

记录代码身份：

```bash
git rev-parse HEAD
git rev-parse 'HEAD^{tree}'
python3 k100opt/verify_source_integration.py
```

## 2. clean baseline

新 shell 中不要设置任何 K100_AI 优化一级变量：

```bash
cd /workspace/sglang-das-k100opt
export HIP_VISIBLE_DEVICES=4,5,6,7
export LD_LIBRARY_PATH=/opt/dtk/.hyhal/rocm_smi/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/workspace/sglang-das-k100opt/python

python3 -u -m sglang.launch_server \
  --model-path /model/Qwen3.6-35B-A3B \
  --tokenizer-path /model/Qwen3.6-35B-A3B \
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
  --speculative-algorithm EAGLE --speculative-num-steps 3 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  --tp 4 --disable-radix-cache --trust-remote-code \
  --disable-custom-all-reduce --page-size 64 \
  --numa-node 0 1 2 3 --host 0.0.0.0 --port 30010
```

## 3. 增量或全量优化

已验证的 Conv1D+GDN+MoE：

```bash
GPU_DEVICES=4,5,6,7 MODEL_PATH=/model/Qwen3.6-35B-A3B PORT=30010 \
  bash k100opt/start_gdn_moe_server.sh
```

全模块候选集成：

```bash
GPU_DEVICES=4,5,6,7 MODEL_PATH=/model/Qwen3.6-35B-A3B PORT=30010 \
  bash k100opt/start_all_optimizations.sh
```

单项 A/B 时，在 clean 命令前只设置目标一级开关。完整开关、代码位置和保护条件见 `SOURCE_INTEGRATION.md`。

## 4. 日志与 smoke

```bash
mkdir -p /workspace/k100opt/results/source-integration
nohup env GPU_DEVICES=4,5,6,7 MODEL_PATH=/model/Qwen3.6-35B-A3B PORT=30010 \
  bash k100opt/start_all_optimizations.sh \
  > /workspace/k100opt/results/source-integration/service.log 2>&1 &

curl -fsS http://127.0.0.1:30010/v1/models

export PORT=30010
export MODEL_PATH=/model/Qwen3.6-35B-A3B
cd /workspace/sglang-das-k100opt/k100opt/bench
python3 smoke.py source-all-first-discard 2>&1 \
  | tee /workspace/k100opt/results/source-integration/smoke-first-discard.log
python3 smoke.py source-all-second-record 2>&1 \
  | tee /workspace/k100opt/results/source-integration/smoke-second-record.log
```

第一遍只用于热形状；第二遍作为正式数据，同时与 smoke 中固定的 clean、full-opt 两个端点比较。

## 5. 停止与回退

先确认进程，再优雅停止：

```bash
ps -ef | grep '[s]glang.launch_server'
pkill -TERM -f 'python3 -u -m sglang.launch_server'
```

最可靠的全回退方式是启动新 shell，不设置任何优化一级开关。单项回退可将对应变量设为 `0`，无需换源码或恢复文件。
