# Conv1D + GDN + MoE manual reproduction

The commands below reproduce the profile validated on node 86 with GPUs
`4,5,6,7`, model `/model/Qwen3.6-35B-A3B`, and port `30010`.

## 1. Get the cumulative branch

```bash
git clone https://github.com/Thomafabien/sglang-das.git /workspace/sglang-das-k100opt
cd /workspace/sglang-das-k100opt
git fetch origin feat/qwen36-k100ai-opt
git switch --track origin/feat/qwen36-k100ai-opt
git log -3 --oneline
```

The top three optimization commits should be Conv1D, GDN, and MoE.

If the repository already exists:

```bash
cd /workspace/sglang-das-k100opt
git switch feat/qwen36-k100ai-opt
git pull --ff-only origin feat/qwen36-k100ai-opt
```

## 2. Stop the previous service

Check the exact process first:

```bash
ps -ef | grep '[s]glang.launch_server'
```

Then stop that service gracefully:

```bash
pkill -TERM -f 'python3 -u -m sglang.launch_server'
```

## 3. Start the cumulative profile

```bash
mkdir -p /workspace/k100opt/results/manual-gdn-moe
cd /workspace/sglang-das-k100opt

nohup env \
  GPU_DEVICES=4,5,6,7 \
  MODEL_PATH=/model/Qwen3.6-35B-A3B \
  PORT=30010 \
  bash k100opt/start_gdn_moe_server.sh \
  > /workspace/k100opt/results/manual-gdn-moe/service.log 2>&1 &

echo $!
tail -f /workspace/k100opt/results/manual-gdn-moe/service.log
```

The script sets the complete verified environment, including:

- `/opt/dtk/.hyhal/rocm_smi/lib` in `LD_LIBRARY_PATH`;
- `k100opt/patch` before `python/` on `PYTHONPATH`;
- Conv1D and the three GDN switches;
- the four MoE switches and `SGLANG_ACT_FLAT_MIN=20000`;
- `SGLANG_MOE_CONFIG_DIR` pointing to this branch's K100_AI tables.

Do not also add `/workspace/k100opt/patch` to `PYTHONPATH`; that would load the
customer's full optimization layer and contaminate this incremental test.

## 4. Health and effective-code checks

```bash
curl -fsS http://127.0.0.1:30010/v1/models

grep -E \
  'Using MoE kernel config|fast_moe_align|moe_pack_w13|act_flat' \
  /workspace/k100opt/results/manual-gdn-moe/service.log
```

Before performance testing, the log should show the K100_AI config, successful
fast-align guard, and packed-w13 execution.  `act_flat` will report an effective
large shape only after a sufficiently large prefill request.

## 5. Run two smoke rounds

The benchmark client must override the container defaults `PORT=30000` and
`MODEL_PATH=/models/...`:

```bash
export LD_LIBRARY_PATH=/opt/dtk/.hyhal/rocm_smi/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/workspace/sglang-das-k100opt/python
export PORT=30010
export MODEL_PATH=/model/Qwen3.6-35B-A3B

cd /workspace/k100opt/bench

python3 smoke.py gdn-moe-first-discard 2>&1 \
  | tee /workspace/k100opt/results/manual-gdn-moe/smoke-first-discard.log

python3 smoke.py gdn-moe-second-record 2>&1 \
  | tee /workspace/k100opt/results/manual-gdn-moe/smoke-second-record.log
```

Use the second round as the formal result.  The node-86 reference for that round
was TTFT `163.8 / 2459.2 / 1666.6 ms` and TPOT
`4.760 / 13.910 / 4.900 ms` for rows 4, 13, and 29 respectively.

## 6. Roll back only MoE

Keep the same source and set the four performance switches to zero before
starting the script:

```bash
SGLANG_LIGHTOP_COMPAT=0 \
SGLANG_FAST_MOE_ALIGN=0 \
SGLANG_ACT_FLAT=0 \
SGLANG_MOE_PACK_W13=0 \
bash k100opt/start_gdn_moe_server.sh
```

This keeps Conv1D and GDN enabled while disabling the runtime MoE layer.
