# Qwen3.6 K100_AI incremental optimizations

This directory contains the K100_AI MoE runtime layer and the tuned MoE
configurations used together with the Conv1D and GDN source integrations on
branch `feat/qwen36-k100ai-opt`.

## Start the verified GDN + MoE profile

Inside the model container:

```bash
cd /workspace/sglang-das-k100opt
GPU_DEVICES=4,5,6,7 \
MODEL_PATH=/model/Qwen3.6-35B-A3B \
PORT=30010 \
bash k100opt/start_gdn_moe_server.sh \
  > /workspace/k100opt/results/gdn4-moe-manual/service.log 2>&1
```

The MoE layer is isolated under `k100opt/patch`.  It is loaded only because the
start script puts that directory before `python/` on `PYTHONPATH` and explicitly
enables the four MoE switches.  A normal SGLang launch from this branch does not
enable the runtime patches.

## Effective-code checks

After the service is ready, the service log must contain all of the following:

- `Using MoE kernel config from ...device_name=K100_AI.json`
- `[fast_moe_align] 守卫判定 EP 已关闭 = True`
- `[moe_pack_w13] ✅ 首次真正生效`
- `[act_flat] 生效·新形状` after a large-prefill request

`m3_fuse` is intentionally excluded because its target is `qwen2_moe`.
`safe_align` is intentionally excluded because it belongs to the later
pair-prefill scheduling integration.
