# K100_AI air-cooled TunableOp profiles

These four files contain the validated per-device rocBLAS selections for a
four-way tensor-parallel K100_AI air-cooled deployment. PyTorch reads the
TunableOp environment before HIP initialization, so select this profile in the
launch environment rather than from an imported SGLang module:

```bash
export PYTORCH_TUNABLEOP_ENABLED=1
export PYTORCH_TUNABLEOP_FILENAME="$SGLANG_REPO/python/sglang/srt/hcu_optim/config/tunableop/air/union%d.csv"
```

`%d` is replaced by the process device index. Do not enable this profile on a
liquid-cooled system or a different GPU/DTK/hipBLASLt stack. Leaving both
variables unset preserves PyTorch's default selection behavior.
