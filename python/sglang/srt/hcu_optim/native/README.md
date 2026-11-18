# 自写分页注意力内核

本轮自研的 GPU 内核由 `sglang.srt.hcu_optim.kv.fp8_kv_hcu` 按显式开关以 ctypes 加载。预编译 `.so` 位于相邻的 `../lib`，本目录保留源码与构建方式。

## 文件

| 文件 | 用途 | 对应 `.so` |
|---|---|---|
| `pa_v5.cpp` | decode 分页注意力。`-DPA_SQ=4` 为验证步、`-DPA_SQ=1` 为草稿步；`-DQFMT=4` 为 e4m3、`=5` 为 e5m2 | `libpa_{e4m3,e5m2}_q{4,1}.so` |
| `pp_v5.cpp` | prefill 分页注意力 | `libpp_v5_q{4,5}.so` |
| `pa_ext.cpp` | 变长 q（≤4）的 split-KV 变体，默认不启用（`SGLANG_PA_EXT=1` 开启） | `libpa_ext_{e4m3,e5m2}.so` |

## 编译

```bash
bash build.sh                  # 一键编译全部变体到 ../lib，并打印资源用量
```

等价的手工命令：

```bash
H=/opt/dtk/bin/hipcc
F="-O3 --offload-arch=gfx928 -ffp-contract=off -shared -fPIC"
cd python/sglang/srt/hcu_optim/native
for SQ in 4 1; do
  $H $F -DQFMT=4 -DPA_SQ=$SQ pa_v5.cpp -o ../lib/libpa_e4m3_q$SQ.so
  $H $F -DQFMT=5 -DPA_SQ=$SQ pa_v5.cpp -o ../lib/libpa_e5m2_q$SQ.so
done
$H $F -DQFMT=4 pp_v5.cpp -o ../lib/libpp_v5_q4.so
$H $F -DQFMT=5 pp_v5.cpp -o ../lib/libpp_v5_q5.so
$H $F -DQFMT=4 pa_ext.cpp -o ../lib/libpa_ext_e4m3.so
$H $F -DQFMT=5 pa_ext.cpp -o ../lib/libpa_ext_e5m2.so
```

查看资源用量（VGPR、溢出、LDS）：

```bash
$H -O3 --offload-arch=gfx928 -ffp-contract=off -S --cuda-device-only -DQFMT=4 -DPA_SQ=4 pa_v5.cpp -o pa_v5.s
grep -oE '(\.vgpr_count|\.vgpr_spill_count|\.group_segment_fixed_size): *[0-9]+' pa_v5.s
```

预期：`pa_v5` VGPR 107、无溢出、LDS 29952 B；`pp_v5` VGPR 162、无溢出、LDS 20736 B。

## 改动后必须重跑的检查

最低限度应覆盖：`seqlen mod 64` 全余数类、不等长 batch 的逐位对拍，以及服务内长上下文检索。仅用整页长度对拍会漏掉尾页同步缺陷。
