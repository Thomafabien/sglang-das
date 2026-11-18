#!/usr/bin/env bash
# 编译自写分页注意力内核，产物输出到 hcu_optim/lib/。
# 用法: bash native/build.sh [输出目录]   默认输出目录为 ../lib
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:-$HERE/../lib}"
H="${HIPCC:-/opt/dtk/bin/hipcc}"
F=(-O3 --offload-arch=gfx928 -ffp-contract=off -shared -fPIC)
mkdir -p "$OUT"; cd "$HERE"
for SQ in 4 1; do
  "$H" "${F[@]}" -DQFMT=4 -DPA_SQ=$SQ pa_v5.cpp -o "$OUT/libpa_e4m3_q$SQ.so"
  "$H" "${F[@]}" -DQFMT=5 -DPA_SQ=$SQ pa_v5.cpp -o "$OUT/libpa_e5m2_q$SQ.so"
done
"$H" "${F[@]}" -DQFMT=4 pp_v5.cpp -o "$OUT/libpp_v5_q4.so"
"$H" "${F[@]}" -DQFMT=5 pp_v5.cpp -o "$OUT/libpp_v5_q5.so"
"$H" "${F[@]}" -DQFMT=4 pa_ext.cpp -o "$OUT/libpa_ext_e4m3.so"
"$H" "${F[@]}" -DQFMT=5 pa_ext.cpp -o "$OUT/libpa_ext_e5m2.so"
echo "已生成到 $OUT:"; ls -la "$OUT"/libpa_*.so "$OUT"/libpp_*.so | awk '{printf "  %8d  %s\n", $5, $9}'
echo "资源用量核对（预期 pa_v5 VGPR 107 / LDS 29952，pp_v5 VGPR 162 / LDS 20736）:"
for s in "pa_v5.cpp -DPA_SQ=4" "pp_v5.cpp"; do
  set -- $s; src=$1; shift
  "$H" -O3 --offload-arch=gfx928 -ffp-contract=off -S --cuda-device-only -DQFMT=4 "$@" "$src" -o /tmp/_k.s 2>/dev/null
  printf "  %-12s %s\n" "$src" "$(grep -oE '(\.vgpr_count|\.vgpr_spill_count|\.group_segment_fixed_size): *[0-9]+' /tmp/_k.s | tr '\n' ' ')"
done
