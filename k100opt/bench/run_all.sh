#!/usr/bin/env bash
# 一条命令完成：开测前自检 → 启动服务 → 生效性自证 → 测量 → 评分 → 停服务
#
# 由快到慢四级，建议按顺序做，前一级正常再做后一级：
#   bash bench/run_all.sh smoke    冒烟：3 个测点，约 8 分钟（含启动）。确认服务正常、量级对
#   bash bench/run_all.sh quick    抽测：10 个代表测点 × 1 遍，约 12 分钟。逐行与基准对照，看哪一档偏了
#   bash bench/run_all.sh once     单遍全表：45 测点 × 1 遍，约 35 分钟。O/P 的单遍值（「O 比三遍低约 1 点」是 R16 时期的观察，R18 后未复核）
#   bash bench/run_all.sh acc      精度快测：gsm8k 全量（自测口径），约 5 分钟
#   bash bench/run_all.sh full     正式：45 测点 × 3 遍，约 110 分钟。正式成绩
# 每一级都会先自检环境、启动服务、核对补丁装载与开关，测完打印与本仓库基准值的逐行对照。
#
# 环境变量：MODEL_PATH / SGLANG_PYTHON / HOST / PORT / OUTDIR；其余环境变量原样传给 start_server.sh
# 本脚本自己启动服务（调用 start_server.sh），调用前不要手动起服务，否则会因检测到残留进程而中止。
# 风冷平台：HIP_VISIBLE_DEVICES=4,5,6,7 SGLANG_HW_PROFILE=air bash bench/run_all.sh once
#   环境变量原样传给 start_server.sh（SGLANG_HW_PROFILE=air 即 start_server_air.sh 的效果）；开测前的空闲检查按 HIP_VISIBLE_DEVICES 所列的卡；
#   期望的补丁装载记录为 25 项；逐行对照的基准为 bench/baseline_air.json（score45.py 按 SGLANG_HW_PROFILE 选取）。
set -u

MODE="${1:-}"
case "$MODE" in
  smoke|quick|once|full|acc) ;;
  *) echo "用法: bash $0 {smoke|quick|once|full|acc}"; exit 1 ;;
esac

BENCH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$BENCH")"
SGLANG_PYTHON="${SGLANG_PYTHON:-/workspace/sglang-das/python}"
MODEL_PATH="${MODEL_PATH:-/workspace/models/Qwen3.6-35B-A3B}"
HOST="${HOST:-127.0.0.1}"; PORT="${PORT:-8023}"
OUTDIR="${OUTDIR:-$BENCH/out}"
LOG="$OUTDIR/server.log"
LOCK="$BENCH/.running"
mkdir -p "$OUTDIR/results"

die () { echo "[中止] $*"; exit 1; }
CARDS="${HIP_VISIBLE_DEVICES:-0,1,2,3}"                    # 本次服务所用的卡，空闲检查按它
if [ "${SGLANG_HW_PROFILE:-}" = "air" ]; then EXPECT=25; else EXPECT=22; fi   # 风冷档位另有 moe_pack_w13、conv1d_cfg、dense_pack
# 注意：容器里 pgrep -f / pkill -f 会匹配到承载本脚本的 shell 自身，必须用 ps + awk 过滤
ourpids () { ps -eo pid,cmd | awk '/sglang\.launch_server|sglang\.benchmark\.serving|run45\.py/ && !/awk/ && !/bash .*run_all/ {print $1}'; }

echo "════════ 0. 开测前自检"
[ -e "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null \
  && die "已有一个实例在跑（pid=$(cat "$LOCK")）。两个压测并跑会造成随测点推进累积的劣化。"
echo $$ > "$LOCK"; trap 'rm -f "$LOCK"' EXIT

LEFT="$(ourpids | tr '\n' ' ')"
[ -n "${LEFT// /}" ] && die "检测到残留的服务/压测进程：$LEFT　请先清理。"

[ -d "$SGLANG_PYTHON/sglang/benchmark" ] || die "SGLANG_PYTHON 不含 sglang/benchmark：$SGLANG_PYTHON"
[ -d "$MODEL_PATH" ] || die "模型路径不存在：$MODEL_PATH"

if command -v hy-smi >/dev/null 2>&1; then
  CARDSEL='BEGIN{n=split(cards,a,",");for(i=1;i<=n;i++)s[a[i]]=1} $1 ~ /^[0-9]+$/ && ($1 in s)'
  BUSY=$(hy-smi 2>/dev/null | awk -v cards="$CARDS" "$CARDSEL"'{gsub("%","",$7); if ($7+0 > 5) c++} END{print c+0}')
  [ "${BUSY:-0}" != "0" ] && echo "[警告] 有 $BUSY 张卡 HCU 利用率 >5%，机器可能非独占，测量结果可能不可比"
  hy-smi 2>/dev/null | awk -v cards="$CARDS" "$CARDSEL"'{printf "  卡%s 温度%s 功耗%s HCU%s\n",$1,$2,$3,$7}'
fi
echo "  通过。模式=$MODE  输出=$OUTDIR  卡=$CARDS  硬件档位=${SGLANG_HW_PROFILE:-（未设，液冷）}"

echo "════════ 1. 启动服务"
env MODEL_PATH="$MODEL_PATH" SGLANG_PYTHON="$SGLANG_PYTHON" HOST="$HOST" PORT="$PORT" \
  nohup bash "$REPO/start_server.sh" > "$LOG" 2>&1 &
disown $! 2>/dev/null || true            # 否则停服务时 shell 会打印一行「Killed」，看着像出错
for i in $(seq 1 60); do
  grep -q "fired up and ready" "$LOG" 2>/dev/null && break
  grep -qE "Traceback|OutOfMemoryError|VMFault" "$LOG" 2>/dev/null && { tail -20 "$LOG"; die "服务启动失败，见 $LOG"; }
  sleep 10
done
grep -q "fired up and ready" "$LOG" || { tail -20 "$LOG"; die "服务 600 秒内未就绪，见 $LOG"; }
echo "  已就绪"

echo "════════ 2. 生效性自证"
grep -m1 '^\[switches\]' "$LOG" | tr ' ' '\n' | grep -E '^(SGLANG_|PYTORCH_TUNABLEOP|GPU_MAX_HW|NCCL_)' | sed 's/^/  /'
N=$(grep -hoE "\[[a-z_0-9]+\] 已安装" "$LOG" | sort -u | wc -l)
echo "  补丁装载记录 $N 项（期望 $EXPECT 项）："
grep -hoE "\[[a-z_0-9]+\] 已安装" "$LOG" | sort -u | sed 's/^/    /'
[ "$N" -lt "$EXPECT" ] && echo "  [警告] 装载记录少于 $EXPECT 项。缺 fast_import 会使并发 16 的测点劣化约 10%"
echo "  MoE 配置来源："
grep -ho "Using MoE kernel config from .*" "$LOG" | sort -u | sed 's/^/    /'
grep -m1 '^\[hw_profile\]' "$LOG" | sed 's/^/  /'
[ "${SGLANG_HW_PROFILE:-}" = "air" ] && ! grep -q '^\[hw_profile\]' "$LOG" && echo "  [警告] 设了 SGLANG_HW_PROFILE=air 但日志中没有 [hw_profile] 一行"

echo "════════ 3. 测量"
cd "$BENCH"
if [ "$MODE" = "acc" ]; then
  python3 eval/gsm8k_full.py 2>&1 | tee "$OUTDIR/acc.txt"
  echo "  参考：交付配置在该自测口径下 gsm8k 约 95.7–96.2%（关思考、temperature 0）。正式口径见 bench/eval/README.md"
elif [ "$MODE" = "smoke" ]; then
  PYTHONPATH="$SGLANG_PYTHON" MODEL_PATH="$MODEL_PATH" HOST="$HOST" PORT="$PORT" \
    python3 smoke.py "$(date +%m%d-%H%M)" 2>&1 | grep -v -E "^(WARNING|INFO|/usr/local|  return self|\[aiter\])"
else
  case "$MODE" in
    quick) export PASSES=1 ROWS="4,10,13,20,24,29,31,34,39,44" ;;   # 覆盖 1024–131072 各输入档与并发 1/2/4/16
    once)  export PASSES=1 ROWS="" ;;
    full)  export PASSES=3 ROWS="" ;;
  esac
  PYTHONPATH="$SGLANG_PYTHON" MODEL_PATH="$MODEL_PATH" HOST="$HOST" PORT="$PORT" OUTDIR="$OUTDIR" \
    python3 run45.py || die "测量中断"
  echo "════════ 4. 评分"
  PYTHONPATH="$SGLANG_PYTHON" OUTDIR="$OUTDIR" python3 score45.py | tee "$OUTDIR/score.txt"
fi

echo "════════ 5. 停服务"
ourpids | xargs -r kill -9 2>/dev/null
sleep 3
wait 2>/dev/null
echo "  完成。服务日志 $LOG"
[ "$MODE" != "smoke" ] && [ "$MODE" != "acc" ] && echo "  评分结果 $OUTDIR/score.txt"
if [ "$MODE" = "full" ] || [ "$MODE" = "once" ]; then
  if [ "${SGLANG_HW_PROFILE:-}" = "air" ]; then echo "  参照：风冷 R21 + 风冷档位，交付方式单遍 O 119.94 / P 144.07（4–7 号卡）"
  else echo "  参照：液冷 R21 全表单遍 O 130.08 / P 160.64（同一批压测输入，CPU 调频 performance、GPU 性能档位 auto）"; fi
fi
[ "$N" -lt "$EXPECT" ] && echo "  [注意] 补丁装载记录只有 $N 项（应为 $EXPECT 项），上面的成绩不是交付配置的成绩，请先解决装载问题"
exit 0
