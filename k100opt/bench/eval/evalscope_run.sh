#!/usr/bin/env bash
# 按实验室口径（evalscope，chat 接口、思考模式、temperature 1.0、top_p 0.95、max_tokens 163840、批 16）评测三项精度。
# 对一个已经启动好的服务执行；三个数据集并行评测（各自批 16）。
#   用法: bash bench/eval/evalscope_run.sh <输出目录> [gsm8k题数 humaneval题数 math_500每个难度档的题数]
#   例:   bash bench/eval/evalscope_run.sh /tmp/acc_final            # 全量
#         bash bench/eval/evalscope_run.sh /tmp/acc_final 100 80 16   # 抽样（math_500 共 5 档，16 即 80 题）
# 对照方法：同一批题、同一 --seed，分别对「厂商默认配置的服务」与「本仓库配置的服务」各跑一次；
#           只想看通信量化的影响时，可在同一实例上 touch /tmp/qar_off 关闭量化后再跑一次（测完务必删除该文件）。
# 依赖: pip install evalscope（建议装到独立目录，避免改动被测环境:
#       pip install --target /opt/es_pkgs evalscope，再以 ES_PKGS=/opt/es_pkgs 运行本脚本）
set -u
OUT=${1:?缺输出目录}; L1=${2:-}; L2=${3:-}; L3=${4:-}
MODEL_PATH=${MODEL_PATH:-/workspace/models/Qwen3.6-35B-A3B}; HOST=${HOST:-127.0.0.1}; PORT=${PORT:-8023}
ES="python3 -m evalscope.cli.cli"; [ -n "${ES_PKGS:-}" ] && ES="env PYTHONPATH=$ES_PKGS $ES"
mkdir -p "$OUT"
run () { local ds=$1 lim=$2
  $ES eval --model "$MODEL_PATH" --api-url "http://$HOST:$PORT/v1/chat/completions" --eval-type openai_api \
    --generation-config '{"max_tokens":163840, "temperature": 1.0, "top_p": 0.95, "timeout": 30000}' --eval-batch-size 16 --seed 42 \
    ${lim:+--limit $lim} --work-dir "$OUT/$ds" --datasets "$ds" > "$OUT/$ds.log" 2>&1; }
run gsm8k "$L1" & run humaneval "$L2" & run math_500 "$L3" & wait
python3 - "$OUT" <<'P'
import sys, json, glob
for f in sorted(glob.glob(sys.argv[1] + "/*/*/reports/*/*.json")):
    d = json.load(open(f)); m = d["metrics"][0]; print(f"{d.get('dataset_name', f.split('/')[-1]):<12} 得分 {m['score']:.4f}  题数 {m['num']}")
P
