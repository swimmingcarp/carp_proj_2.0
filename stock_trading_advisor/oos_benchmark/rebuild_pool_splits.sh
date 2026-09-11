#!/bin/sh
# pool_b 的两个半区是指向 pool_b 的符号链接目录，不入库（*.csv 白名单会让它们半入半不入）。
# 克隆仓库后运行本脚本即可重建，随后 research/harness.py --pool pool_b_dev / pool_b_acc 可用。
set -e
cd "$(dirname "$0")/../data"
for half in dev acc; do
  rm -rf "pool_b_$half"; mkdir -p "pool_b_$half"
  while read -r code; do
    [ -n "$code" ] && ln -sf "../pool_b/${code}_hfq.csv" "pool_b_$half/${code}_hfq.csv"
  done < "../oos_benchmark/pool_b_$half.txt"
  echo "pool_b_$half: $(ls "pool_b_$half" | wc -l) symlinks"
done
