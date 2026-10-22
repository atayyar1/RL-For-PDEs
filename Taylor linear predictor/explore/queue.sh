#!/bin/bash
# runs the Burgers experiments one after the other (each uses 4 processes)
cd "$(dirname "$0")"
PDE=burgers python res_eval.py > res_eval_burgers.out 2>&1
for spec in "fd5:2 1" "hyb:4 1" "fd5:2 0" "lw 0"; do
  set -- $spec
  FAST=1 PDE=burgers GUESS=$1 PICARD=$2 python evaluate.py > "eval_burgers_${1/:/}_p$2.out" 2>&1
done
python res_eval.py > res_eval_advection.out 2>&1
