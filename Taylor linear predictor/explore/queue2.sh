#!/bin/bash
# second batch (do not edit while running: bash reads scripts as it goes)
cd "$(dirname "$0")"
while pgrep -f "python res_eval.py" > /dev/null; do sleep 20; done
PDE=burgers python reuse.py > reuse_burgers.out 2>&1
PDE=burgers python res_eval.py > res_eval_burgers.out 2>&1
python reuse.py > reuse_advection.out 2>&1
NROWS=4 PDE=burgers python highorder.py > highorder_burgers_rows4.out 2>&1
echo QUEUE2 DONE
