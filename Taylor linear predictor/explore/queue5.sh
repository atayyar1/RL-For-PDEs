#!/bin/bash
# fifth batch (do not edit while running)
cd "$(dirname "$0")"
while pgrep -f "queue4.sh" > /dev/null; do sleep 30; done
NROWS=4 PDE=burgers PICARD=3 python highorder.py > highorder_burgers_rows4_p3.out 2>&1
PDE=burgers PICARD=3 python highorder.py > highorder_burgers_rows3_p3.out 2>&1
echo QUEUE5 DONE
