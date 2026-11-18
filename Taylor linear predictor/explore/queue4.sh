#!/bin/bash
# fourth batch: combined Burgers recipes (do not edit while running)
cd "$(dirname "$0")"
while pgrep -f "queue3.sh" > /dev/null || pgrep -f "queue2.sh" > /dev/null; do sleep 30; done
PDE=burgers PICARD=3 python combo.py > combo_lw_p3.out 2>&1
PDE=burgers GUESS=fd5:2 PICARD=3 python combo.py > combo_fd52_p3.out 2>&1
PDE=burgers GUESS=fd5:2 python combo.py > combo_fd52_p1.out 2>&1
echo QUEUE4 DONE
