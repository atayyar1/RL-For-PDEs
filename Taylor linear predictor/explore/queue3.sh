#!/bin/bash
# third batch: converged Picard (3 steps) with the LW and FD-5 guesses (do not edit while running)
cd "$(dirname "$0")"
while pgrep -f "queue2.sh" > /dev/null; do sleep 30; done
FAST=1 PDE=burgers PICARD=3 python evaluate.py > eval_burgers_lw_p3.out 2>&1
FAST=1 PDE=burgers GUESS=fd5:2 PICARD=3 python evaluate.py > eval_burgers_fd52_p3.out 2>&1
echo QUEUE3 DONE
