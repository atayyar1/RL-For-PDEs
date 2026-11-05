#!/bin/bash
# restart after the container restart: queues 3-5 (do not edit while running)
cd "$(dirname "$0")"
FAST=1 PDE=burgers PICARD=3 python evaluate.py > eval_burgers_lw_p3.out 2>&1
FAST=1 PDE=burgers GUESS=fd5:2 PICARD=3 python evaluate.py > eval_burgers_fd52_p3.out 2>&1
PDE=burgers GUESS=fd5:2 PICARD=3 python combo.py > combo_fd52_p3.out 2>&1
PDE=burgers PICARD=3 python combo.py > combo_lw_p3.out 2>&1
NROWS=4 PDE=burgers PICARD=3 python highorder.py > highorder_burgers_rows4_p3.out 2>&1
echo QUEUE6 DONE
