#!/bin/bash
# Regenerate every result and figure in this thread, logging each task.
# Task 4 is the slow one (~8 min): 9 rollouts x 1000 levels x 98 points.
set -e
cd "$(dirname "$0")"
mkdir -p logs figures
for t in task1_certificate task2_cfl task3_geography task5_cost task4_rollout; do
  echo "=== $t ==="
  python3 -u "$t.py" 2>&1 | tee "logs/$t.log"
done
echo "done -- logs/ and figures/ updated"
