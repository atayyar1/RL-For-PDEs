# exploration/ — Joseph's research branch

Start with **[PROGRAM.md](PROGRAM.md)** — the question and the argument, book-shaped.
Then **[CHECKS.md](CHECKS.md)** — three standing checks, each of which produced a confident wrong
number here. Read it before trusting any measurement in this tree, including your own.

| file | what it is |
|---|---|
| `PROGRAM.md` | the research program: the question, the parts, what is proved / measured / open |
| `CHECKS.md` | **standing checks** — the three failure modes that produced wrong numbers here |
| `REEVALUATION.md` | why Phase 1 hit ten classical results, and the filter that would have predicted it |
| `PHASE2.md` | the compression phase, and why it closed |
| `FINDINGS.md` | numbered findings F1–F9 with the numbers and the caveats |
| `LOG.md` | headlines, newest first — what changed and when |
| `core/` | the shared library. Everything imports its grid and ground truth from here |
| `tests/test_core.py` | every verified finding as an assertion. `python tests/test_core.py` |
| `threads/T*/` | the exploratory threads, each with its own `RESULTS.md` |
| `figures/` | generated plots (regenerate with `make figures`) |

## Quick start

```bash
cd exploration
python tests/test_core.py          # 9/9 should pass
python -c "
from core import Problem, rows_taylor, solve_positive, positive_feasible, amplification
import numpy as np
p = Problem()
A, b = rows_taylor(np.array([-p.dx, 0, p.dx]), np.array([-p.dt]*3), p)
w = solve_positive(A, b)
print('FTCS weights', w, 'amplification', amplification(w))
"
```

## The one thing to know

A scheme is `(geometry, weights)`. Consistency fixes the weights, so **the geometry is the
only free choice**. And `sum(w) = 1` always, so

```
||w||_1 = 1   <=>   w >= 0   <=>   convex combination   <=>   errors accumulate linearly
||w||_1 > 1                                             <=>   errors accumulate geometrically
```

`||w||_1` is the currency of the whole program: it is the amplification factor, it is
computable without knowing the true solution, and it prices every trade between stability
and accuracy.

## Changes to the main codebase on this branch

The u_xx consistency row was corrected in 15 files (see `LOG.md`). Archives untouched.
Each edit carries an inline comment explaining the derivation.
