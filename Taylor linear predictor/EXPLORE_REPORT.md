# Exploration report: backward maps with a "stop", robust scores, and RL

Branch `claude/nifty-lovelace-2ucxln` (the session's designated branch, used instead of `explore/<name>`).
Everything is in `explore/`. All numbers below come from the scripts named next to them. The exact solution is only
used for reporting (true error, exact bound), never for a choice.

| script | what it reproduces |
|---|---|
| `python explore/check_core.py` | the notebook checks (LW map identity, score correlation) |
| `python explore/evaluate.py` / `PDE=burgers python explore/evaluate.py` | Tables 1-3 (`results_*.json`, `evaluate_*.out`) |
| `python explore/pareto.py` / `PDE=burgers python explore/pareto.py` | Figures `pareto_advection.png`, `pareto_inflow_front.png` |
| `python explore/fig_footprint.py` | Figure `footprint.png` |
| `PDE=burgers python explore/burgers_robust.py` | the speed-robust score test (Section 5) |
| `python explore/rl_stop.py train ...`, `python explore/bc.py ...` | the PPO runs (Section 6) |

## 0. Setup and checks

* `explore/core.py` = cells 3, 4, 6, 8, 10, 12, 13/20 and the reward units of cell 15 of notebook 12.4, code as written
  (checks and pictures removed); `explore/env.py` = the cell-15 environment.
  `check_core.py`: **LW map at (10, 40): 1134 points, error 4.05e-3, identity 4.0505920081e-3 = 4.0505920081e-3;
  corr(log score, log true error) = 0.98.** Same as the notebook.
* Ali prefers a tougher problem, so `explore/core_burgers.py` extracts notebook 12.3 the same way with
  `PROBLEM = "inflow_front"` (Burgers, alpha = 0.01, T = 1.1, nt = 137: a front forms and leaves). Its LW identity also
  holds to all digits. Every rule below runs on both problems (`PDE=burgers`). The Burgers test queries are those of
  12.3.x: it* in {50, 75, 100} x ix* in {12, 25, 38}, plus 6 new ones.
* My baselines reproduce the table of the brief (advection, 9 unseen queries): greedy x0.47 points / x0.18 bound /
  x0.06 error, lowest score x1.15 / x0.17 / x0.08. The trained agent's checkpoint (`runs10-21/...`) is not in the
  repository, so I could not re-run it; its row is quoted from the brief. "Best fixed" here = the best of the 12 stencils
  most used by greedy / lowest score (an oracle pick by exact bound), which is better than the brief's (x0.98 / x0.70 /
  x0.53 instead of x1.15 / x1.81 / x1.48).

## 1. The stop action (main result)

**Mechanism** (`explore/stop.py`). A point p may skip its stencil and take the coarse-guess value g_p. It is then a leaf
like the IC, and the identity becomes

  e(z*) = sum_{expanded p} beta_p delta_p + sum_{stopped p} beta_p (u_p - g_p).

With `SAME_LEVEL = 0`, every user of p is on a higher level, so beta_p is final when p is popped: a rule can use it.
Check: the identity holds to 1e-16 on every map with stops (`identity_gap` in `results_*.json`; the only larger gaps,
~1e-9 relative, are on maps whose error is 1e3-1e6, i.e. round-off).

**Cost estimate without u_true.** Three LW guesses (2x, 4x, 8x coarser). The guess error is
|g2 - g4| / (R - 1), with R = median |g4 - g8| / median |g2 - g4| observed from the guesses (Richardson with an
observed order). R = 3.14 for advection (close to 4, clean 2nd order) and 2.06 for the Burgers front (pre-asymptotic).
With the textbook /3 the Burgers estimate was 3x too small (median est/true 0.36); with the observed R it is 0.89
(advection 1.21). The extra guesses cost 156 + 42 points (advection) and 364 + 98 (Burgers), shared by all queries.

**Rules.** `stop(c)`: stop when |beta_p| g_err(p) < c EPS / N_LW (a price per point, like lam).
`myopic stop` (greedy only): stop when |beta_p| g_err / EPS beats the best one-step stencil cost.

**Table 1. Advection, 9 unseen queries** (geometric mean of method / LW map of the same query; `evaluate_advection.out`)

| method | points | stops | exact bound | true error | est/bound | est_2g/bound | in Green band | wins |
|---|---|---|---|---|---|---|---|---|
| LW map | x1.00 | 0 | x1.00 | x1.00 | 1.03 | 1.10 | 0.40 | - |
| lowest score | x1.15 | 0 | x0.17 | x0.08 | 0.01 | 1.19 | 0.34 | 9/9 |
| greedy | x0.47 | 0 | x0.18 | x0.06 | 0.12 | 1.18 | 0.44 | 9/9 |
| agent (brief, not re-run) | x1.21 | - | x1.94 | x1.06 | - | - | - | 5/9 |
| best fixed (oracle) | x0.98 | 0 | x0.70 | x0.53 | 1.02 | - | - | 4/9 |
| **LW map + stop(c=1)** | **x0.44** | 54 | x1.02 | x1.00 | 0.99 | 1.06 | 0.81 | 5/9 |
| lowest score + stop(c=1) | x0.43 | 156 | x0.23 | x0.09 | 0.07 | 1.07 | 0.76 | 9/9 |
| greedy + stop(c=1) | x0.24 | 44 | x0.21 | x0.08 | 0.12 | 1.03 | 0.77 | 9/9 |
| **greedy + myopic stop** | **x0.24** | 31 | x0.20 | **x0.06** | 0.14 | 1.11 | 0.80 | 9/9 |
| greedy[2g] | x0.50 | 0 | x0.13 | x0.05 | 0.31 | 0.39 | 0.47 | 9/9 |
| greedy[2g] + myopic stop | x0.25 | 35 | x0.17 | x0.05 | 0.27 | 0.33 | 0.84 | 9/9 |

On the 6 new queries ((25,45) = the agent's training query, (6,50), (44,30), (25,15), (18,52), (33,20)) the picture is
the same: greedy + myopic stop x0.18 points / x0.08 error, LW + stop(c=1) x0.33 / x1.02, greedy x0.47 / x0.04.

**Table 2. Burgers front, 9 unseen queries** (`evaluate_burgers.out`)

| method | points | stops | exact bound | true error | est/bound | est_2g/bound | wins |
|---|---|---|---|---|---|---|---|
| LW map | x1.00 | 0 | x1.00 | x1.00 | 19.79 | 52.31 | - |
| lowest score | x1.08 | 0 | x25.75 | x1.79 | 0.00 | 2.10 | 4/9 |
| greedy | x0.52 | 0 | x25.45 | x1.87 | 0.00 | 2.05 | 3/9 |
| best fixed (oracle) | x1.13 | 0 | x2.85 | x2.57 | 7.62 | - | 2/9 |
| **LW map + stop(c=1)** | **x0.53** | 98 | x1.01 | **x0.99** | 19.64 | 51.88 | 1/9 |
| greedy + stop(c=0.1) | x0.41 | 56 | x25.35 | x1.41 | 0.01 | 2.08 | 4/9 |
| greedy + myopic stop | x0.38 | 45 | x25.33 | x1.97 | 0.01 | 2.07 | 2/9 |
| greedy[2g] | x0.85 | 0 | x74.25 | x13.15 | 0.51 | 0.63 | 1/9 |

**What worked.**
* The stop roughly **halves the points of every rule at unchanged true error**, on both problems, without training:
  LW x1 -> x0.44 (advection) / x0.53 (Burgers); greedy x0.47 -> x0.24 (advection). It is the only change in this
  report that helps on the Burgers front, where every score-driven rule loses to the LW map.
* Pareto sweeps (Figure `pareto_advection.png`): greedy + myopic stop dominates greedy at every lam
  (lam = 0.1: x0.36 points / x0.048 error, vs greedy's best x0.48 / x0.047). LW + stop(c) traces a flat curve from
  x0.61 to x0.26 points at x0.93-1.00 error.
* **Physical evidence** (Figure `footprint.png`, "in Green band" column). The fraction of expanded points inside the
  exact Green's function band (OU mean +- 2 std, clipped at the walls) goes from 0.40 (LW) / 0.44 (greedy) to
  0.81 / 0.80 with stops. In the figure, the stopped points (red crosses) sit on the edge of the band: the rule finds the
  domain of dependence from |beta| and the guess error alone. At (12, 40) the map collapses onto the left-wall strip that
  carries 95% of the weight; at (37, 25) onto the right strip. Where the band covers the whole domain ((25, 55)) there
  is little to cut (LW: 2064 -> 1730 points).

**What didn't.**
* **Rollout stops** (`explore/rollout.py`: one step of policy improvement; stop when the stop cost is below the base
  map's estimated cost-to-go, A_p |beta_p| + lam N_p, computed bottom-up) are spectacular on LW on advection:
  **x0.07 points at x1.08 error (lam = 1), x0.17 at x1.07 (lam = 0.1)**, because the LW score is calibrated
  (est/bound 1.03) so the cost-to-go is right. On greedy maps they lose to the myopic stop (greedy's score is 8x too
  optimistic), and on Burgers they are a disaster (x6-x80 error), because there the score overestimates LW's errors 20x,
  so every subtree looks expensive and everything is stopped. **Rollout is only as good as the score's calibration.**
* Policy iteration (round 2 of rollout) on LW un-did round 1 at first; that was a bug (stopped children were costed 0),
  fixed. Not swept further.

## 2. Score exploitation (item 3)

`explore/robust.py`. Two estimates that a cancellation cannot fool:

* **abs** = sum over Taylor terms of |moment x coefficient|. Useless: x400 too large even for LW. The Taylor terms
  cancel *legitimately* (the three conditions plus the PDE substitution make the low-order terms cancel exactly), so
  per-term absolute values measure nothing. Rules using it blow up (x10-x16 error on advection, x1e5 on Burgers).
* **2g** = max(score with the 2x guess, score with the 4x guess). A cancellation of guessed terms must hold for both
  guesses. On advection this **fixes the calibration of every rule map**: est/bound 1.10 (LW), 1.19 (lowest score),
  1.18 (greedy), 1.03-1.11 with stops, versus 0.01-0.12 for the plain score (Table 1, est_2g column). Rules that
  *optimise* 2g still exploit it a bit (est_2g/bound 0.15-0.39 instead of 0.01-0.12), but less, and they are more
  accurate: greedy[2g] x0.05 error / x0.13 bound vs greedy x0.06 / x0.18; on the new queries x0.02 vs x0.04.
  So 2g is a better reward for RL on advection and an honest evaluation metric there.
* On the Burgers front, 2g makes the rules much worse (greedy[2g] x13 error). The 4x guess has 13 points across a
  front of width ~alpha/Delta u ~ 0.01, so its derivatives are garbage, and the max picks them up.

## 3. Why Burgers (front) is hard for the score

Diagnosis at (25, 100) (`stop.py` / inline check): with the map's own weights (guessed speed, picard 0) LW and greedy
both have error ~1e-2; with the live-speed re-solve of the forward pass (picard 1) 4.8e-5 and 3.5e-5. So on Burgers
**the stencil that is scored (guessed speed) is not the stencil that is used (live speed)**. Greedy's cancelling choices
at the guessed speed give a bound 60x worse than LW's after the re-solve, even when (as here) its true error is lower.
The plain score overestimates LW's local errors 20x (est/bound 19.8) and underestimates the greedy maps' ~100x.

**Speed-robust score** (`explore/burgers_robust.py`): the worst score over speeds a_g, a_g +- D, D = estimated guess
error. On (25, 100) it was much worse (bound 1e-4 -> 5, error 3.5e-5 -> 2.8e-3), probably because the stencils that
survive the max are odd ones that amplify along the map. Not pursued; one query only.

## B. Burgers front (Ali's priority): what was tried after the first pass

All on `PROBLEM = "inflow_front"` (12.3), 9 unseen queries, ratios to the LW map, unless said otherwise.

**B1. Diagnosis: on Burgers the Taylor score is wrong in both directions** (inline check at (25, 100)):

| map | local error with map weights (guessed speed) | with live weights (forward pass) | score says |
|---|---|---|---|
| LW | 1.6e-4 | **1.2e-6** | 6e-5 (50x too high) |
| greedy | 2.6e-4 | 9.8e-5 | **1.1e-6** (100x too low) |

* The live-speed re-solve (picard 1) is what makes LW accurate: map error 1e-2 with picard 0, 4.8e-5 with picard 1.
* The score's high derivatives come from a spline through a coarse guess: median relative error of u_xxx 85%,
  u_xxxx 370% (`burgers_guess.py`). Greedy picks stencils that cancel these wrong derivatives, so its real local errors
  are 100x what it believes. **Fixing the policy (RL) cannot fix this; the estimate must change.**

**B2. Better guesses** (`burgers_guess.py`, `GUESS=fd5:K` / `hyb:K` switch in `stop.py`). FD-5 + RK4 on a coarse grid,
IC and wall data only. Values are much better (median u error 0.2% at 4x coarser vs 1.9% for the LW guess), high
derivatives are not (spline through 13-25 nodes).

| guess (cost) | LW + stop(c=1) points / error | greedy points / error | greedy + myopic stop |
|---|---|---|---|
| LW 2x (1475 + 462) | x0.53 / x0.99 | x0.52 / x1.87 | x0.38 / x1.97 |
| FD-5 2x (9600 + 3168) | **x0.38 / x0.84** | x0.53 / x5.96 | x0.29 / x5.36 |
| hybrid: FD-5 4x values + LW derivatives (3971 + 1326) | x0.48 / x1.12 | x0.54 / x4.83 | x0.35 / x4.25 |

A better guess helps the **stops** (better stop values: 62% fewer points than LW with 16% less error) but not the
score-driven rules, which get worse: the more accurate values change the speeds, but the derivative garbage remains.

**B3. Residual score** (`residual.py`): apply the stencil to the guess *function* (quintic-in-x, cubic-in-t spline of
the coarse solve), s = |G(p) - sum w G(q)|. Uses every order, no high derivatives. Calibration on 400 random stencils:

| estimate | Burgers corr / median est/true | advection corr / median est/true |
|---|---|---|
| Taylor score (cell 6) | 0.45 / 0.54 | **0.98 / 1.03** |
| residual on the LW 2x guess (same cost) | 0.58 / 1.14 | 0.69 / 1.03 |
| residual on the FD-5 2x guess | **0.73 / 1.01** | 0.93 / 1.00 |

Better calibrated than the Taylor score on Burgers, worse on advection (where the Taylor polynomial is the right
model). **But on maps it fails** (`res_eval.py`, Burgers unseen): greedy with the
residual score has x52-x111 true error, because rules that *minimise* it find stencils where it cancels
(est_res/bound 0.00 on their own maps). On advection greedy-residual is x0.29 error vs the Taylor greedy's x0.06.
**Lesson: any estimate that a rule minimises over ~18,000 stencils per point gets exploited, however well it is
calibrated on random stencils.** Calibration must be checked on the maps the rule builds, not on random stencils.

**B4. Reuse stops (Ali's KD-tree idea)** (`reuse.py`): a point takes a stencil made only of points the map already has
(lower levels, so still pending) or data, found by nearest-neighbour search (in the characteristic frame) in a box of
|dix| <= 8, 1-6 levels down; best admissible 5-subset of the 8 nearest by score. No new point, no guess value, identity
exact (gap 1e-16). Advection (25, 40): LW 1344 -> 1035 points with *lower* error (1.91e-3 -> 1.68e-3); greedy 649 -> 453
at the same error. Full tables (`reuse_*.out`):

| problem, unseen 9 | method | points | true error | exact bound | wins vs LW |
|---|---|---|---|---|---|
| advection | LW + reuse(c=10) | x0.67 | **x0.63** | x1.04 | 8/9 |
| advection | greedy + reuse | **x0.31** | x0.06 | x0.23 | 9/9 |
| advection | greedy + myopic stop + reuse | **x0.20** | x0.09 | x0.27 | 9/9 |
| Burgers | LW + reuse(c=1) | x0.61 | x1.23 (median better: 7/9 wins) | x2.37 | 7/9 |
| Burgers (new 6) | LW + reuse(c=1) | x0.67 | **x0.65** | x1.60 | 5/6 |
| Burgers | greedy + reuse | x0.38 | x2.03 | x25.6 | 2/9 |

Reuse cuts points and often *lowers* the error (a wider stencil of existing points, no guess value). It inherits the
score's trouble on Burgers when combined with greedy.

**B5. Semi-Lagrangian deep stencils** (`semilag.py`, no score): jump k levels, take the 5 cells nearest the foot of the
characteristic. Burgers, 2 queries: k = 3 halves the points; error x1.5 at (25, 100) and x11 at (12, 75) (where LW is
unusually accurate, 6.6e-6); bound only 1.5-3x LW's. A robust, score-free way to cut points, at some accuracy cost.
Full table (`semilag_burgers.out`, unseen 9): SL k=2 x0.66 points / x1.58 error,
SL k=3 x0.40 / x2.26, SL k=2 + stop(c=1) **x0.30 / x1.59**, SL k=3 + stop x0.22 / x2.25 (greedy: x0.52 / x1.87).
On the new 6: SL k=2 + stop x0.26 / **x0.75**. Better than greedy on Burgers without any score, but not more accurate
than LW on the unseen set.

**B6. The forward pass is not converged (free gain).** The plain LW map got 2x more accurate with the FD-5 guess,
although its stencil does not depend on the guess. Reason: one Picard step takes its speed from a value computed with
the guessed-speed weights. More steps (`PICARD=3`): (25, 100) 4.75e-5 -> 2.32e-5, (38, 50) 4.2e-4 -> 2.6e-4,
(25, 130) 6.4e-3 -> 2.7e-3; (12, 75) 6.6e-6 -> 3.6e-5 (the 1-step value was a lucky cancellation). Full evaluation (`eval_burgers_lw_p3.out`, `compare_guess.py`): the LW map with
PICARD=3 has **x0.46 true error** vs the notebook's LW map (7/9 wins), x0.32 on the new queries (6/6), same points.

**B7. One baseline for every guess / forward-pass setting** (`compare_guess.py`, against the notebook's LW map):
the best Burgers configuration so far is **LW + stop(c=1) with the FD-5 2x guess: x0.38 points at x0.44 true error,
7/9 wins (unseen); x0.32 / x0.52, 6/6 (new)**. With picard 0 everything is 3-80x worse: the live-speed re-solve is
essential on Burgers.

**Best Burgers recipe found (rules only)** (`combo.py`, `compare_guess.py`; all against the notebook's LW map):

| recipe | points | true error | exact bound | wins (unseen 9) | new 6: points / error / wins |
|---|---|---|---|---|---|
| notebook LW map (LW guess, picard 1) | x1.00 | x1.00 | x1.00 | - | x1.00 / x1.00 / - |
| LW map, PICARD=3 | x1.00 | x0.46 | x0.52 | 7/9 | x1.00 / x0.32 / 6/6 |
| LW + stop(c=1), PICARD=3 | x0.53 | x0.45 | x0.53 | 7/9 | x0.50 / x0.31 / 6/6 |
| **LW + stop(c=1), FD-5 2x guess, PICARD=3** | **x0.38** | **x0.40** | x0.55 | 7/9 | **x0.32 / x0.36 / 6/6** |
| LW + reuse(c=1) + stop(c=10), FD-5, PICARD=3 | x0.25 | x0.74 | x1.39 | 6/9 | x0.17 / x0.49 / 4/6 |
| greedy (any guess, PICARD=3) | x0.52 | x3.7-4.3 | x25 | 1-2/9 | |

**2.6x fewer points and 2.5x smaller error than the notebook's method, no training.** Cost note: the FD-5 2x guess
costs 9600 + 3168 grid points (vs 1937 for the LW guesses), shared by all queries; the LW map of one query is
3000-4000 points, so it pays off from about 5 queries on. With the LW guess (no extra cost) the recipe is x0.53 / x0.45.
Reuse does not help on Burgers once Picard is converged (its choice uses the Taylor score, which is wrong there).

**Fairness check against classical solvers** (`classical_burgers.py`, 9 unseen queries):

| method | cost (grid points) | median error at z* | error / notebook LW map |
|---|---|---|---|
| classical LW, full grid | 6800 (answers all queries) | 7.9e-3 | x55 |
| notebook LW map | ~2900 per query | 1.5e-4 | x1 |
| classical FD-5 + RK4, 2x coarser | 9600 (all queries) | 1.0e-4 | x0.62 |
| **LW + stop(c=1) + PICARD=3, LW guess** | ~1500 per query + 1937 shared guess | - | **x0.45** |
| LW + stop(c=1) + PICARD=3, FD-5 guess | ~1100 per query + 12768 shared guess | - | x0.40 |
| classical FD-5 + RK4, full grid | 39000 (all queries) | 4.9e-6 | x0.04 |

For a single query, the map with stops and converged Picard beats every classical option at lower cost (the 3-point
Taylor stencil with the live speed is 55x more accurate than the classical LW scheme on the same grid). But a 4th-order
classical solve on the full grid is still 10x more accurate and answers every query: **the map's accuracy ceiling is
its 2nd-order stencils**, which is why higher-order stencils that keep the variable-speed terms are the top next step.
Also: the stop sweep (`sweep_burgers.py`) shows that stopping almost everything (c = 100, x0.02 points) with the FD-5
guess gives x0.98 error, i.e. the FD-5 2x guess alone is about as accurate as the notebook's LW map; the map's value
there is halving the guess's error with ~x0.15-0.4 of the LW points.

**Stop-price sweep** (`sweep_burgers.py`, FD-5 guess, PICARD=3, figure `sweep_inflow_front_guess-fd52_picard3.png`):
LW + stop(c): c = 1 x0.38 points / x0.40 error, c = 10 x0.25 / x0.50; **semi-Lagrangian k=2 + stop(c = 3): x0.19 / x0.37,
c = 10: x0.15 / x0.41** (the best trade-off on Burgers, no score involved); SL k=3 stays at x1.1-1.4 error.

**B8. More Taylor conditions** (`highorder.py`, `NROWS` = 4 or 5 rows from exp(xi d + alpha dt d^2), 5 cells one
level down): W5 with 4 rows vs LW: (25, 100) 2.4e-5 vs 4.8e-5, (12, 75) 2.0e-5 vs 6.6e-6; advection about LW's.
Modest: the frozen-speed rows neglect the u u_x (Burgers) / a' (advection) terms, which cap the order. Full table (`highorder_burgers_rows4_p3.out`, PICARD=3, ratios to the PICARD=3 LW map):
W5 x1.13 points / x0.94 error (3/9 wins), W5 + stop x0.53 / x0.94. Not worth the formulation change as is.

**B8b. The variable-speed term in the u_x row** (`slope.py`). Cell 4 of the advection notebook corrects the u_x row
for a speed that varies across the stencil; 12.3's Burgers weights do not. For Burgers the right row is
xi - s DT (DX - a DT) with s = u_x (u_tt = 2 u u_x^2 + ...; the advection form has DX - a DT / 2 because there
u_tt = a a' u_x once). Copying the advection factor made the LW map 50x worse; with the Burgers factor it is an
**exact no-op for one-level stencils** (DT is the same for all neighbours, the row is xi (1 - s DT)), so LW maps do not
change; it only matters for multi-level stencils. Lesson: these rows must be derived per PDE.

**B9. Speed-robust score** (worst score over a_g, a_g +- guess error): much worse on (25, 100) (bound 1e-4 -> 5).
Dropped.

## 4. Depth / level skipping (item 4)

Not run as a separate experiment. Indirect evidence: greedy's saving (x0.47) comes from deep stencils, and the stop
saves points *on top of it* (x0.24), so the two are complementary: depth cuts points inside the band, the stop cuts the
band's edge.

## 5. RL (items 1-2, 5)

**Environment** (`explore/rl_stop.py`): cell 15's MaskablePPO environment plus one action, STOP (allowed on the first
pick of a point, never at z*). Reward of a stop: -|beta_p| g_err(p) / EPS; state + log10 of that stop cost. Same PPO
settings as the notebook (n_steps 1024, batch 1024, gae 0.97, lr 3e-4, ent 0), 4 envs, advection, query (25, 45),
lam = 1. Learning curves (deterministic policy at (25, 45)) in `runs/<name>/curve.json` and `runs_*.out`.

| run (1M steps each) | points | stops | true error at (25, 45) |
|---|---|---|---|
| PPO + stop | 1663 | **0** | 1.46e-3 |
| **PPO + stop + potential-based shaping** | **517** | 141 | **7.1e-4** |
| greedy + myopic stop (rule, for reference) | 535 | 49 | 3.5e-4 |

* **Without shaping the agent never stops** (0 stops at every checkpoint), exactly like the old agent never learned to
  save points. Credit assignment: a stop's cost is paid at once, its gain (a subtree that is never built) arrives as
  "fewer new points" spread over hundreds of later steps.
* **Potential-based shaping fixes it.** F = Phi(s') - Phi(s) with
  Phi(s) = -sum over pending points q of ( |beta_q| A_q / EPS + lam N_q / N_LW ), where A_q, N_q are q's cost-to-go
  (score per unit influence, subtree points) in the LW map of z* (`rollout.cost_to_go`, estimates only). Shaping of this
  form provably leaves the optimal policy unchanged (Ng et al. 1999); check: the shaped and unshaped returns of the same
  episode differ by exactly -Phi(s0) = 4.2430. The agent starts stopping at ~500k steps and reaches 517 points
  (x0.34 of LW) with half the unshaped run's error by 1M. **This is the first agent in the project that learns to save
  points.** It is still single-query and its error is 2x greedy+stop's at its own query. 

**It generalises** (trained on (25, 45) only; `python explore/rl_stop.py eval stop_shaped_25-45`, `runs/stop_shaped_25-45/eval.json`):

| advection | points | true error | exact bound | wins vs LW | in Green band |
|---|---|---|---|---|---|
| old agent (brief), unseen 9 | x1.21 | x1.06 | x1.94 | 5/9 | - |
| **shaped agent + stop, unseen 9** | **x0.07** | **x0.98** | x1.62 | 5/9 | **0.97** |
| shaped agent + stop, new 6 | x0.09 | x0.59 | x1.17 | 5/6 | 0.98 |
| greedy + myopic stop, unseen 9 | x0.24 | x0.06 | x0.20 | 9/9 | 0.80 |
| LW + rollout stop (lam = 1), unseen 9 | x0.07 | x1.08 | x1.69 | - | - |

14x fewer points than the LW map at the LW map's accuracy, on queries it never saw, with 97% of its points inside the
exact Green's function band: the footprint is physically right. It is at the "few points" end of the trade-off (like
LW + rollout stops), not more accurate than greedy; lam (the price of a point) sets where on the curve it sits.

Figure `explore/agent_footprint.png` (`python explore/fig_agent.py`) qualifies "97% in band": the points are *inside*
the band but do not *cover* it. At (12, 40) the map leans left with the band (physically right); at (25, 45) and
(25, 55) it is a straight column around z*, narrower than the band, closed by a wall of stops; at (37, 25) it stops
almost at once (a handful of points, error 5.8e-3, worse than LW). The agent has learned "stop away from z*", not yet
"follow the characteristic spreading".
* **Shaped PPO on the Burgers front fails** (`PDE=burgers SHAPING=1 python explore/rl_stop.py train burgers_shaped_25-75
  1e6 25,75 lw`, accuracy normalised by the LW map's score sum): 0 stops over 1M steps, maps the size of LW's (3379
  points), error at its own query 1-3e-3 vs the LW map's ~1e-5. Consistent with B1: the reward is built on the Taylor
  score, which is wrong on Burgers, and the agent games it as greedy does. Shaping cannot repair a wrong reward.
  RL on Burgers needs a trustworthy local-error estimate first (or a restricted, score-free action set:
  {LW, semi-Lagrangian k, stop}, see next steps).
* The behaviour-cloning warm start (`bc.py`) was written and its teacher episodes checked (valid under the mask) but
  not run: Ali reported earlier trouble with copying greedy, and the shaping result made it less urgent.


## 6. Ranked next steps

NEXT_STEPS
