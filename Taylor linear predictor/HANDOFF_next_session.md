# Handoff — where the project stands (2026-09-29)

For continuing in a new conversation. Read this first, then `refinement_formulation.md` for
the original design doc formulation 3 is built from.

## The three formulations

| | Notebook | Status |
|---|---|---|
| 1. Walker RL | `6.3 RLdiff advection.ipynb` | Abandoned — learned reach/speed but never accuracy, even after 50M steps. Terminal-only reward, no field values in the state. |
| 2. Backward map | `8.3 Backward Implementation.ipynb` | Working but unfinished. Terminal reward, autoregressive 5-neighbour picks per point. t\*=5 run converged to a clean 13-point stencil, error 1.65e-4, reward beat the FD baseline. t\*=20 run got stuck oscillating between a good compact cone and chaotic/failed-solve episodes; more episodes alone may not fix this (see u_xx / L1 note below). |
| 3. Refinement (β/η) | `9. Refinment Formulation.ipynb` | **Active work.** Dense/telescoping reward via the adjoint β and local error indicator η, so the agent doesn't need terminal-only credit assignment. This is the one to keep building. |

Why 3 exists: 1 and 2 both had "terminal-only reward" and "no values while deciding" as
diagnosed failure modes (see thesis §5.9 for the walker's `‖w‖₁` amplification story). 3 fixes
both by giving the agent `û`, `β`, `η` at every step and a reward that telescopes into
`(1/3)log₁₀(e_0/e_final) − λ·points`.

## Formulation 3: exact cell-by-cell state of the notebook

10 cells currently in `9. Refinment Formulation.ipynb`. Built and verified one at a time
(run it, compare printed output, only then move on — keep doing this for new cells).

0. markdown — `## IMPORTS`
1. code — imports (numpy, matplotlib, gymnasium, sb3_contrib MaskablePPO, scipy sparse, cKDTree)
2. code — PDE params (`alpha=0.1, c=1.0`), grid (`nx=100, T=0.5`), exact solution `u_true` (sine series)
3. code — FD reference: `W_FD`, `fd_reference(ix, it) -> (u_fd, E_FD, C_FD)`
4. code — `solve_weights(dxs, dts, w_max=W_MAX)`: 3 PDE-substituted Taylor rows, pinv, admissibility
   = rank 3 + `cond <= COND_MAX (1e4)` + `‖w‖₁ <= W_MAX`. **`W_MAX = 2`** (user's own choice,
   not the 1.5 I originally proposed). **u_xx row already fixed** — line 10 reads
   `(0.5 * (dxs - c * dts)**2 + alpha * dts) / h**2`, the corrected row from the advisor's PR
   (was `0.5 * dxs**2 + alpha * dts`, dropped the `-c*dt` cross terms). Verified: makes the
   3-point stencil exactly Lax-Wendroff instead of FTCS; effect on this problem is real but
   modest (Cell-7 first-solve error 7.2e-5 -> 4.76e-5 with old W_MAX=1.5 test values).
5. code — `Cloud` class: KD-tree (`scipy.spatial.cKDTree`) over grid indices `(ix, it)`.
   `x/dx = ix` and `t/dt = it` exactly, so nearest-neighbour in index space IS the normalised
   distance the design doc asks for. `.add(ix, it)`, `.neighbours(ix, it, k)` (any direction,
   including later times, rebuilds tree lazily via `_dirty` flag — rebuild-per-add is
   `O(N log N)`, a known future perf issue, not fixed yet).
6. code — `U0` (IC on the grid), `is_known(ix, it)` (t=0 or wall), `known_value`,
   `build_stencil(cloud, ix, it, k=5)` — KNN stencil + `solve_weights`, `None` if inadmissible.
7. code — `evaluate(cloud, k=5)`: global sparse solve `(I-W)u=b` over every non-known cloud
   point, stencils rebuilt fresh every call (deliberate — adding a point can change an
   existing point's best neighbours). Returns `(u_hat dict, stencils dict)` or `(None, None)`.
8. code — `adjoint(stencils, z_star)`: `(I-W)^T beta = e_z*`, reuses the matrix `evaluate` built.
9. code — `K_T, R_X, H_TOP = 2, 2, 2`; `WINDOW` (24 relative offsets); `candidates(cloud, t_top)`
   (geometric: empty cells within WINDOW of any existing cloud point, below `t* + H_TOP`);
   `admissible_candidates(cloud, t_top, k=5)` (candidates that would themselves get an
   admissible stencil — filters ~5-10% typically, mostly cells right next to z* that would
   grab z* itself as a neighbour and blow up `‖w‖₁`).

### Missing — send next
**`eta_point`** — the η field (local error indicator, disagreement between a k=5 and a k=7
stencil evaluated on already-computed values). Was designed and verified numerically in an
earlier session but **never actually sent/added to this notebook** — repeatedly conflated
with "the u_xx fix" in conversation, which IS done (Cell 4) but is a different thing. Send
this first in the next session. Formula:

```python
def eta_point(cloud, p, u_hat, k_lo=5, k_hi=7):
    st_hi = build_stencil(cloud, *p, k=k_hi)
    if st_hi is None:
        return None
    nbs_hi, w_hi = st_hi
    value = lambda q: known_value(*q) if is_known(*q) else u_hat[q]
    u_hi = sum(wq * value(q) for q, wq in zip(nbs_hi, w_hi))
    return abs(u_hi - u_hat[p])
```
Numbers need re-verifying against the *current* Cell 4 (post u_xx-fix, `W_MAX=2`) — last
verified numbers were against an earlier `W_MAX=1.5` test version, not what's in the notebook.

## The MDP tuple (agreed design, not yet implemented as a gym.Env)

| | |
|---|---|
| **State** | occupancy/`û`/`β`/`η` in a small fixed window around an **active point** (NOT z* — a box on z* would need to scale with t*, defeating the whole "local, shared, size-independent policy" premise) + active point's offset to z* + budget used |
| **Action** | pick one of the ~24 `WINDOW` cells relative to the active point |
| **Reward** | `r_k = (1/3)[log10(e_k) - log10(e_{k+1})] - lambda`, e_k = true error `|u_hat(z*) - u_true(z*)|` (u_true IS allowed in the reward during training, never in the observation) |
| **Terminal** | budget exhausted, or a stop action |
| **Policy** | one shared plain MLP (explicitly NOT a CNN — decided against it; box-on-z* + CNN was the design doc's "v1" suggestion, rejected as unnecessary complexity for a window this small, and box-on-z* itself was flawed for the scaling reason above). `MaskablePPO`, default `MlpPolicy`, same as 6.3/8.3. |

Observation vector, precisely: for each of the N window cells j (fixed order):
`occ_j, u_j (0 if not computed), beta_j (0 if not computed), eta_j (0 if not computed)`,
concatenated, plus 3 globals `[dx_to_zstar, dt_to_zstar, points_used/budget]`. Shape `R^(4N+3)`.

**Open design question, not resolved:** after adding a point, what becomes the next active
point? Proposed default (matches 8.3): a queue, always pop the point with the latest `t`
first (heapq, like 8.3's `pending` list). Alternative floated but not decided: pick by
highest β (most influential unresolved point next). **Decide this before writing `step()`.**

**Also flagged, not resolved:** occupancy uses `0` for "not computed" and real values
otherwise — network has to infer "unknown" from `occ_j=0` rather than a distinct sentinel.
Left as a known simplification.

## The advisor's PR (`atayyar1/RL-For-PDEs` PR #1, branch `jo-exploration`)

Joseph Bakarji (advisor) ran a Claude Code session (~2 active days, 31 commits, all
`Co-Authored-By: Claude`) and opened a PR with (a) a real bug fix to 15 files, (b) a large
`exploration/` research tree (135 files) that is mostly his own separate research program
("a book"), explicitly said in his own notes to be distinct from this thesis.

Read-only copy checked out at:
`C:\Users\user\OneDrive - American University of Beirut\Desktop\RL-For-PDEs-jo`
(a git worktree of `origin/jo-exploration` — if git commands are needed there, run
`git worktree repair "<that path>"` from the main repo first, since it moved OneDrive folders).

**Verified and applied:**
- u_xx row fix — see Cell 4 above. Applied in formulation 3 only. **Still not applied to 8.3**
  (row appears in 3 places there: `solve_weights`, `taylor_rank`, `valid_sets` — all three
  must change together or the admissibility mask and the actual weights disagree).

**Verified and rejected:**
- Positivity certificate + max-entropy weights (instead of min-norm `pinv`). Tested on our
  own stencils: no accuracy gain (median error ratio ~1.0, sometimes much worse), and the
  positivity mask would reject ~4% of currently-admissible candidates. Not adopting.

**Verified, important, unresolved strategically:**
- **F6**: one wide stencil built once from the IC, weights from matching moments of the exact
  propagator (closed-form only because alpha, c are constant), beats every local-5-neighbour
  method on this exact PDE by ~9 orders of magnitude (67 points, error 1.6e-9, vs backward
  map's 13 points at 1.65e-4, vs FD's 4949 points at 1.9e-4). Needs linear + parabolic +
  CONSTANT coefficients — doesn't apply to Burgers or variable alpha(x).
  **Decision made 2026-09-29: get formulation 3 working on the current constant-coefficient
  PDE FIRST. Treat F6 and greedy-DWR (argmax beta*eta) as evaluation baselines to report
  against once training produces results — not as a reason to change the test problem yet.**
  Variable-coefficient / Burgers extension is explicitly future work, not now.

**Read, not directly actionable:**
- Two related papers Joseph sent: SpeND (arXiv 2609.02833, learns stencil *weights* via a
  projected NN — complementary to your RL choosing *geometry*, doesn't drop into Cell 4 as-is,
  slack is only 2 free dims for your 5-neighbour stencils vs their 16) and NeMDO
  (arXiv 2603.24641, its predecessor, weaker — no exact consistency, needs a filter for
  time-marching stability). Cited an email draft to one of SpeND's authors (Jack King) re:
  PhD supervision, sent by user separately.
- Advisor's `REEVALUATION.md`: your thesis (RL for point placement, known PDE) explicitly
  called "a legitimate student project", ranked separately from his book program. Ladder:
  L1 (learn weights) = classical/solved, L2 (learn geometry, known PDE) = "mostly classical,
  optimum usually a formula" = **this is where formulation 3 currently sits**, L3 (learn
  geometry AND the operator) = "the actual question, barely touched".

## Working style (carry over)

- Cell-by-cell: one small cell per message, run and verify it myself before sending, plain
  explanation tied to the method, picture when it helps. Never a hard-coded fallback — if a
  mask/constraint exists, derive the choice from it.
- Keep discussion replies short, one topic at a time.
- Thesis PDF (`RL_for_PDEs__Ali_Tayyar.pdf`, in the project folder) has prior notation/results
  — check it before asserting what "the thesis says".

## Immediate next step

Send `eta_point` as the next notebook cell (re-verify numbers against current Cell 4 first),
then resolve the active-point-advance question, then build `reset`/`step`/`action_masks`.
