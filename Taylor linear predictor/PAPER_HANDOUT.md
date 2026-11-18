# Handout for a new session: start the paper (Overleaf)

Project folder: `C:\Users\amt35\OneDrive - American University of Beirut\RL For PDEs\Taylor linear predictor`
Author: Ali Tayyar (PhD, AUB). Advisor: Joseph Bakarji. Target venue: JCP or SISC.

## 1. How to work with Ali
- Keep answers short and to the point. Give independent recommendations, not a menu of options.
- **Only measured numbers.** If a number was not computed, say "not computed". Never round, interpolate or predict a result. Check the notebooks or run files before quoting anything.
- Discuss decisions before writing; one change at a time.
- Math is written in LaTeX.
- The reward never uses the exact solution: the method is a-priori. The exact solution is used only for reporting errors.
- The results are **not final yet**: KS training and the comparisons notebook are still to come. Write the method, the setup and the structure now. Leave the result tables as placeholders (`\todo{}`), unless a number is listed below as measured.

## 2. Existing material to read first
- `paper/`: an earlier paper draft (`preamble.tex`, `body.tex`, `sec4_mdp.tex`, `tab_hyper.tex`, `fig_framework.tex`, and `paper_fig_*.py` figure scripts). It predates the standalone two-pass method and the rule-defined menu (sections 3–4 below). Reuse what still holds, and update the rest.
- `MAIN.tex`: Ali's thesis progress document (a living record, report class, ~1,300 lines).
- `HANDOFF_4_burgers.md` … `HANDOFF_7_advection_stops_meeting.md`, `NOTEBOOK_14_LATER.md`: history of the decisions.
- `Advisor update Nov 2026.pptx`: the latest advisor presentation.
- Current notebooks:
  - `15. Backward Burgers Standalone.ipynb`
  - `15.2 Backward Advection Standalone.ipynb`
  - `15.3 Backward KS Standalone.ipynb` (in progress)

## 3. The method (what the paper is about)
**Goal:** compute u(z*) at one query point z* = (x*, t*) of a time-dependent PDE, using as few grid points as possible, without solving the whole field.

- **Backward map.** Starting at z*, each point p gets a stencil: neighbours on lower time levels, with Taylor (PDE-consistent) weights. Points are expanded latest-t first, down to the known data (the IC and the wall columns). A forward pass then evaluates the values bottom-up, with Picard re-solves for nonlinear PDEs.
- **Influence β (discrete adjoint).** β_{z*} = 1, β_q += β_p w_q. Exact identity:
  $$e(z^*) = \sum_p \beta_p \delta_p,$$
  where δ_p is the local error of p's stencil. This gives the bound Σ|β_p δ_p|.
- **Taylor weights.** Min-norm weights matching the frozen-operator exponential series. Rows = PDE order + 1:
  - Burgers and advection-diffusion: 3 rows;
  - KS: 5 rows, 6 neighbours.
  - A stencil is **admissible** if it has full rank, cond ≤ 1e4 and ‖w‖₁ ≤ 2.
- **A-priori score** s_p: the Taylor polynomial residual of the stencil, evaluated with **guessed** derivatives (never the exact solution).
- **Standalone two-pass method** (advisor's request):
  - **Pass 1:** a coarse uniform backward lattice rooted at z* (every KX-th column, every KT-th level, fixed rule). A local polynomial fit gives guessed u and its derivatives. Its points count in the cost. It replaced an earlier background Lax–Wendroff guess, so the method needs nothing external.
  - **Pass 2:** the RL agent builds the fine map, using pass 1's guesses for the speed and the score.
  - Framing: this is one step of Bakarji's iterative-refinement idea (coarse placement, then the agent adds points, repeated). That idea is the roadmap or future work.
- **RL.** MaskablePPO.
  - One action = one stencil shape from a menu, or STOP (reuse existing points).
  - Reward per decision: $-|\beta_p| s_p/\mathrm{EPS} - \lambda\,(\text{new points})/N_{\rm ref}$. λ is sampled per episode and given in the state.
  - Training is multi-query; `best.zip` is chosen on a selection query.
- **The menu (decided Nov 17, 2026; replaces "the shapes greedy used").** Every 6-point shape within ±4 columns and 1–3 levels below p is kept if it is:
  1. admissible for every u in the range, and
  2. stable: its amplification per time step, used everywhere, stays ≤ 1.02 (KS itself grows at most e^{dt/4} ≈ 1.013).

  Added to that: wide rows (every 3rd column, shift −3..3, depth 1–3) and two wall shapes. Nothing is chosen by hand or by greedy. A hand-designed physical family (shift × spacing × depth, about 23 usable shapes for KS) is planned as an **ablation**. Burgers and advection were trained with greedy-derived menus; whether to retrain them with the rule menu is still open.
- **Paper claims:**
  1. A standalone, a-priori, goal-oriented method for a point query.
  2. **Physical evidence** in the learned policy: the stencils follow characteristics (shift), the spacing reflects stiffness (KS: the u_xxxx term forces wide spacing), and β approximates the adjoint / Green's function. Interpretability tests need a null control.
  3. Results hold across 3 PDEs.
  - GFDM is parked (not in this paper).

## 4. The three PDEs and their status
1. **Burgers** (smooth, inflow; notebook 15).
   - Pass 1: 3×3 lattice, wide-5 stencil, quartic fit.
   - Trained seeds 1, 3, 5 on (25, 37); runs in `runs11-12/nb15_menu_z25-37_eps1e-03_seed{s}_2dt_pass1`; greedy menu of 521 shapes.
   - Measured: pass 1 has about the same cost as the old LW guess. An exact-derivative guess improves the bound only 6–11%.
   - The final evaluation is not done yet.
2. **Advection-diffusion** (notebook 15.2).
   - PDE: u_t + a(x)u_x = α u_xx with a = C(x − x0), C = −3, x0 = 0.5, α = 0.01, T = 0.5, nx = 50, nt = 56.
   - Pass 1: 3×3, with wall points at intermediate levels.
   - Training queries: (25,30), (15,25), (35,25), (25,20); seeds 1, 3, 5, 10M steps each.
   - Baselines built: FD-3, LW, and DWR (Becker–Rannacher) with time slabs, explicit and Crank–Nicolson.
   - DWR-CN at (25, 30): about 2,434 points for error 7.2e-4; about 21k points for 2.0e-4.
   - The agents' final tables have not been collected yet.
3. **Kuramoto–Sivashinsky** (notebook 15.3, in progress).
   - PDE: u_t + u u_x + u_xx + u_xxxx = 0, L = 22, T = 10, rescaled to x in [0, 1]; nx = 50, dt = 0.05, walls 2 columns thick (data from the reference).
   - Reference solution: Fourier pseudo-spectral ETDRK4 (Kassam–Trefethen), N = 256.
   - Pass 1 (decided): 3×6 lattice, with a 3-row fallback where 5 rows are not admissible. At (25, 100) it uses 231 points (6% of the 4,071-point cone) and guides the score as well as 2×2 (1,098 points).
   - Finding for the paper: at this dt, compact stencils are admissible but **unstable** (up to 1.6× growth per step). The stability filter in the menu exists because of this.
   - Reference map: every 2nd column, one level down. It is stable but inaccurate (error 0.28 at (25, 100) with 4,235 points).
   - Honest concern: on smooth periodic KS, spectral ETDRK4 (about 1,224 points to (25, 100), highly accurate) may beat the method at equal cost. Report it honestly as a limitation if it does.
   - Still to do: the menu and window cell, greedy, the env with the rule menu, training, and evaluation.

## 5. Comparisons (planned: a separate comparisons notebook, results go straight into the paper)
- **Baselines** (follow McGreivy & Hakim's fair-baseline rules: equal cost, a strong classical method):
  - FD-3, LW, and DWR (explicit and CN) for every PDE;
  - ETDRK4 for KS;
  - possibly semi-Lagrangian and high-order FD/WENO.
  - Greedy is an internal gauge, not a comparator.
- **Metrics:** error at z* against points used (pass 1 + pass 2), over held-out queries.
  - Report **interpolation** (inside the training t* range) and **extrapolation** separately.
- **Ablations:**
  - the rule menu against the hand-designed physical family;
  - pass-1 lattice density;
  - the guess quality (the exact-derivative guess as a diagnostic only).

## 6. First tasks for the new session
1. Read the files in section 2, and propose a paper outline for a JCP-style article: intro and related work, method, MDP, PDEs and setup, results, physical-evidence section, limitations, future work. Agree on it with Ali before writing.
2. Set up the Overleaf structure: a main file plus one file per section, figures and tables folders. Reuse `paper/preamble.tex` where it fits. Ali uploads to Overleaf himself unless he says otherwise.
3. Draft the method section first (sections 3 and 4 above are stable). Leave results as placeholders.
