# Re-evaluation — 2026-09-11, end of session 1

## The data

Ten times today a thread went deep and landed on a classical result:

| what we found | who found it, when |
|---|---|
| positivity ⟺ CFL ⟺ discrete maximum principle | monotone-scheme theory, 1950s–60s |
| monotone linear ⇒ 1st order (hyperbolic) | Godunov 1959 |
| unconditionally positive ⇒ 1st order in time | Bolley–Crouzeix 1978 |
| KM expansion can't truncate above order 2 | Pawula 1967 |
| wide positive stencil, k ∝ m² | Runge–Kutta–Chebyshev / super-time-stepping, 1980s |
| jet recovery by moment regression | moving least squares, Lancaster–Salkauskas 1981 |
| positivity feasibility = realisable moment sequence | truncated moment problem, Hamburger 1920s |
| coarse-graining ⇒ memory | Mori–Zwanzig 1960s |
| propagator moments from the generator | Kramers–Moyal 1940s |
| **adjoint-weighted effort allocation (F16)** | **dual-weighted residual, Becker–Rannacher 1996** |

Ten for ten. That is not bad luck. It is a measurement of the space we chose to work in.

## The diagnosis

**We let the existing code define the question.** Ali's repo works on linear, local, constant-
coefficient PDEs on uniform grids with known operators. That space was thoroughly mined between
1950 and 1996. Anything we found there was going to be a rediscovery, and it was.

The programme's own ladder, written on day one, says where the goal actually lives:

| level | fixed | learned | status after today |
|---|---|---|---|
| L1 | PDE, grid | weights | classical, solved — **this is where we spent the session** |
| L2 | PDE | geometry | mostly classical (AMR, DWR); the optimum is usually a formula |
| L3 | — | geometry **and** the operator | **the actual question, barely touched** |

## The pattern that predicts which threads survive

Every result that survived today drops a classical assumption *and does not reintroduce it in
the method*. Every result that died dropped none, or dropped one and then used the classical
method for the classical problem anyway.

Four assumptions define the mined-out space: **linear** scheme, **local** operator, **known**
operator, **field-oriented** (solve everywhere).

| thread | assumption dropped | reintroduced? | outcome |
|---|---|---|---|
| wide positive stencils | none | — | = RKC1, exactly |
| RL over (m, k) | none | — | optimum is a boundary; nothing to learn |
| Jensen / trilemma | none | — | false, and the true version is Godunov |
| moment-regression discovery | known operator | **yes** — local polynomial fitting | = MLS, exactly |
| target-aware allocation | field-oriented | **yes** — exact adjoint | = DWR |
| positivity on scattered points | uniformity (no symbol ⇒ no von Neumann) | no | **survives** |
| memory under coarse-graining | locality | no | **survives, and is the least classical thing we have** |
| variable-coefficient rows | constant coefficients | no | survives, pending T2's checks |

This is a usable filter. **Before starting a thread, name the assumption it drops and check the
method does not smuggle it back.** Had I applied it this morning, three of five threads would not
have been launched.

## Reframing the question

The honest reading of today: for **local, Markov, known** dynamics, the differential equation is
not one option among many — it is the compressed form of the consistency conditions themselves.
Every alternative we built reduced to it or to a known scheme derived from it. So the book's
question has a sharp answer in that regime, and the answer is yes.

Which relocates the interesting question. Differential equations are **what you get when
forgetting is free** — when the dynamics mixes fast enough that the present screens off the past.
T5 measured exactly where that breaks: coarse-grain space faster than the dynamics mixes, and
positivity is unrecoverable *as a Markov law* but exactly recoverable *with memory* — and the
coarse law turns out to be a pure-delay scheme, B₁ = B₂ = 0. You do not correct a Markov law; you
move it backwards in time.

> **Differential equations are what you get when forgetting is free.
> Everything the framework cares about — coarse-grained physics, multiscale agents, intelligence —
> lives where it isn't.**

That is a claim about the world rather than about numerical schemes, it is supported by a
measured result rather than an analogy, and it points where his existing work already points:
Mori–Zwanzig, the arrow-of-time draft, the multiscale action–perception programme.

## Replan

**Stop.** No more work on linear local stencils for known constant-coefficient operators. The map
is complete and it is a map of other people's results. Keep the library and the tests; treat the
classical-results table above as the literature review it accidentally is.

**Decouple two projects that got fused today.** Ali's thesis — RL for point placement with a known
PDE — is a legitimate student project and the workshop paper should proceed on the corrected code
with the scope honestly stated. Joseph's book programme wants something different and should stop
borrowing Ali's problem setting.

**Three candidate directions, in order of my confidence:**

1. **Memory as the price of compression.** T5's result is the least classical thing produced today
   and the closest to the framework. Open questions that are genuinely open: does the pure-delay
   structure persist for M ≥ 4 and in 2D? Is there a variational characterisation of the minimal
   memory depth? What is the memory kernel of a *learned* coarse-graining, and can positivity be
   imposed on it? This is Mori–Zwanzig with a stability constraint, which is not a combination
   the literature has pushed on.

2. **Operator learning where no adjoint exists.** The one thing DWR cannot do is weight by an
   influence function you cannot compute. Learn the influence function from data, in the regime
   where the operator is unknown, and ask how much of the 18× survives. My crude-cone test hinted
   it mostly does, which is the learnability question. Distinct from the classical literature
   precisely by the missing adjoint.

3. **Ali's L2 question, properly scoped.** Positivity constrains geometry; the feasible boundary
   moves with local cell Péclet; one-sided stencils are categorically infeasible; and the
   certificate is necessary but admits the worst member of its feasible set. A learned selector
   tracking a moving feasible boundary is a real problem — but it is an engineering problem, not
   the book's.

**My recommendation: (1), and treat (2) as its measurement apparatus.** The book question becomes
*where does locality fail and what replaces it*, and memory is the answer we can already measure.

## What today was actually worth

- A real bug in the constraint row, fixed across 15 files and confirmed by four independent routes.
- A shared library with 12 tests, so claims can't rot silently.
- A complete map of the classical boundary of this space — done by experiment rather than by
  reading, expensively, but it is now done and doesn't need repeating.
- One genuinely open direction (memory) and one sharp negative (the certificate's boundary).
- A working method: parallel threads instructed to kill claims, which caught ten of mine.
