# Phase 2 — The price of compression

Opened 2026-09-11 after the re-evaluation. Phase 1 mapped the classical boundary of local
linear stencils and found ten classical results in one session. Phase 2 changes the question.

## The spine

T5 measured that coarse-graining space faster than the dynamics mixes destroys *Markov*
positivity, and that memory buys it back — exactly, at small M, as a **pure-delay** law.

Read that as an interface cost between scales:

> **If agents at scale k compose into an agent at scale k+1, the composite needs memory.
> The memory depth is the price of the interface.**

Which turns the framework's central claim into a measurable question:

> **Does compression across scales have a bounded price?**
>
> Let p\*(M) be the minimal memory depth for a closed, stable, compact coarse law at
> coarsening factor M.
> - p\* bounded or slow-growing ⇒ hierarchies are cheap ⇒ multiscale agents are viable.
> - p\* blowing up ⇒ every scale transition costs more than the last ⇒ hierarchy is expensive
>   and deep multiscale structure should be rare.

That is the book's question — *do we still need differential equations* — in a form that can be
measured rather than argued. A differential equation is the p\* = 0 case: the present screens off
the past. Everything else is what you pay when it doesn't.

## The filter (from REEVALUATION.md — apply before launching anything)

Four assumptions define the mined-out space: **linear** scheme, **local** operator, **known**
operator, **field-oriented**. A thread is worth running only if it:

1. names the assumption it drops, and
2. does not reintroduce that assumption in its method.

Phase 1's failures all violated (2): moment-regression dropped *known operator* then used local
polynomial fitting (→ MLS); target-aware allocation dropped *field-oriented* then used the exact
adjoint (→ DWR).

**Every Phase 2 thread reports literature positioning BEFORE running experiments.**

## Threads

| | question | drops | prior-art risk |
|---|---|---|---|
| **M1** | the memory-depth spectrum p\*(M), with a positivity constraint | locality (in time) | medium — Mori–Zwanzig, optimal prediction |
| **M2** | learn the coarse variables that *minimise* memory | the projection is given | high — Markov state models, VAMP/tICA |

M3 (influence functions without an adjoint) is held back: it is the measurement apparatus for
M1/M2 and should not run until there is something to measure.

## What would make Phase 2 fail honestly

- If p\*(M) is just a restatement of the spectral gap, M1 is a reparameterisation.
- If minimising memory recovers the slow variables that VAMP/tICA already find, M2 adds only the
  positivity constraint — which may still be worth having, but is a much smaller claim.
- If both, the right move is to say so and stop, not to find a third framing.
