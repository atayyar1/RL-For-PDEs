# Standing checks

Three failure modes, each of which produced a confident wrong number in this programme, and each
of which was caught only by someone else. All three share a shape: **the output looks like a
result.** Run these before believing anything, including your own work.

---

### 1. Measure the noise floor before running a comparison
Check that the effect you expect to see exceeds the accuracy of your own apparatus — reference
solution, solver tolerance, fallback regions.

*Cost of skipping it*: three attempts at an "advective accuracy floor" that was a bug in the
reference; three attempts at a variable-coefficient experiment where the upgrade was worse than the
baseline; two failed reproductions of a colleague's result; and a colleague's own 78.1× withdrawn
because its reference was good only to 1.4e-7 against an arm at the same level. (F29, F37)

### 2. Never write the conclusion into the output text — read it off the table
If a script prints an interpretation, the interpretation will survive data that contradicts it.

*Cost of skipping it*: a thread's prose said a coarse law "stays positive with mass 1" while the
table directly above printed `no` in the positive column and mass 1.8192. It was caught only
because a second thread **ran the script before writing its criticism.** The coordinator then
committed the identical bug two hours after recording it — asserting "the stack is full rank N"
while the data underneath said 23/24. (F36, F33)

### 3. A degenerate objective makes every feasible point optimal
The solver then returns an arbitrary vertex, and **any structure you see in it is an artefact of
the pivot rule**, not of the constraint set. Cheap check: **re-solve with a perturbed or randomised
objective and see which features survive.** Credit: M1 for the general statement.

*Cost of skipping it*: one root cause behind three separate threads — a "pure-delay B₁ = B₂ = 0"
structure published as a striking finding that was one vertex among many (B₂ = 0.6917·I in another
solver's answer); `solve_positive` silently returning the *worst* member of its feasible set at
every step size, up to 10¹¹ times less accurate than the interior point; and a positivity
comparison where the spread across choices of feasible point (12–60×) was as large as the effect
being claimed. (F30, F37, F38)

---

### And one for choosing what to work on
Before starting a thread, **name the classical assumption it drops, and check the method does not
reintroduce it.** Phase 1 hit ten classical results in one session because it did not. Two threads
dropped an assumption and then used the classical method for the classical problem — one became
moving least squares to 5e-16, the other dual-weighted residual. (REEVALUATION.md)

**Corollary**: literature positioning goes *before* experiments, not after. It cost two afternoons
in Phase 2 and saved two threads — one of which turned out to have been published under its own
title thirteen years earlier.
