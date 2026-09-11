"""Every verified finding, encoded as an assertion. Run: python tests/test_core.py

Each test name cites the finding it locks down (see ../FINDINGS.md).
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from core.pde import Problem
from core import stencil as st


def test_F8_corrected_row_gives_lax_wendroff():
    """Pure advection + corrected row => exactly Lax-Wendroff (the old row gave FTCS-central)."""
    p = Problem(alpha=1e-30, c=1.0)
    dx, dt, nu = 0.01, 0.004, 0.4
    A, b = st.rows_taylor(np.array([-dx, 0, dx]), np.array([-dt] * 3), p)
    w = np.linalg.solve(A, b)
    lw = np.array([nu * (1 + nu) / 2, 1 - nu**2, -nu * (1 - nu) / 2])
    assert np.allclose(w, lw, atol=1e-12), f"{w} != {lw}"


def test_F_ftcs_is_positive_with_unit_norm():
    """The classical CFL condition IS the positivity condition."""
    p = Problem(nx=100)
    A, b = st.rows_taylor(np.array([-p.dx, 0, p.dx]), np.array([-p.dt] * 3), p)
    w = np.linalg.solve(A, b)
    assert w.min() >= -1e-14, f"FTCS at r={p.r:.3f} should be positive, got {w}"
    assert abs(st.amplification(w) - 1.0) < 1e-12


def test_F_positivity_implies_unit_l1():
    """C2: positive-feasible => ||w||_1 = 1 exactly."""
    p, rng = Problem(nx=100), np.random.default_rng(0)
    n = 0
    for _ in range(400):
        ix, it = rng.integers(-4, 5, 5), -rng.integers(1, 6, 5)
        A, b = st.rows_taylor(ix * p.dx, it * p.dt, p)
        w = st.solve_positive(A, b)
        if w is not None:
            n += 1
            assert abs(st.amplification(w) - 1.0) < 1e-9
    assert n > 100, f"too few feasible stencils to be a real test ({n})"


def test_F_fast_certificate_matches_lp():
    """T1's atan2 convex-hull test is EXACTLY equivalent to the LP, and faster."""
    p, rng = Problem(nx=100), np.random.default_rng(1)
    cases = []
    for _ in range(1500):
        ix, it = rng.integers(-4, 5, 5), -rng.integers(1, 6, 5)
        cases.append(st.rows_taylor(ix * p.dx, it * p.dt, p))
    t0 = time.perf_counter(); lp = [st.solve_positive(A, b) is not None for A, b in cases]
    t_lp = time.perf_counter() - t0
    t0 = time.perf_counter(); fast = [st.positive_feasible(A, b) for A, b in cases]
    t_fast = time.perf_counter() - t0
    dis = sum(a != b_ for a, b_ in zip(lp, fast))
    assert dis == 0, f"{dis} disagreements out of {len(cases)}"
    assert t_fast < t_lp, f"fast test not faster: {t_fast:.4f}s vs {t_lp:.4f}s"
    print(f"      certificate: {t_lp/t_fast:.0f}x faster than LP, 0/{len(cases)} disagreements")


def test_F6_moment_hierarchy_matches_closed_form():
    """Propagator moments from PDE coefficients alone == Gaussian closed form."""
    from scipy.integrate import solve_ivp
    p, P = Problem(), 8
    tau = 400 * p.dt

    def rhs(t, M):
        d = np.zeros_like(M)
        for q in range(1, P + 1):
            d[q] = p.c * q * M[q - 1] + (p.alpha * q * (q - 1) * M[q - 2] if q >= 2 else 0.0)
        return d
    M0 = np.zeros(P + 1); M0[0] = 1.0
    ode = solve_ivp(rhs, (0, tau), M0, rtol=1e-12, atol=1e-14).y[:, -1]
    closed = st.propagator_moments(tau, P, p) * (-1.0) ** np.arange(P + 1)
    rel = np.abs(ode - closed) / np.maximum(np.abs(closed), 1e-300)
    assert rel.max() < 1e-8, f"max rel diff {rel.max():.2e}"


def test_F6_taylor_rows_are_the_p2_moment_case():
    """rows_taylor's constraints are implied by matching moments 0..2.

    m must be wide enough to REACH the required second moment: sum w dx^2 =
    (c tau)^2 + 2 alpha tau needs (m dx)^2 > that. At m=6, tau=40dt it is
    infeasible by 2% -- the stencil physically cannot spread that far.
    """
    p = Problem()
    tau, m = 40 * p.dt, 14
    dxi = np.arange(-m, m + 1) * p.dx
    dti = np.full(len(dxi), -tau)
    A_t, b_t = st.rows_taylor(dxi, dti, p)
    w = st.solve_positive(*st.rows_moment(dxi, dti, 2, p))
    assert w is not None
    assert np.abs(A_t @ w - b_t).max() < 1e-10, "moment p=2 solution violates Taylor rows"


def test_F3_jensen_obstruction():
    """No non-negative w can cancel the dt^2 moment: sum w dt^2 >= (sum w dt)^2."""
    p, rng = Problem(nx=100), np.random.default_rng(2)
    n = 0
    for _ in range(600):
        npts = rng.integers(5, 12)
        dxi = rng.integers(-5, 6, npts) * p.dx
        dti = -rng.integers(1, 8, npts) * p.dt
        A, b = st.rows_taylor(dxi, dti, p)
        w = st.solve_positive(A, b)
        if w is None:
            continue
        n += 1
        assert w @ dti**2 >= (w @ dti) ** 2 - 1e-18
    assert n > 100


def test_F9_corrected_row_fixes_the_rank_degeneracy():
    """BONUS: the F8 fix also removes the vertical-stencil failure mode.

    With the old row (1/2 dx^2 + alpha dt) every entry is AFFINE in dt, so a
    stencil with all neighbours at one x gives rank 2, b is out of range, and
    lstsq silently returns sum(w) = 0.48. The corrected row carries xi^2, which
    is QUADRATIC in dt, so rank 3 is restored and the failure mode disappears.
    """
    p = Problem()
    dxi, dti = np.full(5, 2 * p.dx), -np.arange(1, 6) * p.dt
    h = max(np.max(np.abs(dxi)), np.sqrt(p.alpha * np.max(np.abs(dti))), 1e-12)
    A_old = np.array([np.ones(5), (dxi - p.c * dti) / h,
                      (0.5 * dxi**2 + p.alpha * dti) / h**2])
    assert np.linalg.matrix_rank(A_old) == 2, "old row should be rank-deficient here"
    A_new, b = st.rows_taylor(dxi, dti, p)
    assert np.linalg.matrix_rank(A_new) == 3, "corrected row should restore full rank"
    # and the guard still catches a genuinely inconsistent system
    assert st.solve_minnorm(A_old, b) is None, "guard failed on the rank-2 system"
    assert st.solve_minnorm(A_old, b, guard=False) is not None


def test_F1_diffusive_frontier_formula():
    """k_max(m) ~ (m dx)^2/(2 alpha dt): EXACT for the old row, a tight upper bound
    (within ~15%) for the corrected row, which the advective terms slightly tighten.
    Only valid on the diffusive branch; T1 reports further bounds at large m."""
    p = Problem()
    for m in [2, 3, 4, 6, 8, 12, 16]:
        pred = (m * p.dx) ** 2 / (2 * p.alpha * p.dt)
        ok = lambda k: st.positive_feasible(*st.rows_taylor(
            np.arange(-m, m + 1) * p.dx, np.full(2 * m + 1, -k * p.dt), p))
        kmax = 0
        for k in range(1, int(pred * 1.3) + 3):
            if ok(k):
                kmax = k
            elif kmax:
                break
        assert 0.82 * pred <= kmax <= pred + 1, f"m={m}: k_max={kmax} vs formula {pred:.1f}"


if __name__ == "__main__":
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_")]
    fails = 0
    for name, fn in fns:
        try:
            fn(); print(f"  PASS  {name}")
        except AssertionError as e:
            fails += 1; print(f"  FAIL  {name}\n        {e}")
        except Exception as e:
            fails += 1; print(f"  ERROR {name}\n        {type(e).__name__}: {e}")
    print(f"\n{len(fns)-fails}/{len(fns)} passed")
    sys.exit(1 if fails else 0)
