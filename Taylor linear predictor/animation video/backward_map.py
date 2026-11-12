# Manim teaser of the backward stencil map, replayed from a real agent episode (map_multi_seed1.json).
#   python -m manim -qh backward_map.py Clip1Stencil Clip2Map Clip3Values
from manim import *
import json, os
import numpy as np

config.background_color = BLACK
HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "map_multi_seed1.json")))
NX, Z = D["nx"], tuple(D["z_star"])
TS = Z[1]
STEPS = D["steps"]

GOLD_, ORANGE_, DIM, CYAN_ = "#F2A900", "#E07A1F", "#3A3A3A", "#38BDF8"
TITLE_FONT, BODY_FONT = "Georgia", "Georgia"
H, X0, Y0 = 0.165, -6.6, -3.25                                       # cell size and the grid's lower-left corner


def P(q):
    return np.array([X0 + q[0] * H, Y0 + q[1] * H, 0.0])


def is_known(q):
    return q[1] == 0 or q[0] in (0, NX - 1)


def title(text):
    t = Text(text, font=TITLE_FONT, weight=BOLD, font_size=30, color=WHITE)
    return t.scale_to_fit_width(min(t.width, 4.8)).move_to([PX, 3.35, 0])     # top of the right panel, clear of the grid


PX = 4.3                                                              # centre of the right panel


def caption(text, color=GREY_A, mask=False):
    """Body text; mask=True puts a black backing behind it so grid dots do not show through."""
    t = Text(text, font=BODY_FONT, font_size=26, color=color)
    return t.add_background_rectangle(color=BLACK, opacity=1, buff=0.15) if mask else t


def grid():
    """All grid points up to t*: known data (IC, walls) orange, the rest dim."""
    dots = {}
    for it in range(TS + 1):
        for ix in range(NX):
            q = (ix, it)
            dots[q] = Dot(P(q), radius=0.022, color=ORANGE_ if is_known(q) else DIM)
    return dots


def axes():
    x_ax = Arrow(P((0, 0)) + [-0.1, -0.3, 0], P((NX - 1, 0)) + [0.45, -0.3, 0], buff=0, stroke_width=2,
                 max_tip_length_to_length_ratio=0.03, color=GREY_B)
    t_ax = Arrow(P((0, 0)) + [-0.3, -0.1, 0], P((0, TS)) + [-0.3, 0.45, 0], buff=0, stroke_width=2,
                 max_tip_length_to_length_ratio=0.05, color=GREY_B)
    return VGroup(x_ax, t_ax, MathTex("x", color=GREY_B).next_to(x_ax, RIGHT, buff=0.1),
                  MathTex("t", color=GREY_B).next_to(t_ax, UP, buff=0.1))


def star(q, scale=1.0):
    return Star(n=5, outer_radius=0.13 * scale, inner_radius=0.055 * scale, color=GOLD_, fill_opacity=1).move_to(P(q))


def beta_color(b):
    """|beta| on a log scale, 1e-4 and below (dim violet) to 1 (gold)."""
    s = np.clip((np.log10(max(abs(b), 1e-12)) + 4) / 4, 0, 1)
    return interpolate_color(interpolate_color(ManimColor("#3B1C78"), ManimColor("#C2187F"), min(1, 2 * s)),
                             ManimColor(GOLD_), max(0, 2 * s - 1))


# ── Clip 1: the question and one stencil ──
class Clip1Stencil(Scene):
    def construct(self):
        dots = grid()
        field = VGroup(*dots.values())
        ax = axes()
        self.play(FadeIn(field, lag_ratio=0.002, run_time=2), FadeIn(ax))
        ic = caption("initial condition", ORANGE_).scale(0.8).next_to(P((NX // 2, 0)), DOWN, buff=0.45)
        self.play(FadeIn(ic))

        zs = star(Z)
        q_text = caption("we only want u here", GOLD_).next_to(zs, UP, buff=0.25)
        self.play(GrowFromCenter(zs), Write(q_text))
        self.wait(0.5)

        # the classical explicit cone of z*: every point a scheme like Lax-Wendroff must compute
        cone = [[dots[(ix, TS - j)] for ix in range(max(1, Z[0] - j), min(NX - 2, Z[0] + j) + 1)] for j in range(TS)]
        n = ValueTracker(0)
        counter = always_redraw(lambda: Text(f"explicit scheme: {int(n.get_value())} points", font=BODY_FONT,
                                             font_size=26, color=CYAN_).move_to([4.4, 2.2, 0]))
        self.add(counter)
        self.play(LaggedStart(*[AnimationGroup(*[d.animate.set_color(CYAN_) for d in lvl]) for lvl in cone], lag_ratio=0.3),
                  n.animate.set_value(D["n_lw"]), run_time=3.5, rate_func=linear)
        note = caption("a classical scheme computes\nthe whole cone", GREY_A).scale(0.85).next_to(counter, DOWN, buff=0.3)
        self.play(FadeIn(note))
        self.wait(1.5)
        self.play(*[d.animate.set_color(DIM) for lvl in cone[1:] for d in lvl], FadeOut(counter), FadeOut(note),
                  FadeOut(q_text), FadeOut(ic), FadeOut(ax), run_time=1)

        # zoom on z*: scale the field about z* and move z* to the upper left
        s0 = STEPS[0]
        target = np.array([-3.3, 2.4, 0])
        Pz = lambda q: target + 4 * (P(q) - P(Z))                         # position after the zoom
        keep = lambda q: -7.4 < Pz(q)[0] < 0.2 and Pz(q)[1] > -4.2       # left half only: the right half is the equation panel
        world = VGroup(*[d for q, d in dots.items() if keep(q)], zs)
        self.play(world.animate.scale(4, about_point=P(Z)).shift(target - P(Z)),
                  FadeOut(VGroup(*[d for q, d in dots.items() if not keep(q)])), run_time=2)

        # the window: 21 candidate cells, 1-3 levels below
        window = VGroup(*[Circle(radius=0.16, color=WHITE, stroke_width=2).move_to(Pz((Z[0] + i, Z[1] - j)))
                          for j in (1, 2, 3) for i in range(-3, 4)])
        wtext = caption("21 candidate neighbours, 1–3 levels below", mask=True).next_to(window, DOWN, buff=0.35)
        self.play(LaggedStart(*[Create(c) for c in window], lag_ratio=0.04), FadeIn(wtext), run_time=1.5)
        self.wait(0.5)

        # the agent's stencil: 5 cells, edges to z*
        nbs = [tuple(q) for q in s0["nbs"]]
        picks = VGroup(*[Dot(Pz(q), radius=0.11, color=GOLD_) for q in nbs])
        edges = VGroup(*[Line(Pz(q), target, color=GOLD_, stroke_width=3, stroke_opacity=0.8) for q in nbs])
        ptext = caption("the agent picks one stencil: 5 cells", GOLD_, mask=True).move_to(wtext)
        self.play(FadeTransform(wtext, ptext), LaggedStart(*[GrowFromCenter(p) for p in picks], lag_ratio=0.15),
                  LaggedStart(*[Create(e) for e in edges], lag_ratio=0.15), window.animate.set_stroke(opacity=0.35), run_time=1.5)

        # equations, right panel
        eq1 = MathTex(r"\hat u_{z^*}", r"=", r"\sum_{q} w_q\,\hat u_q", font_size=46)
        rows_label = caption("weights: Taylor expansion + the PDE").scale(0.85)
        rows = MathTex(r"\sum_q w_q &= 1\\ \sum_q w_q\,\xi_q &= 0\\ \sum_q w_q\left(\tfrac12\xi_q^2+\alpha\,\Delta t_q\right) &= 0",
                       font_size=36)
        xi = MathTex(r"\xi_q=\Delta x_q-u\,\Delta t_q", font_size=32, color=GREY_A)
        score = MathTex(r"s_{z^*}=\Big|\sum_q w_q\,P(\Delta x_q,\Delta t_q)\Big|", font_size=36)
        sval = MathTex(rf"\approx {s0['score']:.1e}".replace("e-0", r"\times10^{-").replace("e-", r"\times10^{-") + "}",
                       font_size=34, color=GOLD_)
        VGroup(eq1, rows_label, rows, xi, score, sval).arrange(DOWN, buff=0.3).move_to([4.1, 0, 0])
        score.shift(0.25 * DOWN); sval.shift(0.25 * DOWN)
        self.play(Write(eq1), run_time=1.5)
        self.play(FadeIn(rows_label), Write(rows), run_time=2.5)
        self.play(FadeIn(xi))

        # the solved weights on the neighbours
        wlabels = VGroup(*[DecimalNumber(w, num_decimal_places=2, font_size=24, color=WHITE, include_sign=True)
                           .next_to(Pz(q), DOWN, buff=0.12) for q, w in zip(nbs, s0["w"])])
        self.play(LaggedStart(*[FadeIn(l, shift=0.1 * UP) for l in wlabels], lag_ratio=0.1), run_time=1)
        self.wait(0.5)

        # its estimated local error
        self.play(Write(score), run_time=1.5)
        self.play(FadeIn(sval))
        self.wait(1)

        # the neighbours now need their own stencils
        pending = [q for q in nbs if not is_known(q)]
        rings = VGroup(*[Circle(radius=0.2, color=GOLD_, stroke_width=3).move_to(Pz(q)) for q in pending])
        ntext = caption("each neighbour now needs its own stencil", GOLD_, mask=True).move_to(ptext)
        self.play(FadeTransform(ptext, ntext), LaggedStart(*[Create(r) for r in rings], lag_ratio=0.1), FadeOut(wlabels))
        self.play(*[Indicate(r, color=WHITE, scale_factor=1.3) for r in rings], run_time=1)
        self.wait(1)
        self.play(*[FadeOut(m) for m in self.mobjects])


def links():
    """Every stencil link in expansion order: (p, q, kind), kind = 'new' (q enters the map), 'shared' (q already in it)
    or 'known' (IC / wall)."""
    seen, out = {Z}, []
    for s in STEPS:
        p = tuple(s["p"])
        for q in map(tuple, s["nbs"]):
            kind = "known" if is_known(q) else "shared" if q in seen else "new"
            seen.add(q)
            out.append((p, q, kind))
    return out


LINKS = links()
N_SHARED = sum(k == "shared" for _, _, k in LINKS)
SHARE = N_SHARED / len(LINKS)
BETA = {tuple(s["p"]): s["beta"] for s in STEPS}
UHAT = {tuple(s["p"]): s["u"] for s in STEPS}
LINK_COLOR = {"new": GOLD_, "shared": CYAN_, "known": ORANGE_}


def influence_vs_characteristic():
    """Per level: the beta-weighted centre of the map, and the backward characteristic dx/dt = u from z*
    (traced with the exact u at the map points: for display only)."""
    rows = {}
    for s in STEPS:
        rows.setdefault(s["p"][1], []).append((s["p"][0], abs(s["beta"]), s["u_true"]))
    centre, char, x = [], [(Z[0], TS)], float(Z[0])
    for it in range(TS, 0, -1):
        r = sorted(rows[it])
        xs, bs, us = map(np.array, zip(*r))
        centre.append(((xs * bs).sum() / bs.sum(), it))
        x -= np.interp(x, xs, us) * D["dt"] / D["dx"]
        char.append((x, it - 1))
    return centre, char


CENTRE, CHAR = influence_vs_characteristic()


def base_scene(scene):
    dots = grid()
    scene.add(VGroup(*dots.values()), axes())
    zs = star(Z)
    scene.add(zs)
    return dots, zs


# ── Clip 2: the map grows backward, then the influence flows down ──
class Clip2Map(Scene):
    def construct(self):
        dots, zs = base_scene(self)
        head = title("Building the map backward from z*")
        self.play(FadeIn(head))

        n_pts, n_sh, n_ln = ValueTracker(1), ValueTracker(0), ValueTracker(0)
        c_pts = always_redraw(lambda: Text(f"points in the map: {int(n_pts.get_value())}", font=BODY_FONT, font_size=28,
                                           color=GOLD_).move_to([PX, 2.0, 0]))
        c_sh = always_redraw(lambda: Text(f"links to a reused point: {100 * n_sh.get_value() / max(n_ln.get_value(), 1):.0f}%",
                                          font=BODY_FONT, font_size=26, color=CYAN_).move_to([PX, 1.35, 0]))
        c_cone = Text(f"explicit cone: {D['n_lw']} points", font=BODY_FONT, font_size=24, color=GREY_B).move_to([PX, 0.75, 0])
        legend = VGroup(*[VGroup(Line(ORIGIN, 0.5 * RIGHT, color=c, stroke_width=4), caption(t).scale(0.8)).arrange(RIGHT, buff=0.2)
                          for c, t in [(GOLD_, "new point"), (CYAN_, "reused point"), (ORANGE_, "known data")]]
                         ).arrange(DOWN, aligned_edge=LEFT, buff=0.2).move_to([PX, -0.8, 0])
        self.play(FadeIn(c_pts), FadeIn(c_sh), FadeIn(c_cone), FadeIn(legend))

        lines = VGroup()
        self.add(lines)
        by_step = {}
        for p, q, k in LINKS:
            by_step.setdefault(p, []).append((q, k))
        order = [tuple(s["p"]) for s in STEPS]

        def batch(ps, run_time, width, slow=False):
            anims, new_lines, cnt = [], [], {"new": 0, "shared": 0, "all": 0}
            for p in ps:
                for q, k in by_step[p]:
                    ln = Line(P(q), P(p), color=LINK_COLOR[k], stroke_width=width, stroke_opacity=0.9 if slow else 0.45)
                    new_lines.append(ln)
                    cnt["all"] += 1
                    if k == "new":
                        cnt["new"] += 1
                        anims.append(dots[q].animate.set_color(GOLD_).scale(1.6))
                    elif k == "shared":
                        cnt["shared"] += 1
                        if slow:
                            anims.append(Flash(P(q), color=CYAN_, line_length=0.12, flash_radius=0.1, num_lines=8))
            if slow:
                ring = Circle(radius=0.09, color=WHITE, stroke_width=2).move_to(P(ps[0]))
                anims.append(Succession(Create(ring), FadeOut(ring)))
            lines.add(*new_lines)
            self.play(*[Create(l) for l in new_lines], *anims,
                      n_pts.animate.increment_value(cnt["new"]), n_sh.animate.increment_value(cnt["shared"]),
                      n_ln.animate.increment_value(cnt["all"]), run_time=run_time)

        for p in order[:8]:                                           # slowly: one stencil at a time
            batch([p], 0.8, 2.5, slow=True)
        for i in range(8, 80, 6):                                     # faster
            batch(order[i:i + 6], 0.3, 1.2)
        for i in range(80, len(order), 45):                           # fast-forward
            batch(order[i:i + 45], 0.15, 0.8)
        self.wait(0.5)
        msg = caption(f"{100 * SHARE:.0f}% of the links reuse a point\nalready in the map: the agent\nshares neighbours",
                      CYAN_).move_to([PX, -2.7, 0])
        self.play(FadeIn(msg))
        self.wait(2)

        # influence: beta flows down, level by level
        head2 = title("Influence of every point on z*")
        beq = MathTex(r"\beta_{z^*}=1,\qquad \beta_q \leftarrow \beta_q+\beta_p\,w_q", font_size=34).move_to([PX, 2.2, 0])
        map_pts = [p for p in order if p != Z]
        self.play(lines.animate.set_stroke(opacity=0.06), FadeOut(msg), FadeOut(legend), FadeOut(c_sh), FadeOut(c_cone),
                  FadeOut(c_pts), FadeTransform(head, head2), *[dots[p].animate.set_color(GREY_D) for p in map_pts], run_time=1)
        self.play(Write(beq), run_time=1.5)
        bar = Rectangle(width=0.3, height=2.3, stroke_width=0, fill_opacity=1).move_to([PX - 0.6, 0.4, 0])
        bar.set_fill(color=[beta_color(10 ** e) for e in np.linspace(-4, 0, 9)])
        bar.set_sheen_direction(UP)
        bar_lbl = VGroup(MathTex(r"|\beta|=1", font_size=30).next_to(bar, RIGHT, buff=0.2).align_to(bar, UP),
                         MathTex(r"\le 10^{-4}", font_size=30).next_to(bar, RIGHT, buff=0.2).align_to(bar, DOWN))
        self.play(FadeIn(bar), FadeIn(bar_lbl))
        for it in range(TS, 0, -1):
            lvl = [p for p in map_pts if p[1] == it]
            if lvl:
                self.play(*[dots[p].animate.set_color(beta_color(BETA[p])) for p in lvl], run_time=0.13)
        self.wait(0.5)

        # the influence follows the characteristic
        char = DashedVMobject(VMobject(color=WHITE, stroke_width=3).set_points_smoothly([P(c) for c in CHAR]), num_dashes=40)
        cdots = VGroup(*[Dot(P(c), radius=0.045, color=WHITE) for c in CENTRE[::3]])
        ceq = MathTex(r"\frac{dx}{dt}=u", font_size=40).move_to([PX, -1.9, 0])
        note = caption("the centre of the influence (dots)\nfollows the characteristic (dashes)", WHITE).scale(0.9).move_to([PX, -2.9, 0])
        self.play(LaggedStart(*[GrowFromCenter(d) for d in cdots], lag_ratio=0.08), run_time=1.5)
        self.play(Create(char), FadeIn(note), Write(ceq), run_time=2)
        self.wait(3)
        self.play(*[FadeOut(m) for m in self.mobjects])


# ── Clip 3: values flow up, the answer at z* ──
class Clip3Values(Scene):
    def construct(self):
        dots, zs = base_scene(self)
        for p in UHAT:
            if p != Z:
                dots[p].set_color(beta_color(BETA[p])).scale(1.6)
        head = title("Forward pass: values flow up")
        eq = MathTex(r"\hat u_p=\sum_q w_q\,\hat u_q", font_size=42).move_to([PX, 2.3, 0])
        self.play(FadeIn(head), Write(eq), run_time=1.5)

        u_lo, u_hi = min(UHAT.values()), max(UHAT.values())
        u_color = lambda u: interpolate_color(ManimColor("#2E6FD8"), ManimColor("#F5F5F5"), (u - u_lo) / (u_hi - u_lo))
        for it in range(1, TS + 1):
            lvl = [p for p in UHAT if p[1] == it and p != Z]
            if lvl:
                self.play(*[dots[p].animate.set_color(u_color(UHAT[p])) for p in lvl if p != Z], run_time=0.12)
        self.play(Indicate(zs, color=WHITE, scale_factor=1.8), run_time=1)

        # the answer at z* against the exact solution at t*
        xs, ue = D["exact_at_tstar"]
        ax = Axes(x_range=[0, 1, 0.5], y_range=[round(min(ue) - 0.1, 1), round(max(ue) + 0.1, 1), 0.4],
                  x_length=4.4, y_length=2.6, tips=False, axis_config={"color": GREY_B, "stroke_width": 2}).move_to([PX, -0.3, 0])
        lbl = VGroup(MathTex("x", font_size=28, color=GREY_B).next_to(ax.x_axis, RIGHT, buff=0.1),
                     MathTex(r"u(x,t^*)", font_size=28, color=GREY_B).next_to(ax.y_axis, UP, buff=0.1))
        curve = ax.plot_line_graph(xs, ue, add_vertex_dots=False, line_color=WHITE, stroke_width=3)
        s0 = STEPS[0]
        x_star = Z[0] * D["dx"]
        ours = Dot(ax.c2p(x_star, s0["u"]), radius=0.08, color=GOLD_)
        self.play(Create(ax), FadeIn(lbl), Create(curve), run_time=1.5)
        self.play(TransformFromCopy(zs, ours), run_time=1.2)
        err = abs(s0["u"] - s0["u_true"])
        e_txt = MathTex(rf"|e(z^*)| = {err:.1e}".replace("e-0", r"\times10^{-").replace("e-", r"\times10^{-") + "}",
                        font_size=34, color=GOLD_).next_to(ax, DOWN, buff=0.45)
        ident = MathTex(r"e(z^*)=\sum_p \beta_p\,\delta_p", font_size=36).next_to(e_txt, DOWN, buff=0.3)
        self.play(FadeIn(e_txt))
        self.play(Write(ident), run_time=1.2)
        self.wait(2)

        # payoff
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1)
        l1 = Text(f"{len(STEPS)} points, at twice the explicit time step", font=BODY_FONT, font_size=34, color=WHITE)
        l2 = Text("where Lax–Wendroff and FD-3 blow up", font=BODY_FONT, font_size=28, color=GREY_B)
        l3 = Text("≈10× lower error than explicit schemes at the same cost", font=BODY_FONT, font_size=34, color=GOLD_)
        l4 = Text("(9 unseen queries)", font=BODY_FONT, font_size=24, color=GREY_B)
        l5 = Text("Learned stencils, built backward from the question", font=TITLE_FONT, weight=BOLD, font_size=36, color=WHITE)
        VGroup(l1, l2, l3, l4, l5).arrange(DOWN, buff=0.3).shift(0.2 * UP)
        l3.shift(0.25 * DOWN); l4.shift(0.25 * DOWN); l5.shift(0.7 * DOWN)
        self.play(FadeIn(l1, shift=0.2 * UP), FadeIn(l2, shift=0.2 * UP), run_time=1.2)
        self.wait(0.8)
        self.play(FadeIn(l3, shift=0.2 * UP), FadeIn(l4, shift=0.2 * UP), run_time=1.2)
        self.wait(0.8)
        self.play(Write(l5), run_time=1.5)
        self.wait(2.5)
