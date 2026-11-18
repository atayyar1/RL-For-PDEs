"""Fig. 2 (schematic): one decision of the backward-map MDP.
Schematic only: the map is built with a fixed toy stencil, not by the agent.
Outputs fig_mechanism.{pdf,svg,png} next to this script."""
import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"mathtext.fontset": "cm", "font.family": "serif",
                     "font.serif": ["CMU Serif", "DejaVu Serif"], "font.size": 11})
OUT = os.path.dirname(os.path.abspath(__file__))

# colours
C_IC, C_WALL, C_GRID = "#d9a441", "#a99bd6", "#d0d0d0"
C_DONE, C_PEND, C_P, C_LINK = "#6a5acd", "#2ca02c", "#f2b84b", "#d9a441"
C_AGENT, C_BOX, C_LOOP = "#ddd5f0", "#fbf8ec", "#b5473a"

# ---- toy backward map: grid nx x nt, z* at top, fixed 5-point stencil ----
NX, NT = 15, 10
zs = (8, NT - 1)
STENCIL = [(-2, -1), (-1, -1), (0, -1), (1, -1), (2, -1)]  # toy shape
known = lambda c, l: l == 0 or c == 0 or c == NX - 1
expanded, pending = [], {zs}
K = 14                                   # decisions already taken
while len(expanded) < K:
    p = max(pending, key=lambda q: (q[1], -abs(q[0] - zs[0])))
    pending.remove(p); expanded.append(p)
    for dc, dl in STENCIL:
        q = (min(max(p[0] + dc, 0), NX - 1), p[1] + dl)
        if not known(*q) and q not in expanded:
            pending.add(q)
p = max(pending, key=lambda q: (q[1], -abs(q[0] - zs[0])))   # current point
pending.remove(p)
nbrs = [(p[0] + dc, p[1] + dl) for dc, dl in [(-3, -1), (-2, -1), (-1, -1), (0, -2), (1, -1)]]

fig = plt.figure(figsize=(10.5, 3.8))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 105); ax.set_ylim(-4.5, 34); ax.axis("off")

# ---- centre panel: space-time grid ----
X0, Y0, DX, DY = 34, 6, 2.0, 2.6
P = lambda c, l: (X0 + c * DX, Y0 + l * DY)
for c in range(NX):
    for l in range(NT):
        x, y = P(c, l)
        if l == 0:   ax.plot(x, y, "o", ms=4, color=C_IC, zorder=2)
        elif c in (0, NX - 1): ax.plot(x, y, "o", ms=3.5, color=C_WALL, zorder=2)
        else:        ax.plot(x, y, ".", ms=2, color=C_GRID, zorder=1)
ax.annotate("", xy=(X0 + NX * DX, Y0 - 1.3), xytext=(X0 - 1.5, Y0 - 1.3),
            arrowprops=dict(arrowstyle="->", color="0.3", lw=1))
ax.annotate("", xy=(X0 - 1.5, Y0 + NT * DY), xytext=(X0 - 1.5, Y0 - 1.3),
            arrowprops=dict(arrowstyle="->", color="0.3", lw=1))
ax.text(X0 + NX * DX + 0.6, Y0 - 1.3, "$x$", va="center")
ax.text(X0 - 1.5, Y0 + NT * DY + 0.8, "$t$", ha="center")

for c in range(zs[0] % 3, NX, 3):              # pass-1 coarse lattice (every 3rd col / level)
    for l in range(zs[1] % 3, NT, 3):
        if 0 < c < NX - 1 and 0 < l:
            ax.plot(*P(c, l), "x", ms=6, mew=1.0, color="0.55", zorder=2.5)
for q in expanded[1:]:
    ax.plot(*P(*q), "o", ms=5, color=C_DONE, zorder=3)
for q in pending:
    filled = q in nbrs
    ax.plot(*P(*q), "o", ms=5, color=C_PEND, zorder=3)
for q in nbrs:                                   # chosen stencil of p
    ax.plot([P(*p)[0], P(*q)[0]], [P(*p)[1], P(*q)[1]], color=C_LINK, lw=1.3, zorder=2)
    reused = q in pending or q in expanded or known(*q)
    ax.plot(*P(*q), "o", ms=6, mfc=C_PEND if reused else "white", mec=C_PEND, mew=1.5, zorder=4)
ax.plot(*P(*p), "o", ms=9, color=C_P, mec="#c98a1b", zorder=5)
ax.text(P(*p)[0] + 0.9, P(*p)[1] + 0.9, "$p$", fontsize=12)
ax.plot(*P(*zs), "*", ms=15, color=C_P, mec="#c98a1b", zorder=5)
ax.text(P(*zs)[0] + 1.0, P(*zs)[1] + 0.2, "$z^*$", fontsize=12)

# ---- left: agent ----
ax.add_patch(FancyBboxPatch((2, 13), 22, 12, boxstyle="round,pad=0.4,rounding_size=1",
                            fc=C_AGENT, ec="#9b8fc9"))
ax.text(13, 19, "RL agent\n$\\pi_\\theta(a\\mid s)$\nstencil policy",
        ha="center", va="center", linespacing=1.5)
ax.annotate("", xy=(X0 - 2.6, 19), xytext=(24.8, 19), arrowprops=dict(arrowstyle="->", lw=1.1))
ax.text(28.6, 20.0, "$a$", ha="center", fontsize=12)
ax.text(28.6, 15.4, "stencil\nor STOP", ha="center", fontsize=8.5, linespacing=1.1)

# ---- right: weights, influence, reward ----
BX = 72
ax.add_patch(FancyBboxPatch((BX, 6), 31, 24, boxstyle="round,pad=0.4,rounding_size=1",
                            fc=C_BOX, ec="#d8cfa8"))
ax.text(BX + 15.5, 27.2, "Taylor weights on $\\mathcal{S}(p)$", ha="center")
ax.text(BX + 15.5, 23.0, "$\\mathbf{w}=\\mathbf{A}^{+}\\mathbf{b},\\quad \\|\\mathbf{w}\\|_1\\leq 2$",
        ha="center")
ax.text(BX + 15.5, 18.4, "$\\beta_q \\leftarrow \\beta_q+\\beta_p w_q$", ha="center")
ax.text(BX + 15.5, 13.6, "$r=-\\,|\\beta_p|\\,s_p/\\mathrm{EPS}-\\lambda\\,N_{\\rm new}/N_{\\rm ref}$",
        ha="center", fontsize=10)
ax.text(BX + 15.5, 9.0, "new points join the map", ha="center", fontsize=9.5, style="italic")
ax.annotate("", xy=(BX - 0.6, 19), xytext=(X0 + NX * DX + 2.2, 19),
            arrowprops=dict(arrowstyle="->", lw=1.1))

# ---- loop back ----
ax.plot([BX + 15.5, BX + 15.5, 13], [5.4, 1.2, 1.2], color=C_LOOP, lw=1.6, ls=(0, (4, 2)))
ax.annotate("", xy=(13, 12.4), xytext=(13, 1.2),
            arrowprops=dict(arrowstyle="->", color=C_LOOP, lw=1.6, ls=(0, (4, 2))))
ax.text(BX + 6, 2.0, "next state $s_{k+1}$", color=C_LOOP, fontsize=10.5, ha="center")

# ---- legend ----
ly = -2.6
items = [("o", C_DONE, C_DONE, "expanded"), ("o", C_PEND, C_PEND, "pending / reused"),
         ("o", "white", C_PEND, "new"), ("o", C_IC, C_IC, "initial data"),
         ("o", C_WALL, C_WALL, "wall data"), ("x", "0.55", "0.55", "pass 1 (coarse)")]
for i, (m, fc, ec, lab) in enumerate(items):
    lx = 8 + i * 15.5
    ax.plot(lx, ly, m, ms=6, mfc=fc, mec=ec, mew=1.4)
    ax.text(lx + 1.2, ly, lab, va="center", fontsize=9)

for ext in ("pdf", "svg", "png"):
    fig.savefig(os.path.join(OUT, f"fig_mechanism.{ext}"), dpi=200, bbox_inches="tight")
print("ok", p, nbrs)
