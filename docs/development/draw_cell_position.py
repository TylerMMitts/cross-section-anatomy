"""Draws cell_position.png for docs/development/cell_position.md (schematic, synthetic root). Run: python draw_cell_position.py"""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Wedge, Ellipse

INK, MUTED, STELE, CELL, TY, RO, BAD = "#222", "#8a877f", "#e9c46a", "#cfe0f2", "#b5462a", "#2a78d6", "#d62828"
R_STELE, R_ROOT, NL = 1.0, 4.0, 7

def cells(cx=0, cy=0, sx=1, sy=1):
    out = []
    for k in range(NL):
        r = R_STELE + (k + .5) * (R_ROOT - R_STELE) / NL
        size = 0.13 + 0.07 * np.sin(np.pi * (k + .5) / NL) * 1.6
        n = int(2 * np.pi * r / (2.1 * size))
        for a in np.linspace(0, 2 * np.pi, n, endpoint=False) + k * .2:
            out.append((cx + sx * r * np.cos(a), cy + sy * r * np.sin(a), size, k))
    return out

def root(ax, cs, cx=0, cy=0, sx=1, sy=1, stele=True):
    ax.add_patch(Ellipse((cx, cy), 2 * R_ROOT * sx + .3, 2 * R_ROOT * sy + .3, fc="#f4f1ea", ec=MUTED, lw=1))
    if stele:
        ax.add_patch(Ellipse((cx, cy), 2 * R_STELE * sx, 2 * R_STELE * sy, fc=STELE, ec="#b8952f", lw=1))
    for x, y, s, k in cs:
        ax.add_patch(Circle((x, y), s, fc=CELL, ec="#7da3c8", lw=.5))

fig, axs = plt.subplots(1, 3, figsize=(21, 8))
for ax in axs:
    ax.set_aspect("equal"); ax.axis("off")

# ---- A: Tyler
A = axs[0]
cs = cells()
root(A, cs)
for xy, w, h in (((-5, -7), 10, 7), ((-5, 0), 5, 6.2)):
    A.add_patch(Rectangle(xy, w, h, fc="white", ec="none", alpha=.72, zorder=3))
A.add_patch(Rectangle((0, 0), 4.6, 4.6, fc="none", ec=TY, lw=2, ls="--", zorder=6))
A.text(2.3, 4.75, "one image = one quarter (TR)", color=TY, ha="center", fontsize=10)
A.plot(0, 0, "o", color=TY, ms=11, zorder=5)
A.annotate("plant centre = image corner\n(from the filename, never checked)", (0, 0), (-4.8, -4.6),
           color=TY, fontsize=9.5, zorder=7, arrowprops=dict(arrowstyle="->", color=TY))
q = [c for c in cs if c[0] > 0 and c[1] > 0]
inn = min(q, key=lambda c: np.hypot(c[0], c[1])); out = max(q, key=lambda c: np.hypot(c[0], c[1]))
pick = [c for c in q if c[3] == 3][len([c for c in q if c[3] == 3]) // 2]
for c, lab, col in ((inn, "r_min → 0", TY), (out, "r_max → 1", TY), (pick, "r", INK)):
    A.plot([0, c[0]], [0, c[1]], color=col, lw=1.6, zorder=4)
    A.plot(c[0], c[1], "o", color=col, ms=5, zorder=5)
A.annotate("r_max → 1\n(outermost kept cell)", out[:2], (4.8, 3.6), fontsize=9.5, color=TY, zorder=7,
           arrowprops=dict(arrowstyle="->", color=TY))
A.annotate("r_min → 0\n(innermost kept cell)", inn[:2], (4.8, 0.9), fontsize=9.5, color=TY, zorder=7,
           arrowprops=dict(arrowstyle="->", color=TY))
A.annotate("stele: cells >25% inside\nare DROPPED (YOLO mask)", (.5, .5), (-4.8, 3.2), fontsize=9.5, color="#8a6d1d",
           zorder=7, arrowprops=dict(arrowstyle="->", color="#8a6d1d"))
A.text(-4.8, -2.3, "faded = outside this image", fontsize=9, color=MUTED, zorder=7)
A.text(-4.6, 5.6, "A  Tyler: distance from the corner, min–max within the quarter", fontsize=12, weight="bold")
A.text(-4.6, -6.6, "position = (r − r_min) / (r_max − r_min)\nr = distance from the corner; all angles pooled", fontsize=10.5, color=TY,
       family="monospace")
A.set_xlim(-5, 7.2); A.set_ylim(-7, 6.2)

# ---- B: Rooh
B = axs[1]
root(B, cs)
for ang in np.radians([30, 60]):
    B.plot([R_STELE * np.cos(ang), (R_ROOT + .15) * np.cos(ang)], [R_STELE * np.sin(ang), (R_ROOT + .15) * np.sin(ang)],
           color=RO, lw=1.2, ls=":")
ang = np.radians(45)
c = (2.6 * np.cos(ang), 2.6 * np.sin(ang))
B.plot([R_STELE * np.cos(ang), (R_ROOT + .15) * np.cos(ang)], [R_STELE * np.sin(ang), (R_ROOT + .15) * np.sin(ang)], color=RO, lw=2.2)
B.plot(*c, "o", color=INK, ms=6, zorder=5)
B.plot(R_STELE * np.cos(ang), R_STELE * np.sin(ang), "s", color=RO, ms=8, zorder=5)
B.plot((R_ROOT + .15) * np.cos(ang), (R_ROOT + .15) * np.sin(ang), "s", color=RO, ms=8, zorder=5)
B.text(.2, -.1, "stele\ncentre", fontsize=8.5, ha="center", color="#8a6d1d")
B.text(-.1, .9, "stele edge → 0", fontsize=9.5, color=RO, ha="right")
B.text(3.0, 3.25, "root edge → 1\n(at this angle)", fontsize=9.5, color=RO)
B.text(c[0] + .2, c[1] - .35, "cell", fontsize=9.5)
B.text(-4.6, 5.6, "B  Rooh: stele edge to root edge, at each cell's own angle", fontsize=12, weight="bold")
B.text(-4.6, -6.6, "position = (r − r_stele(θ)) / (r_root(θ) − r_stele(θ))\nr from the stele centre; whole section", fontsize=10.5, color=RO,
       family="monospace")
B.text(-4.6, -5.5, "stele cells kept (as stele); 0 and 1 are tissue edges,\nnot whichever cells survived", fontsize=9.5, color=MUTED)
B.set_xlim(-5, 6.5); B.set_ylim(-7, 6.2)

# ---- C: off-centre / oval root
C = axs[2]
cx, cy, sx, sy = .9, .6, 1.12, .85            # crop corner missed the true centre; root oval
cs2 = cells(cx, cy, sx, sy)
root(C, cs2, cx, cy, sx, sy)
C.add_patch(Rectangle((0, 0), 5.4, 4.6, fc="none", ec=TY, lw=2, ls="--"))
C.plot(0, 0, "o", color=TY, ms=11, zorder=5); C.plot(cx, cy, "+", color=INK, ms=14, mew=2, zorder=5)
C.text(cx + .3, cy + .1, "true centre", fontsize=9.5, weight="bold")
C.text(-.2, -.6, "image corner\n(assumed centre)", fontsize=9.5, color=TY, ha="right")
# same layer, two angles -> different Tyler r
L = [c for c in cs2 if c[3] == 2 and c[0] > 0 and c[1] > 0]
a, b = min(L, key=lambda c: np.hypot(*c[:2])), max(L, key=lambda c: np.hypot(*c[:2]))
for c in (a, b):
    C.plot([0, c[0]], [0, c[1]], color=BAD, lw=1.6); C.plot(c[0], c[1], "o", color=BAD, ms=7, mfc="none", mew=2, zorder=6)
C.annotate(f"3rd layer, distance {np.hypot(*b[:2]):.1f}", b[:2], (4.4, -1.6), color=BAD, fontsize=10,
           arrowprops=dict(arrowstyle="->", color=BAD))
C.annotate(f"3rd layer, distance {np.hypot(*a[:2]):.1f}", a[:2], (-1.0, 5.0), color=BAD, fontsize=10,
           arrowprops=dict(arrowstyle="->", color=BAD))
C.text(-4.6, 5.6, "C  Where they disagree: root off-centre or oval", fontsize=12, weight="bold")
C.text(-4.6, -5.6, "Tyler: one layer lands at different positions → the curve blurs.\n"
       "Rooh: the edges are measured at each cell's angle → the layer stays put.\n"
       "Centred, round root: the two agree, except Tyler's 0 is the\ninnermost surviving cell, not the stele edge.", fontsize=10, color=INK, linespacing=1.5)
C.set_xlim(-5, 7.5); C.set_ylim(-7, 6.2)

fig.tight_layout()
fig.savefig(__file__.replace("draw_cell_position.py", "cell_position.png"), dpi=170)
print("ok")
