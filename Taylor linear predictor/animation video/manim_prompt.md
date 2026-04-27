# Claude Code Prompt: Manim Animation for RL-based PDE Solving

## Task
Create a Manim (Community Edition) animation called `PDESolvingComparison` that runs for approximately 60–80 seconds and illustrates three approaches to solving a 1D PDE (advection-diffusion) in the space-time domain. The animation is for a 10-minute student presentation.

## Technical setup
- Use `manim` community edition (v0.18+)
- Output: 1920x1080, 60fps
- Scene class name: `PDESolvingComparison`
- Use `manim -pqh pde_animation.py PDESolvingComparison` to render

## Domain convention (CRITICAL)
- The domain is SPACE-TIME: x is horizontal (space), t is vertical (time, going upward)
- Label the horizontal axis as `x` and the vertical axis as `t`
- Include tick marks and axis labels on every scene where a grid/domain is shown
- The bottom of the domain is t=0 (initial condition), the top is t=T (later time)
- Points should appear from bottom to top as time progresses — this is physically meaningful

---

## Scene 1: Title card (3 seconds)
- Black background
- White serif text: "How Should a PDE Solver Decide Where to Look?"
- Subtitle: "Three Approaches"
- Fade in, hold 2s, fade out

---

## Scene 2: Traditional Fixed Grid (15 seconds)

**Visual:**
- Draw a rectangular domain (width ~6 units, height ~8 units), centered on screen
- Draw x-axis label at the bottom: `x`, with tick marks at regular intervals
- Draw t-axis label on the left: `t`, with tick marks at regular intervals
- Animate an 8×10 uniform grid of orange lines being drawn — first all vertical lines (left to right), then all horizontal lines (bottom to top)
- As each grid intersection appears, place a small yellow dot at that node
- All nodes appear simultaneously after the grid is drawn

**Text overlay (top left):**
"Traditional Fixed Grid"

**After grid is complete, show annotation:**
- Arrow pointing to the grid with text: "Every point evaluated — even where nothing interesting happens"
- Then show: "100 points total" in white text bottom right

**Timing:** Grid draws in ~8s, annotation appears at ~10s, hold 3s

---

## Scene 3: Adaptive Mesh Refinement (15 seconds)

**Transition:** Transform the uniform grid into AMR grid

**Visual:**
- Keep the same domain with axis labels
- Show the coarse uniform grid in orange (same as before)
- Animate a region of refinement in the CENTER of the domain (not corner) — a 2×2 block of cells gets subdivided into 4×4 finer cells, shown in yellow
- The refined region should be clearly inside the domain, not at the boundary

**Text overlay (top left):**
"Adaptive Mesh Refinement (AMR)"

**Annotations:**
- Arrow to refined region: "Refinement triggered by error estimate"
- Below: "Still requires a mesh structure"
- Point counter: "~60 points (saves 40%)"

**Key point:** AMR still operates on a mesh — the solver decides WHERE to refine but the structure is still a grid. Show this by keeping the coarse grid visible in orange while the refinement is yellow.

**Timing:** Refinement animates in ~6s, annotations ~4s, hold 3s

---

## Scene 4: RL-based Adaptive Sampling (30 seconds)

**Layout:** Split screen
- LEFT half: The full fixed grid from Scene 2 (orange, faded to 40% opacity) — label it "FD Reference" in small text
- RIGHT half: Empty domain with just the boundary box (white outline), axis labels x and t, IC dots along the bottom (yellow dots at t=0, evenly spaced — these are the known initial condition)

**Animation — points appear in RIGHT panel progressively from bottom to top:**

The RL agent places points one at a time. Animate ~25 points total over ~20 seconds. Each point placement should:
1. Show a blue arrow from the current "frontier" of the cloud pointing to the new target location
2. Place an orange/red dot at the target location
3. The arrow fades out after placement

**Point placement logic (HARDCODED positions — do NOT place randomly):**
- First 8 points: roughly evenly spaced in x, at t ~ 0.15 (one timestep above IC) — this represents the first "sweep"
- Next 8 points: at t ~ 0.35, but with DENSER clustering in the center-left region (x ~ 0.2 to 0.5) where a wave/gradient would be — fewer points on the right where solution is flat
- Last 9 points: at t ~ 0.55 to 0.7, again clustered where the gradient is (continuing the wave pattern leftward)

**Color the points by "local error":**
- Points in high-gradient regions: red-orange
- Points in smooth regions: yellow-green
- Use a smooth color gradient between these

**Running counter (bottom of RIGHT panel):**
Show live updating text: "Points placed: N / 25"

**Comparison counter (bottom of screen, full width):**
"FD: 100 pts | RL agent: N pts | Savings so far: X%"
Update this as each RL point is placed.

**Text overlay (top of RIGHT panel):**
"RL-based Adaptive Sampling"

**After all 25 points are placed:**
- Show annotation: "Agent learned: skip smooth regions, focus on gradients"
- Flash the final comparison: "FD: 100 pts ↔ RL: 25 pts — same accuracy"
- Hold 4 seconds

---

## Scene 5: Summary (5 seconds)
- Fade to black
- Show three-row comparison table in white text:
  ```
  Method              | Points | Adapts? | Mesh-free?
  Fixed Grid (FD)     |  100   |   No    |    No
  AMR                 |   60   |   Yes   |    No
  RL Sampling (ours)  |   25   |   Yes   |   Yes ✓
  ```
- Highlight the RL row in gold/yellow
- Hold 4 seconds, fade out

---

## Style guide
- Background: pure black (#000000)
- Grid lines: coral/orange (#FF6B6B)
- Grid nodes (FD): yellow (#FFD700)
- RL agent points: gradient from red-orange (#FF4500) to yellow-green (#ADFF2F)
- Axis lines and labels: white (#FFFFFF)
- Annotations: cyan (#00BFFF) for arrows, white for text
- Font: use Manim's default (LaTeX for math labels, sans-serif for titles)
- All axis labels should use LaTeX: `$x$`, `$t$`

## Known issues to avoid
- DO NOT place RL points only at the bottom of the domain — they must appear at progressive time levels (bottom to top)
- DO NOT forget axis labels on every domain panel
- The AMR refinement should be in the interior of the grid, not at the corner
- The blue "action arrows" should point FROM existing points TO the new target — not from outside the domain

## Output
Single file: `pde_animation.py`
Run with: `manim -pqh pde_animation.py PDESolvingComparison`
