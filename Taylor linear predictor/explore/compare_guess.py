"""Every Burgers guess / picard configuration against ONE baseline: the notebook's LW map (LW guess, picard 1).
    python explore/compare_guess.py   (reads the results_inflow_front*.json written by evaluate.py)"""
import json, os
import numpy as np
Q_UNSEEN = [(ix, it) for it in (50, 75, 100) for ix in (12, 25, 38)]
Q_NEW = [(25, 130), (12, 130), (38, 130), (6, 90), (44, 60), (30, 115)]
gm = lambda v: float(np.exp(np.mean(np.log(v))))
D = os.path.dirname(os.path.abspath(__file__))
B = json.load(open(os.path.join(D, "results_inflow_front.json")))["R"]
FILES = {"LW guess, picard 1": "", "LW guess, picard 0": "_picard0", "FD-5 2x guess, picard 1": "_guess-fd52",
         "FD-5 2x guess, picard 0": "_guess-fd52_picard0", "hybrid 4x guess, picard 1": "_guess-hyb4",
         "LW guess, picard 3": "_picard3", "FD-5 2x guess, picard 3": "_guess-fd52_picard3"}
for qs, lab in [(Q_UNSEEN, "unseen 9"), (Q_NEW, "new 6")]:
    print(f"\n{lab}:\n| guess, forward pass | method | points | true error | exact bound | wins |\n|---|---|---|---|---|---|")
    for tag, suf in FILES.items():
        f = os.path.join(D, f"results_inflow_front{suf}.json")
        if not os.path.exists(f):
            continue
        R = json.load(open(f))["R"]
        for n in ["LW map", "greedy", "greedy+stop(myopic)", "LW map+stop(c=1)"]:
            rel = lambda k: gm([R[str(z)][n][k] / B[str(z)]["LW map"][k] for z in qs])
            w = sum(R[str(z)][n]["error"] < B[str(z)]["LW map"]["error"] for z in qs)
            print(f"| {tag} | {n} | x{rel('points'):.2f} | x{rel('error'):.2f} | x{rel('bound'):.2f} | {w}/{len(qs)} |")
