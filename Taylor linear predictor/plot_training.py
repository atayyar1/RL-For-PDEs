"""Plot every logged quantity from one or more training runs.

    python plot_training.py runs9-9/seed8
    python plot_training.py runs9-9/seed8 runs9-1/t100_.../seed6 --labels tau2 nocon
    python plot_training.py runs9-9/seed8 --n-actions 24 --smooth 0.85

Reads the TensorBoard event files SB3 already writes, so nothing in the
training loop needs to change. Config (err_tol, w_err, N, ...) is parsed out of
the run directory name when it follows the t100_errtol0.001_werr1_... convention.
"""
import argparse
import math
import os
import re

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

CFG_RE = re.compile(
    r't(?P<t_star>\d+)_errtol(?P<err_tol>[\d.eE+-]+)_werr(?P<w_err>[\d.]+)'
    r'_wtime(?P<w_time>[\d.]+)_N(?P<N>\d+)_c(?P<c_shape>[\d.]+)_R(?P<R>\d+)')


def parse_cfg(path):
    """Pull hyperparameters out of the run directory name."""
    cfg = {'err_tol': 1e-3, 'w_err': 1.0, 'w_time': 0.1, 'N': 40, 't_star': None}
    for part in os.path.abspath(path).replace('\\', '/').split('/'):
        m = CFG_RE.search(part)
        if m:
            cfg.update({k: float(v) for k, v in m.groupdict().items()})
            cfg['N'] = int(cfg['N'])
    return cfg


def load(path):
    """Merge every tfevents file under `path` into {tag: (steps, values)}."""
    dirs = sorted({r for r, _, fs in os.walk(path)
                   for f in fs if f.startswith('events.out.tfevents')})
    if not dirs:
        raise SystemExit('no tfevents files under ' + path)

    merged = {}
    for d in dirs:
        ea = EventAccumulator(d, size_guidance={'scalars': 0})
        ea.Reload()
        for tag in ea.Tags()['scalars']:
            merged.setdefault(tag, []).extend(
                (e.step, e.value) for e in ea.Scalars(tag))

    out = {}
    for tag, pairs in merged.items():
        pairs.sort()
        out[tag] = (np.array([p[0] for p in pairs]),
                    np.array([p[1] for p in pairs]))
    return out


def ema(y, alpha):
    """Exponential smoothing; alpha <= 0 disables it."""
    if alpha <= 0 or len(y) < 2:
        return y
    s = np.empty(len(y), dtype=float)
    s[0] = y[0]
    for i in range(1, len(y)):
        s[i] = alpha * s[i - 1] + (1 - alpha) * y[i]
    return s


def derived(run, cfg, n_actions):
    """The quantities we keep recomputing by hand."""
    d = {}
    if 'rollout/mean_r_acc' in run:
        st, r_acc = run['rollout/mean_r_acc']
        d['err_norm'] = (st, -r_acc / cfg['w_err'])

        # Fraction of episodes scoring err_norm = 1 because the star solve
        # failed. Uses the arithmetic mean error, which overestimates the
        # successful episodes' err_norm, so this is a LOWER bound.
        if 'rollout/mean_error' in run:
            se, err = run['rollout/mean_error']
            err = np.interp(st, se, err)
            en_ok = np.minimum(
                np.log10(np.maximum(err, 1e-12) / cfg['err_tol']) / 3.0, 1.0)
            with np.errstate(divide='ignore', invalid='ignore'):
                f = (-r_acc / cfg['w_err'] - en_ok) / (1.0 - en_ok)
            d['failed_frac'] = (st, np.clip(np.nan_to_num(f), 0, 1))

    if 'train/entropy_loss' in run:
        st, ent = run['train/entropy_loss']
        d['eff_actions'] = (st, np.exp(-ent))          # SB3 logs -H
        d['max_actions'] = cfg['N'] * n_actions
    return d


# (title, [(tag, label, linestyle)], log_y, reference_lines)
PANELS = [
    ('Terminal error',
     [('rollout/mean_error', 'all episodes', '-'),
      ('rollout/mean_error_on_reach', 'reached only', '--')], True, 'err_tol'),
    ('Reached rate',
     [('rollout/reached_rate', 'reached', '-')], False, 'unit'),
    ('Episode reward',
     [('rollout/ep_rew_mean', 'ep_rew_mean', '-')], False, None),
    ('Reward components',
     [('rollout/mean_r_acc', 'r_acc (accuracy)', '-'),
      ('rollout/mean_r_eff', 'r_eff (time)', '-')], False, None),
    ('Episode length',
     [('rollout/ep_len_mean', 'all episodes', '-'),
      ('rollout/mean_kterm_on_reach', 'reached only', '--')], False, None),
    ('Value function',
     [('train/explained_variance', 'explained variance', '-')], False, 'unit'),
    ('Value loss',
     [('train/value_loss', 'value_loss', '-')], True, None),
    ('Update size',
     [('train/approx_kl', 'approx_kl', '-'),
      ('train/clip_fraction', 'clip_fraction', '-')], True, None),
    ('Policy gradient loss',
     [('train/policy_gradient_loss', 'pg_loss', '-')], False, None),
]

SUMMARY_ROWS = [
    ('rollout/mean_error_on_reach', min),
    ('rollout/reached_rate', max),
    ('rollout/ep_rew_mean', max),
    ('rollout/mean_r_acc', max),
    ('rollout/mean_r_eff', max),
    ('rollout/mean_kterm_on_reach', min),
    ('train/explained_variance', max),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+', help='run directories (searched recursively)')
    ap.add_argument('--labels', nargs='*', default=None)
    ap.add_argument('--n-actions', type=int, default=24,
                    help='len(ACTIONS); sets the max-entropy reference line')
    ap.add_argument('--smooth', type=float, default=0.8, help='EMA alpha, 0 = off')
    ap.add_argument('--out', default='training_summary.png')
    # overrides, for run dirs whose name does not carry the config
    ap.add_argument('--err-tol', type=float, default=None)
    ap.add_argument('--w-err', type=float, default=None)
    ap.add_argument('--n-walkers', type=int, default=None)
    ap.add_argument('--t-star', type=float, default=None)
    args = ap.parse_args()

    labels = args.labels or [os.path.basename(os.path.normpath(p)) for p in args.runs]
    data = []
    for lab, p in zip(labels, args.runs):
        cfg = parse_cfg(p)
        for key, val in [('err_tol', args.err_tol), ('w_err', args.w_err),
                         ('N', args.n_walkers), ('t_star', args.t_star)]:
            if val is not None:
                cfg[key] = val
        data.append((lab, load(p), cfg))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    ncol = 3
    nrow = math.ceil((len(PANELS) + 3) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 3.3 * nrow))
    axes = axes.ravel()

    for ax, (title, series, logy, refline) in zip(axes, PANELS):
        for ri, (lab, run, _) in enumerate(data):
            for si, (tag, name, style) in enumerate(series):
                if tag not in run:
                    continue
                st, v = run[tag]
                ax.plot(st, ema(v, args.smooth), style,
                        color=colors[(ri * len(series) + si) % 10], lw=1.4,
                        label=name if len(data) == 1 else lab + ': ' + name)
        if refline == 'err_tol':
            for _, _, cfg in data:
                ax.axhline(cfg['err_tol'], color='green', ls=':', lw=1,
                           label='err_tol = %.0e' % cfg['err_tol'])
        elif refline == 'unit':
            ax.axhline(1.0, color='gray', ls=':', lw=1)
            ax.set_ylim(-0.02, 1.05)
        if logy:
            ax.set_yscale('log')
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)

    k = len(PANELS)

    # err_norm -- what the reward actually sees
    ax = axes[k]
    for ri, (lab, run, cfg) in enumerate(data):
        d = derived(run, cfg, args.n_actions)
        if 'err_norm' in d:
            st, v = d['err_norm']
            ax.plot(st, ema(v, args.smooth), color=colors[ri % 10], lw=1.4, label=lab)
    ax.axhline(1.0, color='red', ls=':', lw=1, label='ceiling (failed solve)')
    ax.axhline(0.0, color='green', ls=':', lw=1, label='at err_tol')
    ax.set_title('err_norm  (what the reward sees)', fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    # fraction of episodes pinned at the ceiling
    ax = axes[k + 1]
    for ri, (lab, run, cfg) in enumerate(data):
        d = derived(run, cfg, args.n_actions)
        if 'failed_frac' in d:
            st, v = d['failed_frac']
            ax.plot(st, ema(v, args.smooth), color=colors[ri % 10], lw=1.4, label=lab)
    ax.set_title('failed-solve fraction (lower bound)\nthese episodes carry no gradient',
                 fontsize=9)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    # policy entropy expressed as an effective action count
    ax = axes[k + 2]
    for ri, (lab, run, cfg) in enumerate(data):
        d = derived(run, cfg, args.n_actions)
        if 'eff_actions' in d:
            st, v = d['eff_actions']
            ax.plot(st, ema(v, args.smooth), color=colors[ri % 10], lw=1.4, label=lab)
            ax.axhline(d['max_actions'], color=colors[ri % 10], ls=':', lw=1,
                       label='uniform = %d' % d['max_actions'])
    ax.set_title('effective action count  $e^{H}$\n(at uniform means nothing learned)',
                 fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    for ax in axes[k + 3:]:
        ax.axis('off')
    for ax in axes[:k + 3]:
        ax.set_xlabel('timesteps', fontsize=8)

    fig.suptitle(' | '.join(
        '%s  (t*=%s, err_tol=%.0e, N=%d)'
        % (lab, int(c['t_star']) if c['t_star'] else '?', c['err_tol'], c['N'])
        for lab, _, c in data), fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(args.out, dpi=130)
    print('wrote ' + args.out)

    # text summary: first 5% vs last 5% vs best
    for lab, run, cfg in data:
        print('\n=== %s ===' % lab)
        print('%-28s %11s %11s %11s' % ('quantity', 'start', 'end', 'best'))
        for tag, better in SUMMARY_ROWS:
            if tag not in run:
                continue
            _, v = run[tag]
            w = max(1, len(v) // 20)
            print('%-28s %11.4g %11.4g %11.4g'
                  % (tag.split('/')[-1], v[:w].mean(), v[-w:].mean(), better(v)))
        d = derived(run, cfg, args.n_actions)
        if 'eff_actions' in d:
            _, e = d['eff_actions']
            print('%-28s %11.0f %11.0f   (uniform = %d)'
                  % ('effective actions', e[0], e[-1], d['max_actions']))


if __name__ == '__main__':
    main()
