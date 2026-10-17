import time
import numpy as np

import strategie_minimax as sm
from moteur import State


def region_bounds(target=200):
    return [(0,                  min(151, target)),
            (min(151, target),   min(191, target)),
            (min(191, target),   target)]

def region_of(t):
    return 0 if t <= 150 else (1 if t <= 190 else 2)


def collect_round_decks(policy, n_games=400, target=200):
    sm.TARGET = target
    bounds = region_bounds(target)
    regions = [r for r, (a, b) in enumerate(bounds) if b > a]
    decks = {(r0, r1, s): [] for r0 in regions for r1 in regions for s in (0, 1)}
    for j in range(max(5000, n_games)):
        starts = []
        sm.game((policy, policy), starter=j % 2, verbose=False, round_starts=starts)
        for total, s, c in starts:
            r0, r1 = region_of(total[0]), region_of(total[1])
            decks[r0, r1, s].append(c)
            decks[r1, r0, 1 - s].append(c)
        if j + 1 >= n_games and min(map(len, decks.values())) >= 5:
            return decks, j + 1
    raise RuntimeError('pas assez de manches dans certaines régions')

def create_initial_round_state(starter, decks):
    c = decks[np.random.randint(len(decks))].copy()
    return sm.fresh_round_state(2, starter, deck=c)

def simulate_round(state, total, policy):
    H = [Hi.copy() for Hi in state.H]
    S = list(state.S); sigma = list(state.sigma); c = state.c.copy()
    sim = State(H, S, sigma, c, state.current_player)

    while not sm.is_terminal(sim):
        action = policy(sim, total)
        sim = sm.transition(sim, action)
        nxt = sm.get_next_player(sim)
        if nxt is not None:
            sim.current_player = nxt
        sm.refill_deck_if_empty(sim)
    return sim


# 7 cartes distinctes : 12+11+10+9+8+7+6+15 = 78, marge à 95
def compute_round_distribution_bucketed(policy, n_samples=1500, maxd=95, target=200, decks=None):
    if decks is None:
        decks, _ = collect_round_decks(policy, target=target)
    bounds = region_bounds(target)
    p = np.zeros((3, 3, 2, maxd, maxd))
    for r0, (a0, b0) in enumerate(bounds):
        if b0 <= a0:
            continue
        for r1, (a1, b1) in enumerate(bounds):
            if b1 <= a1:
                continue
            for s in (0, 1):
                sm.ROUND_STARTER = s
                for _ in range(n_samples):
                    sm.V_cache.clear()
                    t0 = int(np.random.randint(a0, b0)) # total tiré alea dans la région
                    t1 = int(np.random.randint(a1, b1))
                    end = simulate_round(create_initial_round_state(s, decks[r0, r1, s]), [t0, t1], policy)
                    d0, d1 = sm.banked_increments(end)
                    if max(d0, d1) >= maxd:
                        raise ValueError('maxd trop petit')
                    p[r0, r1, s, d0, d1] += 1
                tot = p[r0, r1, s].sum()
                if tot > 0:
                    p[r0, r1, s] /= tot
    return 0.5 * (p + p.transpose(1, 0, 2, 4, 3)[:, :, ::-1])


def value_iteration_bucketed(p, target=200, maxd=95, max_iter=500, eps=1e-5, verbose=True):
    bounds = region_bounds(target)

    nz = {}
    for r0 in range(3):
        for r1 in range(3):
            for s in (0, 1):
                nz[(r0, r1, s)] = [(d0, d1, p[r0, r1, s, d0, d1])
                                   for d0 in range(maxd) for d1 in range(maxd)
                                   if p[r0, r1, s, d0, d1] > 1e-6]

    A = np.arange(target + maxd)[:, None]
    B = np.arange(target + maxd)[None, :]
    terminal_val = (A > B).astype(float) + 0.5 * (A == B)
    mask_term = (A >= target) | (B >= target)

    W = np.full((target, target, 2), 0.5)
    it, err = 0, 0.0
    for it in range(max_iter):
        W_new = np.zeros_like(W)
        for s in (0, 1):
            ns = 1 - s
            cont = np.where(mask_term, terminal_val, 0.0)
            cont[:target, :target] = W[:, :, ns]
            for r0, (a0, b0) in enumerate(bounds):
                if b0 <= a0:
                    continue
                for r1, (a1, b1) in enumerate(bounds):
                    if b1 <= a1:
                        continue
                    acc = np.zeros((b0 - a0, b1 - a1))
                    for d0, d1, prob in nz[(r0, r1, s)]:
                        acc += prob * cont[a0 + d0:b0 + d0, a1 + d1:b1 + d1]
                    W_new[a0:b0, a1:b1, s] = acc
        err = np.max(np.abs(W_new - W))
        W = W_new
        if err < eps:
            break
    if verbose:
        print(f"  value iteration bucketée : {it+1} itérations, erreur finale {err:.2e}")
    if err >= eps:
        raise RuntimeError('value iteration non convergée')
    return W


def threshold_policy(state, total):
    return sm.seuil_strat(state)

def minimax_policy(depth):
    return lambda state, total: sm.minimax_strat(state, total, depth=depth)

def policy_iteration(n_outer=3, depth=1, n_samples=1500, target=200, maxd=95, verbose=True, history=None):
    sm.TARGET = target
    sm.W_TABLE = None
    sm.V_cache.clear()
    pi = threshold_policy
    W = None
    W_prev = None
    for n in range(n_outer):
        label = "seuil" if n == 0 else f"minimax(depth={depth})"
        if verbose:
            print(f"--- itération {n} : politique = {label} ---")

        t = time.time()
        decks, n_games = collect_round_decks(pi, target=target)
        # On calcule les distributions de proba (18 = 3*3*2)
        p = compute_round_distribution_bucketed(pi, n_samples=n_samples, maxd=maxd, target=target, decks=decks)
        if verbose:
            print(f"  p calculée ({time.time()-t:.1f}s)")

        t = time.time()
        W = value_iteration_bucketed(p, target=target, maxd=maxd, verbose=verbose)
        sm.W_TABLE = W
        W.setflags(write=False)
        sm.V_cache.clear()
        if history is not None:
            history.append({'iteration': n, 'deck_games': n_games,
                            'deck_counts': {str(k): len(v) for k, v in decks.items()},
                            'start_value': float(W[0, 0].mean())})
        if verbose:
            avg00 = 0.5 * (W[0, 0, 0] + W[0, 0, 1])
            diff = "" if W_prev is None else f"  | ||ΔW||={np.max(np.abs(W - W_prev)):.4f}"
            print(f"  W prête ({time.time()-t:.1f}s) | W[0,0]≈{avg00:.3f}{diff}")
        W_prev = W.copy()

        pi = minimax_policy(depth)
    return W


if __name__ == '__main__':
    from play_with_W import main
    main()
