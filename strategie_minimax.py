import numpy as np

from moteur import (
    State, c0,
    get_cardinal, get_next_player,
    is_terminal, maj_score, transition, state_description,
    refill_deck_if_empty, fresh_round_state, banked_increments,
    main_strat, dumb_strat,
)


def transition_dp(state:State, action, k=None):
    H = [Hi.copy() for Hi in state.H]
    S = list(state.S)
    sigma = list(state.sigma)
    c = state.c.copy()
    sim = State(H, S, sigma, c, state.current_player)
    return transition(sim, action, k)


TARGET = 200
W_TABLE = None
ROUND_STARTER = None

def cumulative_score_heuristic(total:list):
    t0, t1 = total[0], total[1]
    if max(total) >= TARGET:
        return 0.5 if t0 == t1 else float(t0 > t1)
    if W_TABLE is not None:
        if ROUND_STARTER is None:
            return 0.5 * (W_TABLE[t0, t1, 0] + W_TABLE[t0, t1, 1])
        return W_TABLE[t0, t1, 1 - ROUND_STARTER]
    s = t0 + t1
    return t0 / s if s > 0 else 0.5

def terminal_round_value(state:State, total:list):
    inc = banked_increments(state)
    new_total = [total[i] + inc[i] for i in range(len(total))]
    return cumulative_score_heuristic(new_total)

SEUIL = 37

def seuil_strat(state:State, seuil=SEUIL):
    i = state.current_player
    return 'stay' if state.S[i] >= seuil else 'hit'

def rollout_round(state:State, rollout_strats):
    H = [Hi.copy() for Hi in state.H]
    S = list(state.S)
    sigma = list(state.sigma)
    c = state.c.copy()
    sim = State(H, S, sigma, c, state.current_player)

    while not is_terminal(sim):
        i = sim.current_player
        action = rollout_strats[i](sim)
        sim = transition(sim, action)
        next_player = get_next_player(sim)
        if next_player is not None:
            sim.current_player = next_player
        refill_deck_if_empty(sim)
    return sim

ROLLOUT_STRATS = (seuil_strat, seuil_strat)

def expected_one_hit(state:State, i): # Calcule l'esperence du score de la manche après 1 tirage
    Hi = state.H[i]
    Si = state.S[i]
    c = state.c
    N = sum(c.values())
    bonus = 15 if get_cardinal(Hi) == 6 else 0
    if N == 0:
        return Si
    if Hi['SC'] == 0:
        E = 0.0
        for k, ck in c.items():
            if ck == 0:
                continue
            pk = ck / N
            if k == 'SC':
                E += pk * Si
            elif Hi[k] > 0:
                E += pk * 0.0
            else:
                E += pk * (Si + k + bonus)
        return E
    E = float(Si)
    for k, ck in c.items():
        if ck == 0 or k == 'SC':
            continue
        if Hi[k] == 0:
            E += (ck / N) * (k + bonus)
    return E

def greedy_strat(state:State):
    i = state.current_player
    return 'hit' if state.S[i] == 0 or expected_one_hit(state, i) > state.S[i] else 'stay'

def round_increment_estimate(state:State, i):
    sigma = state.sigma[i]
    if sigma == 'busted':
        return 0
    if sigma == 'stayed':
        return state.S[i]
    return max(state.S[i], expected_one_hit(state, i))

def heuristic_value(state:State, total:list):
    t0 = total[0] + round_increment_estimate(state, 0)
    t1 = total[1] + round_increment_estimate(state, 1)
    return cumulative_score_heuristic([int(round(t0)), int(round(t1))])


DEPTH = 4
V_cache = {}

def encode(state:State, total:list, depth):
    H = tuple(tuple(Hi[k] for k in c0) for Hi in state.H)
    c = tuple(state.c[k] for k in c0)
    return (H, tuple(state.S), tuple(state.sigma), c, state.current_player,
            tuple(total), depth, ROUND_STARTER, TARGET, id(W_TABLE))

def get_values(state:State, total:list, depth):
    s_stay = transition_dp(state, 'stay')
    next_player = get_next_player(s_stay)
    if next_player is not None:
        s_stay.current_player = next_player
    v_stay = V(s_stay, total, depth - 1)

    c = state.c
    n = sum(c.values())
    v_hit = 0.0
    for k in c:
        if c[k] == 0:
            continue
        s_hit = transition_dp(state, 'hit', k)
        next_player = get_next_player(s_hit)
        if next_player is not None:
            s_hit.current_player = next_player
        v_hit += c[k] / n * V(s_hit, total, depth - 1)
    return v_hit, v_stay

def V(state:State, total:list, depth):
    key = encode(state, total, depth)
    if key in V_cache:
        return V_cache[key]

    if is_terminal(state):
        v = terminal_round_value(state, total)
    elif depth <= 0:
        v = heuristic_value(state, total)
    else:
        v_hit, v_stay = get_values(state, total, depth)
        i = state.current_player
        v = max(v_hit, v_stay) if i == 0 else min(v_hit, v_stay)

    V_cache[key] = v
    return v


def minimax_strat(state:State, total:list, depth=DEPTH):
    v_hit, v_stay = get_values(state, total, depth)
    if state.current_player == 0:
        return 'hit' if v_hit >= v_stay else 'stay'
    return 'hit' if v_hit <= v_stay else 'stay'

def to2(strat):
    return lambda state, total: strat(state)


def game(strategies=(minimax_strat, minimax_strat), starter=None, verbose=True, round_starts=None):
    global ROUND_STARTER
    n = len(strategies)
    total = [0] * n
    c = c0.copy()
    counter = 0
    if n != 2:
        raise ValueError('minimax prévu pour deux joueurs')
    starter = np.random.randint(n) if starter is None else starter
    while max(total) < TARGET:
        if counter >= 10000:
            raise RuntimeError('partie bloquée')
        ROUND_STARTER = starter
        state = fresh_round_state(n, starter, deck=c)
        V_cache.clear()
        if round_starts is not None:
            round_starts.append((total.copy(), starter, c.copy()))

        while not is_terminal(state):
            curr = state.current_player
            action = strategies[curr](state, total)
            state = transition(state, action)
            #if counter == 0 :
            next_player = get_next_player(state)
            if next_player is not None:
                state.current_player = next_player
            refill_deck_if_empty(state)
            if verbose:
                state_description(state, total)

        maj_score(state, total)
        starter = 1 - starter
        counter += 1

    if verbose:
        gagnant = 'égalité' if total[0] == total[1] else f'J{int(np.argmax(total))}'
        print(f"\n=== Total final : {total} | gagnant : {gagnant} ===")
    return total

if __name__ == '__main__':
    count_total = []
    n_party = 1
    for party in range(n_party):
        print("commence : ", party)
        count_total.append(game())
        print("termine : ", party)
    win_ratio = sum(tt[0] > tt[1] for tt in count_total) / n_party
    print("win_ratio : ", win_ratio)
