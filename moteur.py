import numpy as np
from dataclasses import dataclass

c0 = {i:i for i in range(13)}
c0['SC'] = 3
c0[0]=1

@dataclass
class State:
    H: list[dict]
    S: list[int]
    sigma: list
    c: dict
    current_player: int
    last_draw: object = None   # dernière carte tirée (None si 'stay' ou état initial)

# Fonction utiles :
def get_cardinal(H):
    return sum([H[k] for k in H if k != 'SC'])

def get_sum(H):
    return sum(k * n for k, n in H.items() if k != 'SC')

def get_next_player(state:State):
    n = len(state.sigma)
    next = (state.current_player + 1) % n
    while next != state.current_player:
        if state.sigma[next] == 'playing':
            return next
        next = (next + 1) % n
    return None

def get_next_SC(state:State):
    n = len(state.H)
    next = (state.current_player + 1) % n
    while next != state.current_player:
        if state.H[next]['SC'] == 0 and state.sigma[next] == 'playing':
            return next
        next = (next + 1) % n
    return None

def banked_increments(state:State):
    out = []
    for i in range(len(state.H)):
        if state.sigma[i] == 'busted':
            out.append(0)
        else:
            bonus = 15 if get_cardinal(state.H[i]) == 7 else 0
            out.append(state.S[i] + bonus)
    return out

def maj_score(state:State, total:list):
    inc = banked_increments(state)
    for i in range(len(total)):
        total[i] += inc[i]

def fresh_round_state(n, starter, deck=None):
    model = {i: 0 for i in range(13)}; model['SC'] = 0
    H = [model.copy() for _ in range(n)]
    c = c0.copy() if deck is None else deck
    state = State(H, [0] * n, ['playing'] * n, c, starter)
    refill_deck_if_empty(state)
    return state
    
def is_terminal(state:State):
    return all(s in ['busted', 'stayed'] for s in state.sigma)

def state_description(state:State, total=None):
    if state.last_draw is not None:
        print(f"J{state.current_player} tire : {state.last_draw}")
    print(f"--- Joueur courant : {state.current_player} ---")
    for i in range(len(state.H)):
        cartes = [k for k, n in state.H[i].items() for _ in range(n)]
        marqueur = " <-" if i == state.current_player else ""
        cumul = f" | total={total[i]:>3}" if total is not None else ""
        print(f"J{i} | score={state.S[i]:>3} | {state.sigma[i]:<7}{cumul} | cartes={cartes}{marqueur}")
    restantes = sum(state.c.values())
    print(f"Pioche : {restantes} cartes restantes")

def refill_deck_if_empty(state: State):
    if sum(state.c.values()) == 0:
        n = len(state.H)
        cards_in_hands = {
            k: sum(state.H[i][k] for i in range(n)) for k in c0}
        for k in c0:
            state.c[k] = c0[k] - cards_in_hands[k]


# Transitions entre états

def transition(state:State, action, k=None):
    # Hi -> deck de i 
    # Si -> Score courant de i
    # c -> cartes restantes dans la pioche 
    # t -> tour
    # sigma -> (sigmai) avec sigmai in {'playing', 'stayed', 'busted'}
    H, S, sigma, c, i = state.H, state.S, state.sigma, state.c, state.current_player
    drawn = None
    if action == 'hit':
        refill_deck_if_empty(state)
        if k is None:
            keys = list(c.keys())
            counts = np.array(list(c.values()), dtype=float)
            idx = np.random.choice(len(keys), p=counts/counts.sum())
            k = keys[idx]
        if c[k] <= 0:
            raise ValueError('carte absente de la pioche')
        drawn = k
        c[k] -= 1

        if H[i][k] > 0: # si carte tirée est dans Hi
            if k == 'SC': 
                next_available_SC = get_next_SC(state)
                if next_available_SC != None:
                    state.H[next_available_SC]['SC'] += 1
                else : pass # si tout aucun joueur ne peut recevoir SC, on défausse la carte
            else:
                if H[i]['SC'] > 0: # on a une carte seconde chance
                    H[i]['SC'] -= 1 
                else: # on bust 
                    sigma[i] = 'busted'
                    S[i] = 0  # un joueur qui bust perd les points de la manche
        else: # Si la carte tirée n'est pas dans Hi
            # on l'ajoute à H 
            H[i][k] += 1
            if k != 'SC':
                S[i] += k  # mise à jour du score courant de la manche
            if get_cardinal(H[i]) == 7:
                for j in range(len(state.sigma)):
                    if state.sigma[j] == 'playing' : sigma[j] = 'stayed' 
                    

    elif action == 'stay':
        sigma[i] = 'stayed'
    else : 
        raise ValueError('action invalide')
    new_state = State(H, S, sigma, c, i, drawn)
    refill_deck_if_empty(new_state)
    return new_state


# Stratégies 

def main_strat(state):
    if state.S[state.current_player] < 37:
        return "hit"
    return "stay"
    
def dumb_strat(state:State)-> State:
    if get_cardinal(state.H[state.current_player]) < 7:
        return 'hit'
    else:
        return 'stay'
    

# Simulation de la partie

def game(strategies=(main_strat, dumb_strat)):
    n = len(strategies)
    total = [0] * n                      # score cumulé sur toute la partie
    c = c0.copy()
    counter = 0
    current_player = np.random.randint(n)
    while max(total)<200:               # à remplacer par while max(total) < 200 ensuite
        # --- nouvelle manche : on réinitialise mains, statuts, pioche et joueur ---
        current_player = np.mod(current_player + 1, n)
        state = fresh_round_state(n, current_player, deck=c)

        # --- déroulé de la manche ---
        while not is_terminal(state):
            curr = state.current_player
            action = strategies[curr](state) 
            state = transition(state, action)      # le joueur courant joue (hit/stay)
            #if counter in [0,1,2] :state_description(state, total)
            next_player = get_next_player(state)     # prochain joueur encore 'playing'
            if next_player is not None:
                state.current_player = next_player
            # sinon : plus personne d'autre, le joueur courant continue seul
            
        # --- fin de manche : on banke les points dans le score cumulé ---
        maj_score(state, total)
        counter += 1
        

    gagnant = 'égalité' if total.count(max(total)) > 1 else f'J{int(np.argmax(total))}'
    print(f"\n=== Total final : {total} | gagnant : {gagnant} ===")
    return total

if __name__ == '__main__':
    game()
    
    
