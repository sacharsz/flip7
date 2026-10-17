import csv, hashlib, json, platform, time
from pathlib import Path
import numpy as np
import strategie_minimax as sm
from policy_iteration import policy_iteration


def agent(depth, W):
    def policy(state, total):
        if sm.W_TABLE is not W:
            sm.V_cache.clear()
        sm.W_TABLE = W
        return sm.minimax_strat(state, total, depth=depth)
    return policy

def run_game(strategies, target=200, starter=None):
    sm.TARGET = target
    return sm.game(strategies, starter=starter, verbose=False)

def interval(wins, n):
    z = 1.959963984540054
    p = wins / n
    centre = (p + z*z / (2*n)) / (1 + z*z/n)
    marge = z * np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return [float(centre - marge), float(centre + marge)]

def benchmark(stratA, stratB, n_games=2000, seed=100000, writer=None, label='', target=200):
    if n_games <= 0 or n_games % 4:
        raise ValueError('n_games doit être un multiple positif de 4')
    wins = draws = 0
    timings = []
    counts = {}
    t0 = time.time()

    def timed(state, total):
        t = time.perf_counter()
        action = stratA(state, total)
        timings.append(time.perf_counter() - t)
        return action

    for j in range(n_games):
        seat, starter = j % 2, (j // 2) % 2
        np.random.seed(seed + j)
        strategies = (timed, stratB) if seat == 0 else (stratB, timed)
        total = run_game(strategies, target=target, starter=starter)
        a, b = total[seat], total[1 - seat]
        wins += a > b
        draws += a == b
        counts[seat, starter] = counts.get((seat, starter), 0) + 1
        if writer is not None:
            writer.writerow([label, j, seed+j, seat, starter, a, b])
        if (j + 1) % 250 == 0:
            print(f'  {label}: {j+1}/{n_games}, {(time.time()-t0):.0f}s', flush=True)

    return {'games': n_games, 'wins': int(wins), 'draws': int(draws),
            'losses': int(n_games-wins-draws), 'win_rate': wins/n_games,
            'win_ci95': interval(wins, n_games), 'score_rate': (wins+0.5*draws)/n_games,
            'balanced_counts': {str(k): v for k, v in counts.items()},
            'decision_ms_mean': float(1000*np.mean(timings)),
            'decision_ms_p95': float(1000*np.percentile(timings, 95)),
            'seconds': time.time()-t0}

def winrate(stratA, stratB, n_games, target=200):
    sm.TARGET = target
    # Compatibilité avec les anciens scripts.
    return benchmark(stratA, stratB, n_games, target=target)['win_rate']

def main(depth=2, n_outer=3, n_samples=600, n_games=2000,
         train_seed=7, test_seed=100000, out='results', train=True):
    if min(depth, n_outer, n_samples) <= 0 or n_games <= 0 or n_games % 4:
        raise ValueError('paramètres invalides')
    root = Path(__file__).resolve().parent
    folder = root / out
    folder.mkdir(parents=True, exist_ok=True)
    params = {'depth': depth, 'n_outer': n_outer, 'n_samples': n_samples,
              'target': 200, 'train_seed': train_seed}
    sources = {name: hashlib.sha256((root/name).read_bytes()).hexdigest()
               for name in ('moteur.py', 'strategie_minimax.py', 'policy_iteration.py')}
    sm.TARGET = params['target']

    if train:
        np.random.seed(train_seed)
        history = []
        t = time.time()
        W = policy_iteration(n_outer, depth, n_samples, history=history)
        metadata = {'parameters': params, 'history': history, 'seconds': time.time()-t,
                    'source_sha256': sources}
        np.save(folder / 'W.npy', W)
        (folder / 'training.json').write_text(json.dumps(metadata, indent=2))
    else:
        metadata = json.loads((folder / 'training.json').read_text())
        if metadata['parameters'] != params:
            raise ValueError('paramètres différents de ceux de la table sauvegardée')
        if metadata['source_sha256'] != sources:
            raise ValueError('code différent de celui de la table sauvegardée')
        W = np.load(folder / 'W.npy')
    W.setflags(write=False)

    if not np.isfinite(W).all() or W.min() < -1e-12 or W.max() > 1+1e-12:
        raise ValueError('table de probabilités invalide')
    symmetry = float(np.max(np.abs(W + W.transpose(1, 0, 2)[:, :, ::-1] - 1)))
    if symmetry > 1e-4:
        raise ValueError('table non symétrique')

    results = {'parameters': params, 'test_seed': test_seed, 'training': metadata,
               'model': 'deux joueurs, cartes numériques et seconde chance, égalité = 0.5',
               'symmetry_error': symmetry, 'python': platform.python_version(),
               'numpy': np.__version__, 'machine': platform.machine(),
               'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in root.glob('*.py')}, 'benchmarks': {}}
    opponents = [('seuil_37', sm.to2(sm.seuil_strat)),
                 ('greedy', sm.to2(sm.greedy_strat)),
                 ('minimax_sans_W', agent(depth, None))]
    with (folder / 'games.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['opponent', 'game', 'seed', 'agent_seat', 'starter', 'agent_score', 'opponent_score'])
        for i, (name, opponent) in enumerate(opponents):
            print(f'Benchmark vs {name}', flush=True)
            result = benchmark(agent(depth, W), opponent, n_games,
                               test_seed + i*1000000, writer, name)
            results['benchmarks'][name] = result
            f.flush()
            (folder / 'benchmark.json').write_text(json.dumps(results, indent=2))
            lo, hi = result['win_ci95']
            print(f"  victoires {result['win_rate']:.1%}, IC95 [{lo:.1%}, {hi:.1%}], "
                  f"égalités {result['draws']}, décision {result['decision_ms_mean']:.2f}ms", flush=True)
    return results


if __name__ == '__main__':
    main()
