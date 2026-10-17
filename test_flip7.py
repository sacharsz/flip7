import unittest
from unittest.mock import patch
import numpy as np
import moteur as m
import strategie_minimax as sm
import policy_iteration as pi
import play_with_W as bench


class Tests(unittest.TestCase):
    def setUp(self):
        sm.W_TABLE = None
        sm.ROUND_STARTER = 0
        sm.TARGET = 200
        sm.V_cache.clear()

    def test_cache(self):
        state = m.fresh_round_state(2, 0)
        state.sigma = ['stayed', 'stayed']
        W = np.empty((200, 200, 2))
        W[:, :, 0], W[:, :, 1] = 0.2, 0.8
        sm.W_TABLE = W
        self.assertAlmostEqual(sm.V(state, [0, 0], 1), 0.8)
        sm.ROUND_STARTER = 1
        self.assertAlmostEqual(sm.V(state, [0, 0], 1), 0.2)
        sm.W_TABLE = np.full_like(W, 0.3)
        self.assertAlmostEqual(sm.V(state, [0, 0], 1), 0.3)

    def test_ties(self):
        self.assertEqual(sm.cumulative_score_heuristic([200, 200]), 0.5)
        p = np.zeros((3, 3, 2, 3, 3))
        p[0, 0, :, 1, 1] = 1
        W = pi.value_iteration_bucketed(p, target=2, maxd=3, verbose=False)
        self.assertEqual(W[0, 0, 0], 0.5)
        self.assertEqual(W[1, 0, 0], 1)

    def test_second_chance(self):
        state = m.fresh_round_state(2, 0)
        state.H[0][4] = state.H[0]['SC'] = 1
        state.S[0] = 4
        sim = sm.transition_dp(state, 'hit', 4)
        self.assertEqual(sim.H[0]['SC'], 0)
        self.assertEqual(sim.S[0], 4)
        self.assertEqual(state.H[0]['SC'], 1)
        sim = sm.transition_dp(sim, 'hit', 4)
        self.assertEqual(sim.sigma[0], 'busted')
        self.assertEqual(sim.S[0], 0)

    def test_seven_cards(self):
        state = m.fresh_round_state(2, 0)
        for k in range(1, 7):
            state.H[0][k] = 1
        state.S[0] = 21
        state.c = dict.fromkeys(m.c0, 0)
        state.c[7] = 1
        self.assertEqual(sm.expected_one_hit(state, 0), 43)
        sim = sm.transition_dp(state, 'hit', 7)
        self.assertTrue(m.is_terminal(sim))
        self.assertEqual(m.banked_increments(sim), [43, 0])
        self.assertTrue(all(v >= 0 for v in sim.c.values()))

    def test_deck(self):
        state = m.fresh_round_state(2, 0, dict.fromkeys(m.c0, 0))
        self.assertEqual(state.c, m.c0)
        sim = sm.transition_dp(state, 'hit', 2)
        self.assertEqual(state.c[2], 2)
        self.assertEqual(sim.c[2], 1)
        empirical = m.c0.copy()
        empirical[12] = 2
        sim = pi.create_initial_round_state(0, [empirical])
        self.assertEqual(sim.c[12], 2)
        sm.transition_dp(sim, 'hit', 12)
        self.assertEqual(empirical[12], 2)

    def test_probability_table(self):
        np.random.seed(8)
        decks = {(r0, r1, s): [m.c0.copy()] for r0 in range(3) for r1 in range(3) for s in (0, 1)}
        p = pi.compute_round_distribution_bucketed(pi.threshold_policy, n_samples=20, decks=decks)
        np.testing.assert_allclose(p.sum(axis=(3, 4)), 1)
        W = pi.value_iteration_bucketed(p, verbose=False)
        self.assertTrue(np.isfinite(W).all())
        self.assertGreaterEqual(W.min(), 0)
        self.assertLessEqual(W.max(), 1)
        np.testing.assert_allclose(W + W.transpose(1, 0, 2)[:, :, ::-1], 1, atol=1e-8)

    def test_benchmark(self):
        seats = []
        def fake(strategies, target=200, starter=None):
            seats.append(starter)
            state = m.fresh_round_state(2, 0)
            strategies[0](state, [0, 0])
            strategies[1](state, [0, 0])
            return [200, 200]
        with patch.object(bench, 'run_game', side_effect=fake):
            result = bench.benchmark(sm.to2(sm.seuil_strat), sm.to2(sm.seuil_strat), n_games=8)
        self.assertEqual(result['wins'], 0)
        self.assertEqual(result['draws'], 8)
        self.assertEqual(result['score_rate'], 0.5)
        self.assertEqual(sorted(result['balanced_counts'].values()), [2, 2, 2, 2])
        self.assertEqual(seats, [0, 0, 1, 1, 0, 0, 1, 1])
        lo, hi = bench.interval(1000, 2000)
        self.assertLess(lo, 0.5)
        self.assertGreater(hi, 0.5)
        self.assertLess(hi-lo, 0.05)

    def test_complete_games(self):
        policies = (sm.to2(sm.seuil_strat), sm.to2(sm.greedy_strat))
        np.random.seed(21)
        starts = []
        first = sm.game(policies, starter=0, verbose=False, round_starts=starts)
        np.random.seed(21)
        self.assertEqual(first, sm.game(policies, starter=0, verbose=False))
        self.assertGreaterEqual(max(first), 200)
        self.assertTrue(all(0 <= total[i] < 200 for total, _, _ in starts for i in (0, 1)))
        self.assertTrue(any(c != m.c0 for _, _, c in starts[1:]))
        self.assertTrue(all(c[k] >= 0 for _, _, c in starts for k in c))


if __name__ == '__main__':
    unittest.main()
