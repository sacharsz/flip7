# Flip 7: a search agent for a push-your-luck card game

An agent for the two-player version of Flip 7, a card game where each turn you either draw another card or stop and bank your points.
The agent combines a depth-limited expectiminimax search with a table of game-winning probabilities, learned by approximate policy iteration.

Over 2,000 test games per opponent, it wins 55.8% of games against a greedy expected-score strategy, and 77.3% against a fixed 37-point threshold.

## The game

The deck has 79 number cards (one 0, and *k* copies of each value *k* from 1 to 12) plus 3 "Second Chance" cards.
On each turn, a player either **hits** (draws a card) or **stays** (keeps the sum of their cards for the round).

- Drawing a number already in your hand busts you: you score 0 for the round.
- A Second Chance card cancels one bust.
- Collecting 7 distinct numbers gives a 15-point bonus and ends the round for everyone.
- The deck carries over between rounds, and the first player to reach 200 points wins.

The other special cards of the commercial game (Freeze, Flip Three, score modifiers) are not modelled.

## Method

The difficulty is that a good decision depends on two horizons at once: the current round (risk of busting given the cards left in the deck) and the whole game (how much risk is worth taking given both cumulative scores).

**1. Round level: expectiminimax search.**
At each decision, the agent searches two plies ahead.
Draws are chance nodes weighted by the exact composition of the remaining deck, and the two players are max and min nodes.

**2. Game level: a value table `W`.**
`W[t0, t1, s]` estimates the probability that player 0 wins the game, given the cumulative scores `t0` and `t1` at the start of a round and the player `s` who starts it.
It is used to evaluate the end of a round, and the leaves of the search.

`W` is the fixed point of a Bellman equation over rounds, and is computed by value iteration.
It needs the distribution of round score increments, which is estimated by Monte Carlo: 600 simulated rounds for each combination of score region and starting player.
To keep this tractable, scores are grouped into three regions (0-150, 151-190, 191-199), and the deck at the start of each simulated round is sampled from real game trajectories.

**3. Policy iteration.**
Training starts from the 37-point threshold policy, computes `W`, then replaces the policy by the search agent using that `W`, and repeats three times.
Training takes about 8 minutes on an Apple Silicon laptop CPU.

## Results

The table is frozen after training, then evaluated against three opponents, 2,000 games each, with different random seeds.
The agent's seat and the starting player are balanced (500 games per combination).

| Opponent | Win rate | 95% CI (Wilson) | Draws |
|---|---:|---|---:|
| Stop at 37 points | 77.3% | 75.4% - 79.1% | 6 |
| Greedy: draw while the expected score after one more card is higher | 55.8% | 53.6% - 58.0% | 7 |
| Same search without the value table | 56.7% | 54.5% - 58.8% | 7 |

The last line isolates the contribution of `W`: with the same search depth, adding the game-level value table wins 56.7% of games head to head.
A decision takes about 1.1 ms on average.

**Limitations.**
The search is depth-limited and the score regions are coarse, so this is an approximate method with no optimality guarantee.
The confidence intervals cover the test games for this one trained table, not the variability between training runs.

## Usage

Requires Python 3.9+ and NumPy.

```bash
pip install -r requirements.txt

python3 -m unittest -v test_flip7.py                          # 8 tests
python3 play_with_W.py                                         # train, then run the 3 benchmarks
python3 -c "from play_with_W import main; main(train=False)"   # benchmarks only, with the saved table
```

Training uses seed 7, and the benchmarks use seeds 100000-101999, 1100000-1101999 and 2100000-2101999.
Loading the saved table checks that the parameters and the SHA-256 of the source files match those used for training.

## Files

| File | Content |
|---|---|
| `moteur.py` | Game engine: state, transitions, rules |
| `strategie_minimax.py` | Expectiminimax search, baseline strategies |
| `policy_iteration.py` | Monte Carlo estimation of round outcomes, value iteration, policy iteration |
| `play_with_W.py` | Training and benchmark script |
| `test_flip7.py` | Unit tests: rules (Second Chance, 7-card bonus, deck refill), ties, cache, value table, benchmark, complete games |
| `results/W.npy` | Trained value table, shape (200, 200, 2) |
| `results/training.json` | Training parameters, history and source hashes |
| `results/benchmark.json` | Wins, losses, draws, confidence intervals and decision times |
| `results/games.csv` | Seed, seats and final score of every test game |

Code identifiers and comments are in French.
