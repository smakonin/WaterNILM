import itertools
import json
import math
import random
import unittest

from waternilm.model import Model, activity_ranges, capped_viterbi, train
from notebook_reference import train as original_train, viterbi as original_decode


def path_score(path, observed, model):
    history = (0,) * model.order
    padded = [0] * model.order + observed
    score = 0.0
    for t, state in enumerate(path, model.order):
        next_history = history[1:] + (state,)
        transition = model.transition.get(history, {}).get(state, 0)
        emission = model.emission.get(next_history, {}).get(tuple(padded[t-model.order+1:t+1]), 0)
        if not transition or not emission:
            return -math.inf
        score += math.log(transition) + math.log(emission)
        history = next_history
    return score


class ModelTests(unittest.TestCase):
    def test_activity_edges(self):
        self.assertEqual(activity_ranges([]), [])
        self.assertEqual(activity_ranges([0, 0]), [])
        self.assertEqual(activity_ranges([1, 0, 1]), [(0, 1), (2, 3)])
        self.assertEqual(activity_ranges([1, 0, 1], 1), [(0, 3)])
        self.assertEqual(activity_ranges([0, 1, 1]), [(1, 3)])

    def test_training_and_legacy_decoder_match_notebook(self):
        rng = random.Random(13)
        hidden = [rng.randrange(3) for _ in range(90)]
        observed = [rng.randrange(3) for _ in hidden]
        intervals = [(i, i+9) for i in range(0, 90, 9)]
        for order in range(1, 5):
            with self.subTest(order=order):
                expected_t, expected_e = original_train(hidden, observed, 3, intervals, order=order)
                model = train(hidden, observed, intervals, num_observed=3, order=order)
                self.assertEqual(model.transition, expected_t)
                self.assertEqual(model.emission, expected_e)
                for start, end in intervals:
                    expected = original_decode(observed[start:end], hidden[start:end], expected_t, expected_e, order)
                    self.assertEqual(capped_viterbi(observed[start:end], hidden[start:end], model, mode="legacy"), expected)

    def test_exact_matches_exhaustive_enumeration(self):
        rng = random.Random(41)
        for order in range(1, 5):
            hidden = [rng.randrange(2) for _ in range(100)]
            observed = [rng.randrange(2) for _ in hidden]
            model = train(hidden, observed, [(i, i+10) for i in range(0, 100, 10)], num_observed=2, order=order)
            for _ in range(10):
                obs = [rng.randrange(2) for _ in range(7)]
                caps = [rng.randrange(2) for _ in obs]
                result = capped_viterbi(obs, caps, model)
                best = max(path_score(list(path), obs, model)
                           for path in itertools.product(*[range(c+1) for c in caps]))
                self.assertAlmostEqual(path_score(result, obs, model), best, places=10)
                self.assertTrue(all(0 <= x <= c for x, c in zip(result, caps)))

    def test_history_pruning_can_lose_optimum(self):
        # Both histories ending in state 0 must survive time 2: the one with
        # a worse prefix has the better transition at time 3.
        model = Model(2, 2, .5,
                      {(0,0): {0:.9, 1:.1}, (0,1): {0:1.0}, (1,0): {1:1.0}},
                      {(0,0): {(0,0):.5, (0,1):.01, (1,1):.49},
                       (0,1): {(0,0):.5, (0,1):.5},
                       (1,0): {(0,0):1.0}})
        obs, caps = [0, 0, 1], [1, 0, 1]
        exact = capped_viterbi(obs, caps, model)
        legacy = capped_viterbi(obs, caps, model, mode="legacy")
        self.assertGreater(path_score(exact, obs, model), path_score(legacy, obs, model))

    def test_log_decoder_survives_long_sequence(self):
        model = Model(1, 2, .5, {(0,): {0:1.0}}, {(0,): {(0,):.01, (1,):.99}})
        self.assertEqual(capped_viterbi([0]*2000, [0]*2000, model), [0]*2000)

    def test_invalid_and_impossible_inputs(self):
        model = train([0,1,0], [0,1,0], [(0,3)], num_observed=2)
        self.assertEqual(capped_viterbi([], [], model), [])
        for observed, caps in [([0], []), ([2], [1]), ([0], [-1]), ([.5], [0])]:
            with self.assertRaises(ValueError):
                capped_viterbi(observed, caps, model)
        impossible = Model(1, 1, .5, {(0,): {1:1.0}}, {(1,): {(0,):1.0}})
        with self.assertRaisesRegex(ValueError, "No feasible"):
            capped_viterbi([0], [0], impossible)
        with self.assertRaises(ValueError):
            train([0], [0], [], num_observed=1)

    def test_serialization_and_non_mutating_prediction(self):
        model = train([0,1,0,1], [0,1,0,1], [(0,4)], num_observed=2)
        before = json.dumps(model.to_dict(), sort_keys=True)
        restored = Model.from_dict(json.loads(before))
        self.assertEqual(model, restored)
        capped_viterbi([0,1], [0,1], model)
        self.assertEqual(before, json.dumps(model.to_dict(), sort_keys=True))
        malformed = model.to_dict()
        malformed["transition"][0][1][0][1] = -1
        with self.assertRaises(ValueError):
            Model.from_dict(malformed)


if __name__ == "__main__":
    unittest.main()
