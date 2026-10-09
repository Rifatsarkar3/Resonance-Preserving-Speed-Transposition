import numpy as np

from tevc.optimizers import B, N_GENES, decode, encode, ga_search, polynomial_mutation, random_search, sbx
from tevc.recipe import ANCHORS, COPIES, N_FFTS, sample
from tevc.stream import Environment

ENV = Environment("fake@1000", 1000.0, {}, {})


def test_decode_always_gives_a_valid_recipe():
    rng = np.random.default_rng(0)
    for g in np.vstack([rng.random((500, N_GENES)), np.zeros(N_GENES), np.ones(N_GENES)]):
        r = decode(g)
        assert 1 <= r.n_targets <= 4 and 0.2 <= r.q_lo <= r.q_hi <= 1.2 and 0.0 <= r.beta <= 1.0
        assert r.n_fft in N_FFTS and r.copies in COPIES and isinstance(r.keep_real, bool)


def test_encode_decode_round_trip():
    rng = np.random.default_rng(1)
    for r in [sample(rng) for _ in range(300)] + [ANCHORS["H2_LIKE"], ANCHORS["RESAMPLE_H2"]]:
        assert decode(encode(r)) == r


def test_operators_stay_in_the_unit_box():
    rng = np.random.default_rng(2)
    for _ in range(200):
        a, b = rng.random(N_GENES), rng.random(N_GENES)
        for c in sbx(a, b, rng) + (polynomial_mutation(a, rng),):
            assert np.all((c >= 0.0) & (c <= 1.0))


def test_ga_spends_exactly_the_budget_and_reevaluates_top3_with_the_second_seed(fake_evaluator):
    trials = ga_search(fake_evaluator, ENV, np.random.default_rng(3).random((12, N_GENES)),
                       np.random.default_rng(4), (1, 2))
    assert len(trials) == B == 30
    assert [t.seed for t in trials].count(2) == 3
    assert len({(t.recipe, t.seed) for t in trials}) == 30         # no evaluation wasted on a duplicate


def test_ga_improves_on_its_initial_population(fake_evaluator):
    trials = ga_search(fake_evaluator, ENV, np.random.default_rng(5).random((12, N_GENES)),
                       np.random.default_rng(6), (1, 2))
    assert max(t.fitness for t in trials[12:27]) > max(t.fitness for t in trials[:12])


def test_random_search_shares_its_first_12_recipes_with_the_ga_start(fake_evaluator):
    rand = random_search(fake_evaluator, ENV, np.random.default_rng(7), seed=1)
    ga = ga_search(fake_evaluator, ENV, np.random.default_rng(7).random((12, N_GENES)), np.random.default_rng(8),
                   (1, 2))
    assert len(rand) == 30 and [t.recipe for t in rand[:12]] == [t.recipe for t in ga[:12]]
