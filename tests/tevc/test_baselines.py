from dataclasses import asdict

import numpy as np

from tevc.baselines import (DSR_PUBLISHED, METHODS, COTConfig, DSRConfig, SHOTConfig, UDAConfig, config_from_dict,
                            sample_configs, search)
from tevc.optimizers import B
from tevc.recipe import Recipe
from tevc.stream import Environment


def test_candidates_start_from_the_published_setting_and_repeat_within_a_run():
    for m in METHODS:
        c = sample_configs(m, 42)
        assert len(c) == B and c == sample_configs(m, 42)
        assert (c != sample_configs(m, 43)) == (m != "COT")          # COT has no hyperparameters
    assert sample_configs("DSR", 42)[0] == DSRConfig(0.02, 0.1, 0.02, 4) == DSR_PUBLISHED
    assert sample_configs("CDAN", 7)[0] == UDAConfig("CDAN", 1e-3, 1.0)
    assert sample_configs("SHOT", 7)[0] == SHOTConfig(1e-3, 0.3)


def test_result_entries_rebuild_their_configuration():
    for c in (DSR_PUBLISHED, COTConfig(), UDAConfig("DANN", 1e-3, 0.5), SHOTConfig(1e-3, 0.3),
              Recipe(2, 0.3, 0.8, 128, 0.0, 40, True)):
        assert config_from_dict({**asdict(c), "seed": 1, "fitness": 0.2}) == c


def test_search_evaluates_the_candidates_in_order_with_their_own_seeds(fake_evaluator):
    trials = search("CDAN", fake_evaluator, Environment("fake@1000", 1000.0, {}, {}), 7)
    assert [t.recipe for t in trials] == sample_configs("CDAN", 7)
    assert [t.seed for t in trials] == list(range(700, 730))
