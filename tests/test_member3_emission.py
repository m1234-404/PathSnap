
import numpy as np
import pandas as pd
import pytest

from member_3_pavankumar.hmm.emission import (
    build_emission_probabilities,
)


def test_emissions_normalize_per_observation():
    candidates = pd.DataFrame({
        "obs_idx": [0, 0, 1, 1],
        "arc_id": [10, 11, 12, 13],
    })

    result = build_emission_probabilities(
        candidates,
        [0.8, 0.2, 0.3, 0.7],
    )

    assert np.allclose(result[0], [0.8, 0.2])
    assert np.allclose(result[1], [0.3, 0.7])


def test_emissions_reject_wrong_probability_count():
    candidates = pd.DataFrame({
        "obs_idx": [0, 0],
        "arc_id": [10, 11],
    })

    with pytest.raises(ValueError):
        build_emission_probabilities(candidates, [0.8])


def test_emissions_reject_invalid_probabilities():
    candidates = pd.DataFrame({
        "obs_idx": [0],
        "arc_id": [10],
    })

    with pytest.raises(ValueError):
        build_emission_probabilities(candidates, [1.5])
