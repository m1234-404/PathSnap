
"""Emission probabilities for candidate road arcs.

Classifier probabilities are converted into a probability distribution
over the candidates belonging to each GPS observation.
"""

import numpy as np


def build_emission_probabilities(
    candidates,
    classifier_probabilities,
    obs_col="obs_idx",
    prob_floor=1e-12,
):
    """Build normalized emission probabilities for each observation.

    Parameters
    ----------
    candidates : pandas.DataFrame
        Candidate rows. Must contain obs_idx (or obs_col).
    classifier_probabilities : array-like
        Positive-class classifier probabilities, one per candidate row.
    obs_col : str
        Column identifying the GPS observation.
    prob_floor : float
        Small positive value to avoid taking log(0).

    Returns
    -------
    dict[int, numpy.ndarray]
        Observation index -> normalized candidate probabilities.
        Array order matches the original candidate row order.
    """
    probabilities = np.asarray(classifier_probabilities, dtype=float)

    if obs_col not in candidates.columns:
        raise ValueError(f"Missing candidate column: {obs_col}")

    if len(probabilities) != len(candidates):
        raise ValueError(
            "There must be exactly one classifier probability per candidate."
        )

    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Classifier probabilities must be finite.")

    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Classifier probabilities must be between 0 and 1.")

    if prob_floor <= 0:
        raise ValueError("prob_floor must be positive.")

    result = {}

    obs_values = candidates[obs_col].to_numpy()

    for obs_id in np.unique(obs_values):
        mask = obs_values == obs_id
        scores = np.maximum(probabilities[mask], prob_floor)
        total = scores.sum()

        if not np.isfinite(total) or total <= 0:
            raise ValueError(f"Invalid emission scores for observation {obs_id}")

        result[int(obs_id)] = scores / total

    return result
