import numpy as np
from pyuncertainnumber import area_metric
import scipy.stats as sps


def test_area_metric_on_sample_data():
    """test area_metric function on sample data and verify against scipy wasserstein_distance"""
    rng = np.random.default_rng(42)
    sample1 = rng.normal(loc=0.0, scale=1.0, size=500)
    sample2 = rng.normal(loc=0.5, scale=1.2, size=800)

    am_test = area_metric(sample1, sample2)
    am_ref = sps.wasserstein_distance(sample1, sample2)
    assert np.isclose(am_test, am_ref)
