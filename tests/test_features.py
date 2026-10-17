import pytest

from prodml.features import build_features


@pytest.mark.parametrize(
    "pickup, dropoff, distance, expected",
    [
        (43, 236, 2.5, {"PU_DO": "43_236", "trip_distance": 2.5}),  # normal
        (None, 236, 2.5, {"PU_DO": "unknown_236", "trip_distance": 2.5}),  # missing
        (43, 236, 0, {"PU_DO": "43_236", "trip_distance": 0.0}),  # zero distance
        ("43", "236", "2.5", {"PU_DO": "43_236", "trip_distance": 2.5}),  # strings
    ],
)
def test_build_features_edge_cases(pickup, dropoff, distance, expected):
    assert build_features(pickup, dropoff, distance) == expected


def test_unseen_pair_does_not_crash(trained_model):
    """A PU_DO pair never seen in training must still give a number."""
    result = trained_model.predict_one({"PU_DO": "9999_9999", "trip_distance": 3.0})
    assert isinstance(result, float)
