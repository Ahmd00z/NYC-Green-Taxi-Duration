import json
from pathlib import Path

from prodml.config import settings
from prodml.data import read_dataframe, split_data


def test_read_dataframe_filters_durations(synthetic_csv):
    df = read_dataframe(synthetic_csv)
    assert df["duration"].between(1, 60).all()


def test_split_is_chronological(synthetic_csv):
    train_df, val_df = split_data(read_dataframe(synthetic_csv), 0.8)
    assert len(train_df) > len(val_df)
    assert (
        train_df["lpep_pickup_datetime"].max() <= val_df["lpep_pickup_datetime"].min()
    )


def test_training_writes_all_artifacts(model_dir):
    for name in (settings.pickle_name, settings.onnx_name, settings.metadata_name):
        assert (Path(model_dir) / name).exists()
    meta = json.loads((Path(model_dir) / settings.metadata_name).read_text())
    assert meta["val_mae"] < 3  # fake data is easy: MAE should be small
