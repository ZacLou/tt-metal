import pytest
from fuser.validator import FpuBlockDefaultsSchema, FpuBlockSchema, PackBlockSchema


def test_block_resolves_lists_scalars_sequences_and_auto():
    block = FpuBlockSchema.model_validate(
        {
            "in0": {"start": 1, "step": 2, "count": 4},
            "in1": 0,
            "dest": "auto",
        }
    ).resolve(("in0", "in1", "dest"))

    assert [(call.in0, call.in1, call.dest) for call in block.calls] == [
        (1, 0, 0),
        (3, 0, 1),
        (5, 0, 2),
        (7, 0, 3),
    ]


def test_block_preserves_mapping_order():
    block = PackBlockSchema.model_validate(
        {"dest": [3, 1, 2, 0], "out": [0, 1, 2, 3]}
    ).resolve(("dest", "out"))

    assert [(call.dest, call.out) for call in block.calls] == [
        (3, 0),
        (1, 1),
        (2, 2),
        (0, 3),
    ]


def test_block_rejects_different_mapping_lengths():
    schema = FpuBlockSchema.model_validate({"in0": [0, 1], "in1": [0], "dest": [0, 1]})

    with pytest.raises(ValueError, match="equal lengths"):
        schema.resolve(("in0", "in1", "dest"))


def test_block_rejects_unknown_fields():
    with pytest.raises(ValueError):
        FpuBlockSchema.model_validate({"in0": [0], "unknown": [0]})


def test_block_accepts_in_bounds_negative_sequence():
    block = FpuBlockSchema.model_validate(
        {"in0": {"start": 3, "step": -1, "count": 4}}
    ).resolve(("in0",))

    assert [call.in0 for call in block.calls] == [3, 2, 1, 0]


def test_block_rejects_negative_resolved_index():
    block = FpuBlockSchema.model_validate({"in0": {"start": 1, "step": -2, "count": 2}})

    with pytest.raises(ValueError, match="non-negative"):
        block.resolve(("in0",))


def test_block_rejects_empty_mapping():
    with pytest.raises(ValueError):
        FpuBlockSchema.model_validate({"in0": []})


def test_block_defaults_round_trip():
    defaults = FpuBlockDefaultsSchema.model_validate({"in1": 0})

    assert FpuBlockDefaultsSchema.model_validate(defaults.model_dump()) == defaults
