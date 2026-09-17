from pathlib import Path
import sys

sys.path.append(str(Path.cwd()))

from examples.custom_methods import DEFAULT_METHODS
from examples.run_framework import _parse_args, _selected_methods


def test_default_custom_method_keys_are_framework_families():
    assert DEFAULT_METHODS == {
        "opf": "socp",
        "dispatch": "dummy_merit_order",
        "physical": "bfsa",
        "economic": "bfsa_dlmp",
    }


def test_framework_preset_can_be_overridden_by_family():
    args = _parse_args(["--preset", "comparison", "--opf", "none", "--economic", "none"])

    selected = _selected_methods(args)

    assert selected["opf"] == "none"
    assert selected["dispatch"] == "dummy_merit_order"
    assert selected["physical"] == "bfsa"
    assert selected["economic"] == "none"
