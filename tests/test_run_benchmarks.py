from gpyjoules.run_benchmark import load_experiment_definition


def test_load_experiment_definition():
    """Test loading the experiment definition."""
    experiments = load_experiment_definition("gpu-dgx-01")
    assert len(experiments) == 1
    assert experiments[0]["experiment_name"] == "power-limit"
    assert experiments[0]["power_limits"] == [150, 200, 250, 300]
    print(experiments)


def test_load_experiment_definition_with_none():
    """Test loading the experiment definition with a None value."""
    experiments = load_experiment_definition("gpunode-01")
    assert experiments[0]["power_limits"] == [None]

