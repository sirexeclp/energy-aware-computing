from gpyjoules.run_benchmark import load_experiment_definition


def test_load_experiment_definition():
    """Test loading the experiment definition."""
    import os
    os.environ["HOST"] = "gpu-dxg-01"
    experiments = load_experiment_definition()
    import os
    os.getcwd()
    print(experiments)