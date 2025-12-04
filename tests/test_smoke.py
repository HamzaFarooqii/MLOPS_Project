def test_smoke_imports():
    # Basic import smoke tests to ensure modules load
    import scripts.extract  # noqa: F401
    import scripts.transform  # noqa: F401
    import scripts.quality_check  # noqa: F401
    import scripts.train  # noqa: F401

