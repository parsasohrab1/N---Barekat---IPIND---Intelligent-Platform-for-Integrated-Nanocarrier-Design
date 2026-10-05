"""Shared test fixtures.

``smoke`` models are trained once for the whole session (~20 seconds). Their accuracy does not matter;
the goal of the tests is the correctness of the pipeline's *behavior*. Real accuracy is reported in ``docs/MODEL_VALIDATION.md`` and
with the ``standard`` profile.
"""

import os

import pytest

# Test-only keys (never to be used in production)
os.environ.setdefault("IPIND_JWT_SECRET", "test-only-jwt-secret-0123456789abcdef0123456789")


@pytest.fixture(scope="session")
def encryption_key():
    from ipind2.security import generate_key

    return generate_key()


@pytest.fixture(scope="session")
def smoke_bundle():
    from ipind2.training import train_bundle

    return train_bundle("smoke", seed=123, verbose=False)


@pytest.fixture(scope="session")
def small_dataset():
    from ipind2.data_generation.synthetic_data_generator import SyntheticDataGenerator

    return SyntheticDataGenerator(77).generate_dataset(400, include_pareto_labels=False)
