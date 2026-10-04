"""فیکسچرهای مشترک تست‌ها.

مدل‌های ``smoke`` یک‌بار برای کل جلسه آموزش می‌بینند (~۲۰ ثانیه). دقت آن‌ها اهمیتی ندارد؛
هدف تست‌ها درستی *رفتار* خط لوله است. دقت واقعی در ``docs/MODEL_VALIDATION.md`` و با
پروفایل ``standard`` گزارش می‌شود.
"""

import os

import pytest

# کلیدهای آزمایشی فقط برای تست (هرگز در تولید استفاده نشوند)
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
