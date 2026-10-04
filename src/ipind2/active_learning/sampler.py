"""
نمونه‌برداری آگاه از عدم‌قطعیت + Query-by-Committee (FR-06).

معیار عدم‌قطعیت = واریانس پیش‌بینی بین اعضای ensemble (SRS §4.6). برای اینکه چند ساختار
بسیار مشابه هم‌زمان برای آزمایش پیشنهاد نشود (هزینه واقعی آزمایش!)، انتخاب نهایی با
farthest-first در فضای ویژگی، تنوع را هم لحاظ می‌کند.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..featurization import extended_matrix
from ..nn.predictor import EnsemblePropertyPredictor


def committee_uncertainty(
    predictors: Sequence[EnsemblePropertyPredictor], smiles: Sequence[str]
) -> Tuple[np.ndarray, List[int]]:
    """
    امتیاز عدم‌قطعیت هر ساختار: میانگین انحراف‌معیار نرمال‌شده بین اعضای ensemble.

    انحراف‌معیار هر هدف بر انحراف‌معیار آموزشی همان هدف تقسیم می‌شود تا اهداف با
    مقیاس‌های متفاوت (nm در برابر PDI) سهم یکسانی داشته باشند.

    Returns:
        (امتیازها به طول تعداد ساختار معتبر، اندیس ساختارهای معتبر در ورودی)
    """
    if not predictors:
        raise ValueError("حداقل یک پیش‌بین لازم است")
    per_predictor: List[np.ndarray] = []
    kept: Optional[List[int]] = None
    for predictor in predictors:
        members, valid = predictor.member_predictions(smiles)
        if len(predictor.models) < 2:
            raise ValueError("Query-by-Committee به ensemble با حداقل ۲ عضو نیاز دارد")
        spread = members.std(axis=0, ddof=1) / predictor.scaler.std[None, :]
        per_predictor.append(spread.mean(axis=1))
        kept = valid
    return np.mean(per_predictor, axis=0), kept or []


def _farthest_first(features: np.ndarray, scores: np.ndarray, n: int) -> List[int]:
    """انتخاب n نقطه: شروع از پرعدم‌قطعیت‌ترین، سپس دورترین از انتخاب‌شده‌ها."""
    chosen = [int(np.argmax(scores))]
    distances = np.linalg.norm(features - features[chosen[0]], axis=1)
    while len(chosen) < min(n, len(features)):
        # ترکیب فاصله و عدم‌قطعیت؛ از امتیاز صفر جلوگیری می‌کنیم که نقطه تکراری برنگردد
        priority = distances * (scores + 1e-9)
        priority[chosen] = -np.inf
        nxt = int(np.argmax(priority))
        chosen.append(nxt)
        distances = np.minimum(distances, np.linalg.norm(features - features[nxt], axis=1))
    return chosen


def select_samples(
    smiles: Sequence[str],
    predictors: Sequence[EnsemblePropertyPredictor],
    n: int,
    strategy: str = "hybrid",
    candidate_factor: int = 5,
    seed: int = 0,
) -> List[int]:
    """
    انتخاب n ساختار برای آزمایش بعدی.

    Args:
        strategy: ``'hybrid'`` (پیش‌فرض: نیمی تصادفی برای پوشش توزیع + نیمی
            diverse-uncertainty)، ``'diverse_uncertainty'``، ``'uncertainty'`` یا
            ``'random'``. در آزمایش shift شدید توزیع (مدل آموزش‌دیده روی لیپید، داده
            جدید پلیمر)، uncertainty خالص از random بدتر بود (نمونه‌های پرت)؛ بنابراین
            پیش‌فرض ترکیبی است. نگاه کنید به docs/MODEL_VALIDATION.md.
        candidate_factor: در حالت diverse، ابتدا ``n × factor`` پرعدم‌قطعیت‌ترین انتخاب
            و سپس از میان آن‌ها با farthest-first n تا برگزیده می‌شود.

    Returns:
        اندیس‌های انتخاب‌شده در فهرست ورودی ``smiles``.
    """
    if n <= 0:
        raise ValueError("n باید مثبت باشد")
    if strategy not in ("uncertainty", "diverse_uncertainty", "random", "hybrid"):
        raise ValueError(f"strategy ناشناخته: {strategy!r}")

    if strategy == "random":
        _, valid = extended_matrix(smiles)
        rng = np.random.default_rng(seed)
        return [int(i) for i in rng.choice(valid, size=min(n, len(valid)), replace=False)]

    if strategy == "hybrid":
        n_random = n // 2
        _, all_valid = extended_matrix(smiles)
        rng = np.random.default_rng(seed)
        random_pick = [int(i) for i in rng.choice(all_valid, size=min(n_random, len(all_valid)), replace=False)]
        remaining = [i for i in all_valid if i not in set(random_pick)]
        sub = [smiles[i] for i in remaining]
        informed = select_samples(
            sub, predictors, n - len(random_pick), "diverse_uncertainty", candidate_factor, seed
        )
        return random_pick + [remaining[i] for i in informed]

    scores, valid = committee_uncertainty(predictors, smiles)
    if not valid:
        return []
    order = np.argsort(-scores)
    if strategy == "uncertainty":
        return [valid[i] for i in order[:n]]

    top = order[: max(n * candidate_factor, n)]
    features, _ = extended_matrix([smiles[valid[i]] for i in top])
    mean, std = features.mean(axis=0), features.std(axis=0)
    features = (features - mean) / np.where(std < 1e-6, 1.0, std)
    picks = _farthest_first(features, scores[top], n)
    return [valid[int(top[p])] for p in picks]
