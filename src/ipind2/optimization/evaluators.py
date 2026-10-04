"""
ارزیاب‌های تابع هدف برای بهینه‌ساز: پیش‌بین‌های واحد ۲/۳ و oracle مرجع.

* ``PredictorObjective`` — ارزیاب تولیدی: خروجی GNN فیزیکوشیمیایی + Transformer زیستی.
* ``oracle_objective`` — قوانین حقیقت زمینی داده سنتتیک بدون نویز؛ فقط برای
  بنچمارک بهینه‌ساز (آیا PG-RL از جست‌وجوی تصادفی بهتر است؟) و تست‌های سریع.
"""

from typing import Sequence

import numpy as np
import pandas as pd

from ..data_generation.properties import biological_truth, physicochemical_truth
from ..featurization import extended_dict, parse_smiles
from .objectives import OBJECTIVE_COLUMNS


def _scaffold_of(smiles: str) -> str:
    """حدس نوع اسکلت از روی عناصر/گروه‌ها (برای ارزیاب oracle که قالب را نمی‌بیند)."""
    mol = parse_smiles(smiles)
    symbols = {a.GetSymbol() for a in mol.GetAtoms()}
    if symbols & {"Au", "Fe", "Si", "Zn", "Mn", "Gd"}:
        return "metal"
    features = extended_dict(mol)
    # پلیمرها: تعداد زیاد پیوند استری/اتری تکرارشونده؛ لیپیدها: دم‌های بلند بدون تکرار
    repeat_signal = features["n_ether_ch2"] + features["n_ester"] + features["n_amide"]
    return "polymer" if repeat_signal >= 4 else "lipid"


def oracle_objective(smiles_list: Sequence[str]) -> pd.DataFrame:
    """ارزیاب بدون نویز مبتنی بر قوانین ساختار→خاصیت (برای بنچمارک/تست)."""
    rng = np.random.default_rng(0)
    rows = []
    for smiles in smiles_list:
        mol = parse_smiles(smiles)
        features = extended_dict(mol)
        scaffold = _scaffold_of(smiles)
        physico = physicochemical_truth(scaffold, features, rng, noise=0.0)
        bio = biological_truth(scaffold, features, physico, rng, noise=0.0)
        rows.append({**{f"phys_{k}": v for k, v in physico.items()}, **{f"bio_{k}": v for k, v in bio.items()}})
    return pd.DataFrame(rows)


class PredictorObjective:
    """
    تابع هدف مبتنی بر پیش‌بین‌های آموزش‌دیده.

    ``__call__(smiles) -> DataFrame`` با ستون‌های ``OBJECTIVE_COLUMNS``؛ ردیف‌های
    SMILES نامعتبر حذف نمی‌شوند بلکه با مقدار بد پر می‌شوند تا ترتیب حفظ شود.
    """

    def __init__(self, physico_predictor, bio_predictor, uncertainty_penalty: float = 0.0):
        self.physico = physico_predictor
        self.bio = bio_predictor
        self.uncertainty_penalty = uncertainty_penalty

    def __call__(self, smiles_list: Sequence[str]) -> pd.DataFrame:
        smiles_list = list(smiles_list)
        phys_mean, phys_std, phys_kept = self.physico.predict_with_uncertainty(smiles_list)
        bio_mean, bio_std, bio_kept = self.bio.predict_with_uncertainty(smiles_list)
        frame = pd.DataFrame(index=range(len(smiles_list)), columns=list(OBJECTIVE_COLUMNS.values()), dtype=float)
        frame[:] = np.nan
        for column in OBJECTIVE_COLUMNS.values():
            source_mean, source_std = (
                (phys_mean, phys_std) if column.startswith("phys_") else (bio_mean, bio_std)
            )
            values = source_mean[column]
            # بدبینی در برابر عدم‌قطعیت: از اهداف حداکثرسازی، k·σ کم می‌شود. اندازه
            # مستثنی است چون هدفش «نزدیکی به بازه» است نه حداکثرسازی خطی.
            if self.uncertainty_penalty and column != OBJECTIVE_COLUMNS["size_fit"]:
                values = values - self.uncertainty_penalty * source_std[column]
            frame.loc[values.index, column] = values.to_numpy()
        return frame.fillna(-1e6)
