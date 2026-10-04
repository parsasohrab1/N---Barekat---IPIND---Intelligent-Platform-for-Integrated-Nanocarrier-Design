"""
اجزای مشترک شبکه‌های عصبی واحدهای ۲ و ۳ (پیاده‌سازی با torch خالص).

عمداً از ``torch-geometric`` استفاده نمی‌شود: گراف‌های نانوحامل کوچک‌اند (≤۱۲۸ اتم) و
message passing متراکم (dense) روی آن‌ها هم سریع است و هم نصب پلتفرم را روی CPU/GPU
بدون wheel اختصاصی ممکن می‌کند (نگاه کنید به docs/ARCHITECTURE.md).
"""

from .layers import AttentionReadout, DenseMessagePassing, masked_softmax
from .training import EarlyStopping, TargetScaler, train_regressor

__all__ = [
    "masked_softmax",
    "DenseMessagePassing",
    "AttentionReadout",
    "TargetScaler",
    "EarlyStopping",
    "train_regressor",
]
