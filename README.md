# IPIND — Intelligent Platform for Integrated Nanocarrier Design

پلتفرم یکپارچه طراحی هوشمند نانوحامل‌های دارویی: ترکیب مدل‌های مولد عمیق، شبکه‌های عصبی گرافی، یادگیری تقویتی چندهدفه و شبیه‌سازی دینامیک مولکولی برای کوتاه‌سازی چرخه طراحی نانوحامل از ۳-۵ سال به ۶-۱۲ ماه.

سند کامل الزامات نرم‌افزاری (SRS): [`docs/SRS.md`](docs/SRS.md)
مقایسه فنی با نمونه‌های بین‌المللی (NanoForge، Chemistry42 و ...): [`docs/BENCHMARK.md`](docs/BENCHMARK.md)

## ساختار پروژه

```
docs/                   مستندات (SRS، بنچمارک بین‌المللی، معماری)
docs/patent/            مواد مرتبط با ثبت اختراع — رمزگذاری‌شده، دسترسی محدود (README داخلش را ببینید)
sql/schema.sql           طرح پایگاه داده
src/ipind2/
  generation/             واحد ۱ - تولید ساختار (Conditional VAE/GAN)
  physicochemical/        واحد ۲ - پیش‌بینی فیزیکوشیمیایی (Multi-Task GNN)
  biological/              واحد ۳ - پیش‌بینی زیستی (Multi-Task Transformer/GNN)
  optimization/            واحد ۴ - بهینه‌سازی چندهدفه (Pareto-Guided RL)
  md_simulation/           واحد ۵ - شبیه‌سازی دینامیک مولکولی (GROMACS/OpenMM)
  active_learning/         واحد ۶ - یادگیری فعال و بازخورد آزمایشگاهی
  interpretability/        واحد ۷ - تفسیرپذیری (Attention + SHAP/LIME)
  nlp_interface/           واحد ۸ - رابط پرس‌وجوی زبان طبیعی
  lab_automation/          واحد ۹ - یکپارچگی با آزمایشگاه خودکار (lab-in-the-loop)
  benchmarking/            واحد ۱۰ - بنچمارک داخلی مستمر در برابر دیتاست‌های عمومی
  database/                لایه داده
  api/                     لایه رابط کاربری/API
  data_generation/         تولیدکننده داده‌های سنتتیک برای آموزش مدل‌ها
tests/                    تست‌ها
```

## وضعیت

| | |
|---|---|
| **TRL (محاسبه‌شده از شواهد)** | **۴** — TRL ۵ نیازمند داده/MD/استقرار واقعی است؛ [`docs/TRL_ASSESSMENT.md`](docs/TRL_ASSESSMENT.md) دقیقاً می‌گوید چه چیزی کم است |
| واحدهای ۱–۱۰ | پیاده‌سازی و آزموده‌شده (۳۲۷ تست)؛ MD واقعی فقط آداپتور |
| دقت | روی **داده سنتتیک** — [`docs/MODEL_VALIDATION.md`](docs/MODEL_VALIDATION.md) |
| معماری | [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) · عملیات: [`docs/OPERATIONS.md`](docs/OPERATIONS.md) · FAIR: [`docs/FAIR_MIRIBEL.md`](docs/FAIR_MIRIBEL.md) |

## شروع سریع

```bash
pip install -r requirements-dev.txt
python -m ipind2.training.train --profile smoke --out models/dev          # ~۳۰ ثانیه؛ برای دقت واقعی: --profile release
export IPIND_JWT_SECRET="$(python -c 'import secrets;print(secrets.token_urlsafe(48))')"
export IPIND_ENCRYPTION_KEY="$(python -c 'from ipind2.security import generate_key;print(generate_key())')"
export IPIND_MODEL_DIR=models/dev
python -m ipind2.api.manage create-user admin --role admin                 # سپس تأیید TOTP (docs/OPERATIONS.md)
python -m ipind2.api.manage serve                                          # داشبورد: http://127.0.0.1:8000
```

استفاده برنامه‌نویسی:

```python
from ipind2.training import ModelBundle
from ipind2.pipeline import DesignPipeline
result = DesignPipeline(ModelBundle.load("models/dev")).design("نانوحامل لیپیدی برای تومور، اندازه بین ۸۰ تا ۱۲۰ نانومتر")
print(result.final_candidates[0]["predictions"], result.warnings)
```

## آزمون

```bash
pytest -q     # ~۸ دقیقه روی CPU؛ مدل‌های smoke یک‌بار آموزش می‌بینند
```

> پیام‌های `joblib ... wmic` و گاه `access violation` (faulthandler) روی Windows بی‌ضررند و بر نتایج اثر ندارند.
