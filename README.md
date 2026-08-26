# IPIND² — Intelligent Platform for Integrated Nanocarrier Design

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

## شروع سریع

```bash
pip install -r requirements.txt
python -m ipind2.data_generation.synthetic_data_generator
```
