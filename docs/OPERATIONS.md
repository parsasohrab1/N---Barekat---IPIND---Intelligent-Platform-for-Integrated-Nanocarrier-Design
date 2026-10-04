# راهنمای استقرار و عملیات

## راه‌اندازی اولیه

```bash
pip install -r requirements.txt            # + requirements-optional.txt برای PostgreSQL/Redis/OpenMM
python -m ipind2.training.train --profile standard --out models/v1     # ~۲۰–۳۰ دقیقه روی CPU
python -m ipind2.training.validate --model-dir models/v1 --out docs/validation_report.json
```

### رازها (هرگز در git یا image نباشند)

```bash
export IPIND_JWT_SECRET="$(python -c 'import secrets;print(secrets.token_urlsafe(48))')"
export IPIND_ENCRYPTION_KEY="$(python -c 'from ipind2.security import generate_key;print(generate_key())')"
export IPIND_DATABASE_URL="postgresql+psycopg2://ipind:***@db:5432/ipind2"   # پیش‌فرض: SQLite محلی (فقط توسعه)
export IPIND_MODEL_DIR=models/v1
```

- **کلید رمزنگاری را گم نکنید:** راز TOTP کاربران و پشتیبان‌ها با آن رمز شده‌اند؛ بدون کلید قابل بازیابی نیستند. در secret manager نگه دارید و چرخش را برنامه‌ریزی کنید (قالب داده بایت نسخه دارد).
- بدون `IPIND_JWT_SECRET` (≥ ۳۲ نویسه) ورود کار نمی‌کند و بدون `IPIND_ENCRYPTION_KEY` ساخت کاربر شکست می‌خورد (fail-closed).

### اولین کاربر ادمین

```bash
python -m ipind2.api.manage create-user admin --role admin
# URI خروجی را در برنامه احراز هویت اسکن کنید، سپس:
curl -X POST https://HOST/auth/enroll/confirm -d '{"username":"admin","password":"…","code":"123456"}'
```
تا تأیید TOTP ورود ممکن نیست (SEC-01 برای همه کاربران).

### اجرا

```bash
python -m ipind2.api.manage serve --host 0.0.0.0 --port 8443 --certfile c.pem --keyfile k.pem
# یا docker compose up (nginx با TLS 1.3 + PostgreSQL + Redis؛ deploy/nginx.conf)
```
اجرا روی رابط غیرمحلی بدون TLS عمداً رد می‌شود (SEC-03).

## پشتیبان‌گیری روزانه (SEC-06)

```bash
# cron: 02:00 هر شب
0 2 * * * python -c "from ipind2.security import backup_database; backup_database('$IPIND_DATABASE_URL', '/backups', retention=14)"
```
خروجی `ipind2-<زمان>.dump.enc` است (AES-256-GCM)؛ بازیابی با `decrypt_file` و سپس `pg_restore`/کپی SQLite. **یک بازیابی آزمایشی را فصلی تمرین کنید** — پشتیبانی که بازیابی‌اش آزموده نشده، پشتیبان نیست. (تست خودکار بازیابی SQLite موجود است؛ مسیر PostgreSQL به `pg_dump` وابسته و در CI آزموده نمی‌شود.)

## ارتقای مدل (FR-12)

۱. آموزش نسخه جدید ⇒ ۲. `ipind2-benchmark --model-dir models/new --history benchmarks/history.json` ⇒ ۳. فقط در صورت PASS جایگزین `models/current`.
دروازه PASS یعنی: افت نسبی RMSE ≤ ۵٪، افت R² ≤ ۰٫۰۲ نسبت به نسخه قبل **و** برقراری NFR-01/02/03. اگر قوانین تولید داده/مرجع تغییر کند، اثرانگشت مرجع عوض و تاریخچه از نو شروع می‌شود.

## بازخورد آزمایشگاهی (FR-06/FR-11)

نتایج را با `POST /lab/results` (یا `lab_automation` CSV/REST) ثبت کنید؛ `POST /active-learning/retrain` پس از ۱۰ نتیجه جدید (یا `force`) مدل را fine-tune می‌کند و نتایج را «مصرف‌شده» علامت می‌زند. **هشدار:** replay فعلی داده سنتتیک است؛ در تولید داده آموزش واقعی را به‌عنوان `replay` بدهید تا فراموشی فاجعه‌بار کم شود.

## محدودیت‌های عملیاتی

- یک پردازه API، کار طراحی را روی thread-pool (پیش‌فرض ۲) اجرا می‌کند؛ برای هم‌زمانی بیشتر چند replica با صف کار (Celery/RQ) اضافه کنید — کد فعلی صف توزیع‌شده ندارد.
- محدودکننده نرخ (rate limit) فقط قفل حساب پس از ۵ ورود ناموفق است؛ محدودیت IP را در nginx/WAF اعمال کنید.
- NFR-07 (دسترس‌پذیری ۹۹٫۹٪) به زیرساخت (چند replica، DB با replication، health-check) بستگی دارد و با کد قابل اثبات نیست.
