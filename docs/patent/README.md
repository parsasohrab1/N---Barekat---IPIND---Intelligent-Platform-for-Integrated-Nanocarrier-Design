# Patent material — restricted access

فایل `patentability_analysis.md.gpg` حاوی تحلیل قابلیت ثبت اختراع، استراتژی ادعانامه و مقایسه رقابتی است و **با AES-256 رمزگذاری شده** (`gpg --symmetric --cipher-algo AES256`). نسخه متنی آن هرگز کامیت نمی‌شود (نگاه کنید به `.gitignore` ریشه پروژه).

فقط دارندگان پس‌واژه (مالک/مخترعان اصلی) می‌توانند آن را باز کنند:

```bash
gpg -d docs/patent/patentability_analysis.md.gpg > docs/patent/patentability_analysis.md
```

پس‌واژه از طریق یک کانال جدا (نه در این ریپو) در اختیار افراد مجاز قرار می‌گیرد. سایر اعضای تیم که پس‌واژه ندارند فقط این فایل باینری رمزشده را می‌بینند و محتوای آن برایشان قابل خواندن نیست.

⚠️ پس از باز کردن، فایل متنی `patentability_analysis.md` را کامیت نکنید — `.gitignore` آن را نادیده می‌گیرد اما مراقب باشید به‌صورت دستی force-add نشود.
