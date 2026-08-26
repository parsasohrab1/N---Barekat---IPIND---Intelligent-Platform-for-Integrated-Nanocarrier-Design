# بنچمارک بین‌المللی: IPIND² در برابر نمونه‌های خارجی

این سند، پلتفرم IPIND² (مطابق [`docs/SRS.md`](SRS.md)) را با مهم‌ترین نمونه‌های صنعتی و پژوهشی بین‌المللی در حوزه طراحی هوشمند نانوحامل/نانوذره دارویی مقایسه می‌کند. نتیجه این بنچمارک به شکل الزامات جدید (FR-09 تا FR-13، NFR-09 و NFR-10) به SRS اضافه شده است.

⚠️ این یک تحلیل فنی/رقابتی عمومی است، نه یک سند حقوقی یا ادعای ثبت اختراع — برای تحلیل قابلیت ثبت اختراع به `docs/patent/` (رمزگذاری‌شده) مراجعه کنید.

## نمونه‌های بررسی‌شده

| نمونه | ماهیت | ویژگی کلیدی |
|---|---|---|
| **NanoForge (METiS TechBio، چین، سپتامبر ۲۰۲۵)** | پلتفرم صنعتی end-to-end | شیمی کوانتومی + MD، غربالگری آزمایشگاهی اختصاصی با توان بالا، عامل‌های AI، کتابخانه >۱۰ میلیون ساختار لیپیدی، ورودی زبان طبیعی |
| **Chemistry42 (Insilico Medicine)** | پلتفرم صنعتی برای مولکول‌های کوچک (نه اختصاصاً نانوذره) | ۴۲ الگوریتم مولد در قالب چندعامل RL، به‌کارگیری توسط ۲۰+ شرکت دارویی (از جمله Merck KGaA) |
| **آزمایشگاه خودکار دانشگاه تورنتو** | نمونه پژوهشی self-driving lab برای LNP | حلقه کاملاً بسته با رباتیک سنتز/آزمایش واقعی، نه فقط شبیه‌سازی |
| **AGILE** | پژوهشی | ترکیب DL + شیمی ترکیبی برای غربالگری این‌سیلیکوی لیپید یونیزه‌شونده |
| **COMET / دیتاست LANCE، LANTERN** | پژوهشی/بنچمارک صنعتی نوظهور | پیش‌بینی چندوظیفه‌ای Transformer برای LNP + چارچوب بنچمارکینگ استاندارد صنعت |

## جدول مقایسه فنی

| بُعد فنی | IPIND² (طبق SRS فعلی) | NanoForge | Chemistry42 | Self-driving lab (تورنتو) |
|---|---|---|---|---|
| انواع اسکلت پوشش‌داده‌شده | لیپیدی + پلیمری + فلزی (۳ نوع) | عمدتاً لیپیدی/اسید نوکلئیک | مولکول کوچک عمومی | فقط LNP |
| تعداد ویژگی پیش‌بینی هم‌زمان | ≥۷ (فیزیکوشیمیایی + زیستی) | نامشخص/عمومی | N/A | محدود (چند ویژگی ترانسفکشن) |
| روش بهینه‌سازی | Pareto-Guided RL | عامل‌های AI (جزئیات افشا نشده) | چندعامل RL با ۴۲ مدل مولد (تنوع بالا) | بهینه‌سازی بیزی/تجربی |
| یکپارچگی با آزمایشگاه فیزیکی | فقط ورود دستی داده آزمایشگاهی | غربالگری اختصاصی high-throughput | برون‌سپاری به شرکای دارویی | کاملاً رباتیک، حلقه بسته زنده |
| مقیاس کتابخانه seed اولیه | تعریف‌نشده (فقط ≥۵۰k داده آموزشی) | >۱۰ میلیون ساختار | N/A | کوچک، متمرکز بر LNP |
| سطح محاسبات فیزیک/شیمی | MD کلاسیک (GROMACS/OpenMM) | شیمی کوانتومی + MD | N/A | تجربی |
| تفسیرپذیری رسمی (XAI) | نبود (تا پیش از این بنچمارک) | نامشخص | نامشخص | ندارد |
| رابط کاربری | داشبورد فرم‌محور | پرس‌وجوی زبان طبیعی | پلتفرم یکپارچه سازمانی | رابط آزمایشگاهی |
| بنچمارک مستمر در برابر دیتاست عمومی | نبود (تا پیش از این بنچمارک) | نامشخص | دارای سابقه اعتبارسنجی گسترده صنعتی | خودِ داده تجربی، منبع بنچمارک است |
| استانداردسازی داده (FAIR/MIRIBEL) | نبود | نامشخص | نامشخص | نامشخص |

## نتیجه‌گیری: مزیت‌های فنی که IPIND² کم داشت و اضافه شد

بر اساس این مقایسه، پنج الزام فنی جدید به SRS اضافه شد تا فاصله با پیشروترین نمونه‌های بین‌المللی پر شود:

1. **FR-09 — تفسیرپذیری رسمی (SHAP/LIME/Attention)**: اکثر رقبا (از جمله NanoForge) این را به‌صراحت اعلام نکرده‌اند؛ رسمی‌کردنش در IPIND² یک مزیت رقابتی واقعی است، نه فقط جبران عقب‌ماندگی.
2. **FR-10 — رابط پرس‌وجوی زبان طبیعی**: برای همتراز شدن با تجربه کاربری NanoForge.
3. **FR-11 — یکپارچگی با آزمایشگاه خودکار (lab-in-the-loop)**: مهم‌ترین شکاف — رقبای برتر (NanoForge، self-driving lab تورنتو) حلقه بازخورد را با رباتیک واقعی می‌بندند، نه ورود دستی داده.
4. **FR-12 — بنچمارک داخلی مستمر در برابر دیتاست‌های عمومی (LNP-622، LANCE)**: برای اثبات شفاف رقابتی‌بودن دقت مدل‌ها، به‌ویژه در برابر ادعاهای دقت بالای Chemistry42 و LANTERN.
5. **FR-13 / NFR-10 — استانداردسازی داده (FAIR/MIRIBEL) و کتابخانه seed بزرگ‌مقیاس (≥۱ میلیون ساختار)**: کاهش cold-start مدل مولد و افزایش تعامل‌پذیری با شرکا، در راستای روند صنعت.
6. **NFR-09 — دقت محاسبات سطح شبه‌کوانتومی (اختیاری، مکمل MD کلاسیک)**: برای نزدیک‌شدن به دقت محاسباتی NanoForge که از شیمی کوانتومی در کنار MD استفاده می‌کند.

## منابع

- [METiS Launches World's First AI-Driven Nano-Delivery Platform NanoForge](https://www.metistechbio.com/en/qyxw/180.html)
- [Chemistry42: An AI-Driven Platform for Molecular Design and Optimization — J. Chem. Inf. Model.](https://pubs.acs.org/doi/10.1021/acs.jcim.2c01191)
- [Merck KGaA to Deploy Insilico Medicine's Chemistry42](https://www.drugdiscoveryonline.com/doc/merck-kgaa-darmstadt-germany-chemistry-ai-platform-generative-chemistry-0001)
- [AI Self-Driving Lab Identifies New Lipid Nanoparticles for mRNA Therapeutics](https://www.labmanager.com/ai-powered-self-driving-lab-accelerates-discovery-of-mrna-delivery-materials-35042)
- [Machine Learning-guided Lipid Nanoparticle Design for mRNA Delivery (arXiv 2308.01402)](https://arxiv.org/abs/2308.01402)
- [A Machine Learning Benchmarking Framework for LNP Transfection Efficiency (LANTERN, arXiv 2507.03209)](https://arxiv.org/html/2507.03209)
- [Designing lipid nanoparticles using a transformer-based neural network — Nature Nanotechnology](https://www.nature.com/articles/s41565-025-01975-4)
- [Explainable AI for Material Design and Engineering Applications](https://onlinelibrary.wiley.com/doi/10.1002/msd2.70017)
- [Artificial Intelligence-Driven Development and Characterization of Nanomedicine — BioNanoScience](https://link.springer.com/article/10.1007/s12668-026-02476-x)
