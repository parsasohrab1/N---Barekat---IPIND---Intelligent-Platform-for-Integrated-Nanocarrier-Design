FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN useradd --create-home --uid 10001 ipind
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY pyproject.toml README.md ./
COPY src ./src
COPY sql ./sql
RUN pip install --no-deps .
USER ipind

# راز‌ها (IPIND_JWT_SECRET، IPIND_ENCRYPTION_KEY) هرگز در image نیستند؛ هنگام اجرا تزریق می‌شوند.
ENV IPIND_MODEL_DIR=/models IPIND_DATABASE_URL=sqlite:////data/ipind2.db
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["python", "-m", "ipind2.api.manage", "serve", "--host", "0.0.0.0", "--port", "8000"]
