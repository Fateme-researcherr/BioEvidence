FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir "uvicorn[standard]>=0.27,<1.0"

COPY . .

EXPOSE 8000

# Default command runs the api service (api/main.py -> app).
# The worker service overrides this via docker-compose's
# `command: python -m orchestration.worker`.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
