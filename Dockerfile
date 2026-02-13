FROM python:3.12-slim-trixie

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock /app/

RUN uv sync --locked

COPY . .

CMD ["uv", "run", "python", "scripts/final_training_and_eval.py"]
#CMD ["uv","run","python", "scripts/comparing_models.py"]
