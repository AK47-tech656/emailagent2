FROM python:3.11-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app
COPY . .

# Install as a system package using uv
RUN uv pip install --system --no-cache -e .

EXPOSE 7860
CMD ["python", "-m", "uvicorn", "emailagent.app:app", "--host", "0.0.0.0", "--port", "7860"]