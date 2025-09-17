FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install uv
RUN pip install uv

ENV PYTHONUNBUFFERED=1

# Copy project files
COPY pyproject.toml ./
COPY main.py ./

# Install dependencies
RUN uv sync --frozen

# Expose port
EXPOSE 8000

# Run the application
CMD ["uv", "run", "python", "main.py"]
