# 1. Use an official Python runtime as a parent image
FROM python:3.11-slim

# 2. Set the working directory in the container
WORKDIR /app

# 3. Install uv, the package manager
RUN pip install uv

# 4. Copy dependency definition files
COPY pyproject.toml uv.lock ./

# 5. Install dependencies using the lock file for reproducibility
RUN uv sync --frozen

# 6. Copy the application source code, including models
COPY src/ ./src/

# 7. Expose the port the app will run on
EXPOSE 8000

# 8. Define the command to run the application
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
