# 1. Use an official Python runtime as a parent image
FROM python:3.11-slim

# 2. Set the working directory in the container
WORKDIR /app

# 3. Install uv, the package manager
RUN pip install uv

# 4. Copy dependency definition files
COPY pyproject.toml uv.lock ./

# 5. Install build dependencies required for packages like pyarrow
RUN apt-get update && apt-get install -y cmake build-essential

# 6. Create a virtual environment with a specific python version
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 7. Install dependencies using the lock file for reproducibility
RUN uv sync --frozen

# 8. Copy the application source code, including models
COPY src/ ./src/

# 9. Expose the port the app will run on
EXPOSE 8000

# 10. Define the command to run the application
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
