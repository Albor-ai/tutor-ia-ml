# 1. Use a stable Python runtime as a parent image
FROM python:3.11-slim

# 2. Set the working directory in the container
WORKDIR /app

# 3. Copy the requirements file
COPY requirements.txt .

# 4. Install dependencies using pip
# --no-cache-dir reduces image size, --upgrade pip is a good practice
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 5. Copy the application source code
COPY src/ ./src/
COPY main.py ./main.py

# 6. Expose the port the app will run on
EXPOSE 8000

# 7. Define the command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
