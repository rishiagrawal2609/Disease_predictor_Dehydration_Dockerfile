FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy the required files
COPY ./app/requirements.txt ./
COPY ./app/main.py ./
COPY ./app/model.h5 ./

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the FastAPI port
EXPOSE 8000

# Command to run the FastAPI application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]