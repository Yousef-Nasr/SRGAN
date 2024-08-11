# Use the official Python 3.12 image as the base image
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file to the working directory
COPY  requirements.txt .


# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# RUN pip install --no-cache-dir matplotlib

# RUN pip install --no-cache-dir python-multipart

# # Copy the content of the local src directory to the working directory
# COPY ../src/ .

# # Copy the rest of the application code to the working directory
# COPY . .

# Copy only the src folder to the working directory
COPY src /app/src

COPY api /app/api

COPY SRGAN.pth /app/SRGAN.pth
# Expose the port FastAPI will run on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

