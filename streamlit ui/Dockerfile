FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt first 
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port if needed
EXPOSE 8501

# Command to run your Python Streamlit Application
CMD ["streamlit", "run", "app.py"]