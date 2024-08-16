# Stage 1: Build the React app
FROM node:14 AS react-build
WORKDIR /app/frontend

# Copy package files and install dependencies
COPY frontend/package.json frontend/package-lock.json ./
RUN npm install

# Copy the rest of the React app and build it
COPY frontend/ ./
RUN npm run build

# Stage 2: Build the Django app
FROM python:3.9

# Set the working directory
WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Django project files
COPY . .

# Copy the React build output to the Django static files directory
COPY --from=react-build /app/frontend/build /app/static

# Set the STATIC_ROOT environment variable
ENV STATIC_ROOT /app/static

# Install additional Python packages if needed
RUN pip install --no-cache-dir numpy==1.21.5 && \
    python -m spacy download en_core_web_sm
RUN pip install transformers
# Expose the port the app runs on
EXPOSE 8000

# Command to run the app
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
