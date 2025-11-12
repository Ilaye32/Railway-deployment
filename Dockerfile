# ---- Base Image ----
FROM python:3.11-slim

# ---- Set Environment ----
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# ---- Set Working Directory ----
WORKDIR /app

# ---- Copy Requirements and Install (DO THIS BEFORE COPYING CODE) ----
COPY requirements.txt .

# Install dependencies with optimizations
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ---- Copy Application Code (AFTER installing dependencies for better caching) ----
COPY ./code .

# ---- Expose Port ----
EXPOSE 8501

# ---- Default Command ----
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
