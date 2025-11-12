# ---- Base Image ----
FROM python:3.11-slim

# ---- Set Environment ----
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ---- Set Working Directory ----
WORKDIR /app

# ---- Copy Only Requirements First (for caching) ----
COPY requirements.txt .

# ---- Install Dependencies ----
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install -r requirements.txt

# ---- Update PATH to use venv ----
ENV PATH="/opt/venv/bin:$PATH"

# ---- Copy Application Code ----
COPY ./code .

# ---- Expose Streamlit Port ----
EXPOSE 8501

# ---- Default Command ----
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

