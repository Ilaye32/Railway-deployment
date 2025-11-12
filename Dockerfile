# ---- Base Image ----
FROM python:3.11-slim

# ---- Set Environment ----
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ---- Create Virtual Environment ----
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# ---- Set Working Directory ----
WORKDIR /app

# ---- Copy Requirements and Install ----
COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip && \
    pip install -r /tmp/requirements.txt

# ---- Copy Application Code ----
COPY ./code .

# ---- Default Command ----
# Streamlit runs your RAG interface
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
