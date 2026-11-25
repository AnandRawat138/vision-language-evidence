# CPU image for tests, the review API and the dashboard (no model weights).
# NOTE: not built during this study (Docker was not used on the development machine).
FROM python:3.12-slim
WORKDIR /app
COPY requirements/requirements.txt requirements/requirements.txt
RUN pip install --no-cache-dir torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements/requirements.txt
COPY src src
COPY configs configs
COPY tests tests
COPY data/manifests data/manifests
COPY data/annotations data/annotations
ENV PYTHONPATH=/app/src
CMD ["python", "-m", "uvicorn", "vle.api.app:app", "--host", "0.0.0.0", "--port", "8008"]
