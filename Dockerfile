FROM python:3.10-slim
WORKDIR /workspace
COPY . /workspace
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 5001
CMD ["python3", "scripts/launch_assistant.py"]
