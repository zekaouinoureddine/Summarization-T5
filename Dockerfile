FROM python:3.10.9
WORKDIR /code
COPY . /code
RUN pip install --no-cache-dir -r /code/requirements.txt
EXPOSE 8000
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]